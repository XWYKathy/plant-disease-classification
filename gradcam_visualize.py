import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# =========================
# Config
# =========================
MODEL_PATH = "outputs/checkpoints/efficientnet_transfer.keras"
IMG_PATH = "data/test/Tomato_Late_blight/0ab1cab4-a0c9-4323-9a64-cdafa4342a9b___GHLB2 Leaf 8918.JPG"   # 改成你自己的图片路径
IMG_SIZE = (224, 224)

# =========================
# Load model
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Loaded model from: {MODEL_PATH}")

# =========================
# Helper: load image
# =========================
def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# Find the actual backbone model and last Conv2D layer
# =========================
print("\nTop-level layers:")
for layer in model.layers:
    print(layer.name, type(layer))

backbone = None
last_conv_layer_name = None

for candidate in model.layers:
    if isinstance(candidate, tf.keras.Model):
        for sublayer in reversed(candidate.layers):
            if isinstance(sublayer, tf.keras.layers.Conv2D):
                backbone = candidate
                last_conv_layer_name = sublayer.name
                break
    if backbone is not None:
        break

if backbone is None or last_conv_layer_name is None:
    raise ValueError("Could not find a nested backbone model with Conv2D layers.")

print("Backbone model:", backbone.name)
print("Last conv layer:", last_conv_layer_name)

# =========================
# Build Grad-CAM backbone model
# =========================
last_conv_layer = backbone.get_layer(last_conv_layer_name)

# This model works INSIDE EfficientNet only:
# input: preprocessed tensor
# outputs: last conv feature map + backbone output
backbone_grad_model = tf.keras.models.Model(
    inputs=backbone.input,
    outputs=[last_conv_layer.output, backbone.output]
)

# Grab the classification head layers from the full model
aug_layer = model.get_layer("sequential")
gap_layer = model.get_layer("global_average_pooling2d")
dense_layer = model.get_layer("dense")
dropout_layer = model.get_layer("dropout")
classifier_layer = model.get_layer("dense_1")

# =========================
# Compute Grad-CAM heatmap
# =========================
def make_gradcam_heatmap(img_array, pred_index=None):
    # Forward pass through augmentation + preprocessing + backbone + classifier head
    with tf.GradientTape() as tape:
        # Outer model preprocessing path
        x = aug_layer(img_array, training=False)
        x = tf.keras.applications.efficientnet.preprocess_input(x)#把原始图像变成 EfficientNet 需要的输入格式

        # Backbone forward pass
        conv_outputs, backbone_outputs = backbone_grad_model(x, training=False)

        # Classification head forward pass
        x = gap_layer(backbone_outputs)
        x = dense_layer(x)
        x = dropout_layer(x, training=False)
        preds = classifier_layer(x)

        #如果你没指定要解释哪个类别，就默认解释：模型当前预测出来的那个类别
        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        class_channel = preds[:, pred_index]

    # Compute gradients of class score w.r.t. last conv feature map
    grads = tape.gradient(class_channel, conv_outputs) #求梯度

    # Global average pooling on gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) #对梯度做全局平均


    #加权求和形成热力图
    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weight channels by importance
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Keep positive contributions only
    heatmap = tf.maximum(heatmap, 0)

    # Normalize to [0, 1]
    #归一化
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy(), int(pred_index.numpy()), preds.numpy()

# =========================
# Overlay heatmap on original image
# =========================
def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    # Load original image
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    # Resize heatmap to match image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (img.shape[0], img.shape[1])
    ).numpy().astype("uint8")
    heatmap = np.squeeze(heatmap)

    # Colormap
    colormap = plt.get_cmap("jet")
    colored_heatmap = colormap(heatmap / 255.0)[:, :, :3]
    colored_heatmap = np.uint8(255 * colored_heatmap)

    # Superimpose
    superimposed_img = colored_heatmap * alpha + img
    superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))

    return img.astype("uint8"), heatmap, superimposed_img

# =========================
# Run
# =========================
img_array = get_img_array(IMG_PATH, IMG_SIZE)

# IMPORTANT:
# Your training model already included EfficientNet preprocessing inside the model,
# so here we feed the raw resized image directly.
heatmap, pred_index, preds = make_gradcam_heatmap(img_array)

original_img, heatmap_img, overlay_img = save_and_display_gradcam(IMG_PATH, heatmap)

print("Predicted class index:", pred_index)
print("Prediction probabilities:", preds[0])

# =========================
# Show results
# =========================
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(original_img.astype("uint8"))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_img, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay_img)
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()