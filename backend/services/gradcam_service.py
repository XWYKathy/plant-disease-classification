"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for the EfficientNetB0
transfer learning model.

Model architecture (from src/train_transfer.py):
    Input (224, 224, 3)  [raw, 0–255]
    → Sequential(RandomFlip, RandomRotation)   [named "sequential"]
    → efficientnet.preprocess_input            [Lambda, inside functional graph]
    → EfficientNetB0 backbone                  [nested Keras Model, include_top=False]
    → GlobalAveragePooling2D
    → Dense(128, relu)
    → Dropout(0.3)
    → Dense(num_classes, softmax)

Grad-CAM approach:
  1. Decompose the forward pass manually so we can intercept gradients at the
     last Conv2D layer inside the EfficientNet backbone.
  2. Compute the gradient of the target class score w.r.t. that feature map.
  3. Weight channels by their gradient mean and sum → unnormalised heatmap.
  4. ReLU + normalise → [0, 1] heatmap.
"""
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")   # Non-interactive backend — no display needed in a server

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from services.model_service import get_model


# ── Backbone discovery ────────────────────────────────────────────────────────

def _find_backbone_and_last_conv(
    model: tf.keras.Model,
) -> tuple[tf.keras.Model, str]:
    """
    Walk the top-level layers to find the first nested Keras Model (the
    EfficientNet backbone) and the name of its last Conv2D layer.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return layer, sublayer.name
    raise ValueError(
        "Could not find a nested Keras Model with Conv2D layers.  "
        "Check that the correct model file is loaded."
    )


# ── Grad-CAM computation ──────────────────────────────────────────────────────
#计算热力图
def compute_gradcam(
    image_batch: tf.Tensor, #输入图像
    pred_index: int | None = None, #预测类别索引
) -> tuple[np.ndarray, int]: #返回热力图和预测类别索引
    """
    Compute a Grad-CAM heatmap for one image.

    Args:
        image_batch: shape (1, H, W, 3), float32, values in [0, 255].
        pred_index:  class index to explain.  Uses argmax prediction if None.

    Returns:
        heatmap:    (h, w) float32 array, normalised to [0, 1].
                    (h, w) is the spatial resolution of the last conv layer,
                    NOT necessarily the input resolution — use render_gradcam_overlay
                    to resize it back.
        pred_index: the class index that was explained.
    """
    model = get_model()

    backbone, last_conv_name = _find_backbone_and_last_conv(model)
    last_conv_layer = backbone.get_layer(last_conv_name)

    # Sub-model that exposes both the last conv feature map and the backbone output.
    # Its input is the *preprocessed* tensor (after efficientnet.preprocess_input).
    backbone_grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[last_conv_layer.output, backbone.output],
    )

    # Retrieve head layers by name (names are stable after model.save / load_model)
    aug_layer        = model.get_layer("sequential")
    gap_layer        = model.get_layer("global_average_pooling2d")
    dense_layer      = model.get_layer("dense")
    dropout_layer    = model.get_layer("dropout")
    classifier_layer = model.get_layer("dense_1")

    with tf.GradientTape() as tape:
        # ── Forward pass, decomposed ──────────────────────────────────────────
        x = aug_layer(image_batch, training=False)
        x = tf.keras.applications.efficientnet.preprocess_input(x)

        # tape watches conv_outputs so we can differentiate through it
        conv_outputs, backbone_outputs = backbone_grad_model(x, training=False)
        tape.watch(conv_outputs)

        x = gap_layer(backbone_outputs)
        x = dense_layer(x)
        x = dropout_layer(x, training=False)
        preds = classifier_layer(x)

        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]).numpy())

        # Score for the target class (scalar)
        class_score = preds[:, pred_index]

    # ── Gradient → importance weights ────────────────────────────────────────
    grads = tape.gradient(class_score, conv_outputs)    # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,) — mean over spatial dims

    # ── Weighted combination of feature maps ──────────────────────────────────
    conv_map = conv_outputs[0]                            # (h, w, C)
    heatmap = conv_map @ pooled_grads[..., tf.newaxis]    # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                         # (h, w)

    # ReLU: keep only features that positively contribute to the class score
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalise to [0, 1]
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), pred_index


# ── Overlay rendering ─────────────────────────────────────────────────────────
#将热力图叠加到原始图像上
def render_gradcam_overlay(
    original_image: np.ndarray, #原始图像
    heatmap: np.ndarray, #热力图
    alpha: float = 0.4, #叠加权重
) -> tuple[np.ndarray, np.ndarray]: #返回叠加后的图像和原始图像
    """
    Resize the heatmap to the original image size and blend them together.

    Args:
        original_image: (H, W, 3) array, uint8 or float32 [0, 255].
        heatmap:        (h, w) float32 [0, 1] from compute_gradcam().
        alpha:          blend weight of the colorized heatmap (0 = original only,
                        1 = heatmap only).

    Returns:
        heatmap_rgb: (H, W, 3) uint8 — jet-colorized heatmap at full resolution.
        overlay:     (H, W, 3) uint8 — heatmap blended over original image.
    """
    img = np.clip(original_image, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]

    # Resize heatmap from (h_conv, w_conv) → (H, W)
    heatmap_uint8 = np.uint8(255 * heatmap)                      # (h_conv, w_conv)
    heatmap_resized = (
        tf.image.resize(heatmap_uint8[..., np.newaxis], (h, w))  # (H, W, 1)
        .numpy()
        .squeeze()
        .astype(np.uint8)
    )

    # Apply jet colormap
    colormap = plt.get_cmap("jet")
    colored = np.uint8(255 * colormap(heatmap_resized / 255.0)[:, :, :3])  # (H, W, 3)

    # Alpha blend
    overlay = np.uint8(np.clip(colored * alpha + img * (1 - alpha), 0, 255))

    return colored, overlay
