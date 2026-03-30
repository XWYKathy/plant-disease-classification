import os
import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
SEED = 123
AUTOTUNE = tf.data.AUTOTUNE

MODEL_SAVE_PATH = "outputs/checkpoints/efficientnet_transfer.keras"

# =========================
# Load datasets
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# =========================
# Data augmentation
# =========================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
])

# =========================
# Build transfer learning model
# =========================
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

base_model.trainable = False  # freeze backbone first

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)

# EfficientNet preprocessing
x = tf.keras.applications.efficientnet.preprocess_input(x)

x = base_model(x, training=False) #提取视觉特征（边缘、纹理、形状）
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# =========================
# Stage 1: Train classifier head
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n===== Stage 1: Train classification head =====")
history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS
)

# =========================
# Stage 2: Fine-tune upper layers
# =========================
base_model.trainable = True

# Freeze lower layers, fine-tune top layers only
fine_tune_at = 200  # you can adjust this
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n===== Stage 2: Fine-tuning =====")
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history_initial.epoch[-1] + 1
)

# =========================
# Final evaluation
# =========================
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# =========================
# Save model
# =========================
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/checkpoints", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# =========================
# Merge history for plotting
# =========================
acc = history_initial.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history_initial.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history_initial.history["loss"] + history_fine.history["loss"]
val_loss = history_initial.history["val_loss"] + history_fine.history["val_loss"]

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, acc, label="train_accuracy")
plt.plot(epochs_range, val_acc, label="val_accuracy")
plt.axvline(x=INITIAL_EPOCHS, color="gray", linestyle="--", label="start fine-tuning")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EfficientNet Transfer Learning Accuracy")
plt.legend()
plt.savefig("outputs/figures/transfer_accuracy.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, loss, label="train_loss")
plt.plot(epochs_range, val_loss, label="val_loss")
plt.axvline(x=INITIAL_EPOCHS, color="gray", linestyle="--", label="start fine-tuning")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNet Transfer Learning Loss")
plt.legend()
plt.savefig("outputs/figures/transfer_loss.png")
plt.close()

print("Training finished.")