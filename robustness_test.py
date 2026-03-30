import os
import tensorflow as tf

# =========================
# Config
# =========================
MODEL_PATH = "outputs/checkpoints/efficientnet_transfer.keras"
TEST_DIR = "data/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# =========================
# Load model
# =========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print(f"Loaded model from: {MODEL_PATH}")

# =========================
# Load clean test dataset
# =========================
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = test_ds.class_names
print("Classes:", class_names)

test_ds = test_ds.cache().prefetch(AUTOTUNE)

# =========================
# Perturbation functions
# Inputs from image_dataset_from_directory are float32 tensors
# in the range [0, 255].
# =========================
def apply_blur(image):
    """Approximate Gaussian blur with average pooling."""
    image = tf.expand_dims(image, axis=0)  # (1, H, W, C)
    image = tf.nn.avg_pool2d(
        image,
        ksize=5,
        strides=1,
        padding="SAME"
    )
    return tf.squeeze(image, axis=0)

def apply_brightness(image):
    """Random brightness change."""
    image = tf.image.random_brightness(image, max_delta=40.0)
    return tf.clip_by_value(image, 0.0, 255.0)

def apply_rotation(image):
    """Rotate by 0/90/180/270 degrees."""
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    return tf.image.rot90(image, k=k)

def apply_crop(image):
    """Random crop then resize back to original size."""
    cropped = tf.image.random_crop(image, size=[180, 180, 3])
    resized = tf.image.resize(cropped, IMG_SIZE)
    return tf.clip_by_value(resized, 0.0, 255.0)

def apply_noise(image):
    """Add Gaussian noise."""
    noise = tf.random.normal(
        shape=tf.shape(image),
        mean=0.0,
        stddev=15.0,
        dtype=tf.float32
    )
    noisy = image + noise
    return tf.clip_by_value(noisy, 0.0, 255.0)

# Optional approximation of background clutter:
# This adds a random colored rectangle patch to simulate clutter/occlusion.
def apply_background_clutter(image):
    """Add a random rectangular patch to simulate background clutter."""
    h = IMG_SIZE[0]
    w = IMG_SIZE[1]

    patch_h = tf.random.uniform([], minval=40, maxval=90, dtype=tf.int32)
    patch_w = tf.random.uniform([], minval=40, maxval=90, dtype=tf.int32)
    top = tf.random.uniform([], minval=0, maxval=h - patch_h + 1, dtype=tf.int32)
    left = tf.random.uniform([], minval=0, maxval=w - patch_w + 1, dtype=tf.int32)

    color = tf.random.uniform([1, 1, 3], minval=0.0, maxval=255.0, dtype=tf.float32)
    patch = tf.ones([patch_h, patch_w, 3], dtype=tf.float32) * color

    # Create mask
    mask = tf.pad(
        tf.ones([patch_h, patch_w, 3], dtype=tf.float32),
        paddings=[
            [top, h - top - patch_h],
            [left, w - left - patch_w],
            [0, 0]
        ]
    )

    patch_full = tf.pad(
        patch,
        paddings=[
            [top, h - top - patch_h],
            [left, w - left - patch_w],
            [0, 0]
        ]
    )

    image = image * (1.0 - mask) + patch_full * mask
    return tf.clip_by_value(image, 0.0, 255.0)

# =========================
# Helper to build perturbed datasets
# =========================
def make_perturbed_dataset(dataset, perturb_fn):
    return dataset.map(
        lambda x, y: (tf.map_fn(perturb_fn, x), y),
        num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

# =========================
# Create perturbed test sets
# =========================
blur_test_ds = make_perturbed_dataset(test_ds, apply_blur)
brightness_test_ds = make_perturbed_dataset(test_ds, apply_brightness)
rotation_test_ds = make_perturbed_dataset(test_ds, apply_rotation)
crop_test_ds = make_perturbed_dataset(test_ds, apply_crop)
noise_test_ds = make_perturbed_dataset(test_ds, apply_noise)
clutter_test_ds = make_perturbed_dataset(test_ds, apply_background_clutter)

# =========================
# Evaluation helper
# =========================
def evaluate_and_print(name, dataset):
    loss, acc = model.evaluate(dataset, verbose=1)
    print(f"{name:<20} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
    return loss, acc

# =========================
# Run evaluations
# =========================
print("\n===== Robustness Evaluation =====")
results = {}

results["clean"] = evaluate_and_print("Clean Test", test_ds)
results["blur"] = evaluate_and_print("Blur Test", blur_test_ds)
results["brightness"] = evaluate_and_print("Brightness Test", brightness_test_ds)
results["rotation"] = evaluate_and_print("Rotation Test", rotation_test_ds)
results["crop"] = evaluate_and_print("Crop Test", crop_test_ds)
results["noise"] = evaluate_and_print("Noise Test", noise_test_ds)
results["clutter"] = evaluate_and_print("Clutter Test", clutter_test_ds)

print("\n===== Summary =====")
for name, (loss, acc) in results.items():
    print(f"{name:<12} -> loss={loss:.4f}, accuracy={acc:.4f}")