"""
Model loading and inference service.

The model is loaded once at application startup and reused for every request.
This avoids the high cost of loading a TensorFlow model on each call.
"""
import numpy as np
import tensorflow as tf

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_PATH, CLASS_NAMES, TOP_K, IMG_SIZE

# Module-level singleton — populated by load_model() at startup
_model: tf.keras.Model | None = None


# ── Lifecycle ─────────────────────────────────────────────────────────────────

def load_model() -> None:
    """Load the Keras model into memory.  Called once via FastAPI lifespan."""
    global _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run src/train_transfer.py first to generate the model."
        )
    print(f"[model_service] Loading model from {MODEL_PATH} ...")
    _model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[model_service] Model loaded.  Input shape: {_model.input_shape}")


def get_model() -> tf.keras.Model:
    """Return the loaded model, or raise if load_model() was not called."""
    if _model is None:
        raise RuntimeError("Model is not loaded.  Ensure load_model() ran at startup.")
    return _model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(image_array: np.ndarray) -> tf.Tensor:
    """
    Add a batch dimension to a (H, W, 3) float32 array.

    The model graph already includes:
      data_augmentation → efficientnet.preprocess_input → EfficientNetB0
    so we only need to batch the raw [0, 255] image here.
    """
    return tf.expand_dims(tf.cast(image_array, tf.float32), axis=0)  # (1, H, W, 3)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    image_array: np.ndarray,
) -> tuple[str, float, list[dict]]:
    """
    Run a forward pass and return the top prediction + top-k results.

    Args:
        image_array: (H, W, 3) float32, values in [0, 255], already resized to IMG_SIZE.

    Returns:
        predicted_class: display name of the top class
        confidence:      softmax probability of the top class (0–1)
        top_k:           list of {"label": str, "confidence": float}, length = TOP_K
    """
    model = get_model()
    x = preprocess(image_array)

    # training=False keeps BatchNorm / Dropout in inference mode
    probs: np.ndarray = model(x, training=False).numpy()[0]   # (num_classes,)

    top_k_indices = np.argsort(probs)[::-1][:TOP_K]
    top_k = [
        {"label": CLASS_NAMES[i], "confidence": float(probs[i])}
        for i in top_k_indices
    ]

    best_idx = int(np.argmax(probs))
    return CLASS_NAMES[best_idx], float(probs[best_idx]), top_k
