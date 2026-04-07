"""
Image I/O helpers: loading, validation, and base64 encoding.
All functions are stateless and have no framework dependencies.
"""
import base64
import io

import numpy as np
from PIL import Image

# ── Allowed upload types ──────────────────────────────────────────────────────
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/jpg"}


def validate_image_content_type(content_type: str) -> None:
    """Raise ValueError if the MIME type is not an accepted image format."""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Unsupported file type '{content_type}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
        )


def read_image_from_bytes(file_bytes: bytes, target_size: tuple[int, int]) -> np.ndarray:
    """
    Decode raw bytes → PIL Image → resized RGB numpy array.

    Returns:
        np.ndarray of shape (H, W, 3), dtype float32, values in [0, 255].
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((target_size[1], target_size[0]))   # PIL uses (width, height)
    return np.array(img, dtype=np.float32)


def ndarray_to_base64_png(image_array: np.ndarray) -> str:
    """
    Convert a (H, W, 3) or (H, W) uint8/float32 array to a base64-encoded PNG string.

    Frontend usage:
        <img src="data:image/png;base64,{returned_string}" />
    """
    arr = np.clip(image_array, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        # Grayscale heatmap → RGB for consistency
        arr = np.stack([arr] * 3, axis=-1)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
