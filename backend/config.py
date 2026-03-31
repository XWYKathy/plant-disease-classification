"""
Central configuration for the Plant Disease Classification API.
Edit the values here to adapt the backend to your environment.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# BASE_DIR points to the project root (one level above backend/)
# 获取项目根目录（backend/ 的父目录）
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = str(BASE_DIR / "outputs" / "checkpoints" / "efficientnet_transfer.keras")

# ── Model ────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
TOP_K = 3   # how many top predictions to return

# Class names must match the order used during training.
# tf.keras.utils.image_dataset_from_directory sorts them with Python's sorted(),
# which is case-sensitive (uppercase before lowercase).
# Verify this list against: print(train_ds.class_names) from your training script.
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy",
]

# ── CORS ─────────────────────────────────────────────────────────────────────
# Replace "*" with your frontend origin (e.g. "http://localhost:3000") in production
ALLOWED_ORIGINS = ["*"]

# ── Auth (mock) ───────────────────────────────────────────────────────────────
# Replace with a real database / identity provider before going to production.
# 模拟用户，用于开发环境
MOCK_USERS: dict[str, str] = {
    "demo": "password123",
    "admin": "admin123",
}
