"""
Central configuration for the Plant Disease Classification API.
Edit the values here to adapt the backend to your environment.
"""
import os
from dotenv import load_dotenv
load_dotenv()
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

# ── Database ──────────────────────────────────────────────────────────────────
# Read from environment variable DATABASE_URL.
# Example: postgresql://postgres:password@localhost:5432/plant_disease
DATABASE_URL: str = os.environ["DATABASE_URL"]

# ── JWT ───────────────────────────────────────────────────────────────────────
# JWT_SECRET_KEY must be a long random string in production.
# Generate one with:  python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY: str = os.environ["JWT_SECRET_KEY"]
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
