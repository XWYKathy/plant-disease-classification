import tensorflow as tf
from fastapi import APIRouter, File, HTTPException, UploadFile

from config import CLASS_NAMES, IMG_SIZE
from schemas.predict import PredictResponse, PredictionItem
from services.gradcam_service import compute_gradcam, render_gradcam_overlay
from services.model_service import run_inference
from utils.image_utils import (
    ndarray_to_base64_png,
    read_image_from_bytes,
    validate_image_content_type,
)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, summary="Classify plant leaf image")
async def predict(file: UploadFile = File(..., description="Plant leaf image (JPEG / PNG)")):
    """
    Accept an uploaded image, run EfficientNet inference, generate a Grad-CAM
    heatmap, and return the prediction with both heatmap images as base64 PNGs.
    """
    # ── 1. Validate file type ────────────────────────────────────────────────
    try:
        validate_image_content_type(file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    file_bytes = await file.read()

    # ── 2. Decode and resize image ───────────────────────────────────────────
    try:
        image_array = read_image_from_bytes(file_bytes, IMG_SIZE)   # (H, W, 3) float32
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    # ── 3. Run inference ─────────────────────────────────────────────────────
    predicted_class, confidence, top_k = run_inference(image_array)

    # ── 4. Grad-CAM ──────────────────────────────────────────────────────────
    pred_index = CLASS_NAMES.index(predicted_class)
    image_batch = tf.expand_dims(image_array, axis=0)   # (1, H, W, 3)

    try:
        heatmap, _ = compute_gradcam(image_batch, pred_index=pred_index)
        heatmap_rgb, overlay = render_gradcam_overlay(image_array, heatmap)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {exc}")

    # ── 5. Return response ───────────────────────────────────────────────────
    return PredictResponse(
        predicted_class=predicted_class,
        confidence=round(confidence, 4),
        top_k=[PredictionItem(**item) for item in top_k],
        gradcam_heatmap=ndarray_to_base64_png(heatmap_rgb),
        overlay_image=ndarray_to_base64_png(overlay),
    )
