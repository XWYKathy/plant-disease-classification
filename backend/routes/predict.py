import tensorflow as tf
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from config import CLASS_NAMES, IMG_SIZE
from db.models.user import User
from db.session import get_db
from schemas.predict import PredictResponse, PredictionItem
from services.gradcam_service import compute_gradcam, render_gradcam_overlay
from services.model_service import run_inference
from services.prediction_service import create_prediction_record, save_image_file
from utils.dependencies import get_current_user
from utils.image_utils import (
    ndarray_to_base64_png,
    read_image_from_bytes,
    validate_image_content_type,
)

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, summary="Classify plant leaf image")
async def predict(
    file: UploadFile = File(..., description="Plant leaf image (JPEG / PNG)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Authenticated endpoint.  Uploads a plant leaf image, runs EfficientNet
    inference, generates a Grad-CAM heatmap, persists a PredictionRecord,
    and returns the full result including base64-encoded heatmap images.

    The returned `record_id` can be used later to:
      - Submit feedback   → PATCH /predictions/{record_id}/feedback
      - View in history   → GET  /predictions/history
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

    # ── 5. Persist to DB ─────────────────────────────────────────────────────
    # Save the original file bytes to disk; store relative path in DB
    original_filename = file.filename or "upload"
    image_path = save_image_file(current_user.id, original_filename, file_bytes)

    record = create_prediction_record(
        db,
        user_id=current_user.id,
        original_filename=original_filename,
        image_path=image_path,
        predicted_class=predicted_class,
        confidence=round(confidence, 4),
    )

    # ── 6. Return response ───────────────────────────────────────────────────
    return PredictResponse(
        record_id=record.id,
        predicted_class=predicted_class,        #预测类别
        confidence=round(confidence, 4),        #置信度
        top_k=[PredictionItem(**item) for item in top_k],  #top_k结果
        gradcam_heatmap=ndarray_to_base64_png(heatmap_rgb), #热力图
        overlay_image=ndarray_to_base64_png(overlay),       #叠加后的图像
    )
