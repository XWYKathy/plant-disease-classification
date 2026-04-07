from pydantic import BaseModel


class PredictionItem(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    record_id: int                  # DB primary key — use for feedback/history lookups
    predicted_class: str
    confidence: float
    top_k: list[PredictionItem]
    # Both images are base64-encoded PNGs.
    # Frontend usage:  <img src="data:image/png;base64,{gradcam_heatmap}" />
    gradcam_heatmap: str
    overlay_image: str
