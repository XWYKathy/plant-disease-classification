from fastapi import APIRouter
from services.model_service import _model

router = APIRouter()


@router.get("/health", summary="Health check")
def health_check():
    """Returns API status and whether the model has been loaded successfully."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }
