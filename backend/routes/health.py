from fastapi import APIRouter

import services.model_service as model_service

router = APIRouter()


@router.get("/health", summary="Health check")
def health_check():
    """Returns API status and whether the model has been loaded successfully."""
    return {
        "status": "ok",
        "model_loaded": model_service._model is not None,
    }
