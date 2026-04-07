from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from db.models.user import User
from db.session import get_db
from schemas.history import PredictionHistoryItem
from services.prediction_service import get_user_predictions
from utils.dependencies import get_current_user

router = APIRouter()


@router.get(
    "/predictions/history",
    response_model=list[PredictionHistoryItem],
    summary="Get prediction history for the current user",
)
def get_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns all prediction records for the authenticated user, newest first.
    Each item includes the predicted class, confidence, timestamp,
    and any feedback the user has already submitted.
    """
    records = get_user_predictions(db, current_user.id)
    return [
        PredictionHistoryItem(
            record_id=r.id,
            original_filename=r.original_filename,
            predicted_class=r.predicted_class,
            confidence=r.confidence,
            uploaded_at=r.uploaded_at,
            user_feedback_correct=r.user_feedback_correct,
            user_feedback_label=r.user_feedback_label,
        )
        for r in records
    ]
