from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from db.models.user import User
from db.session import get_db
from schemas.history import FeedbackRequest, FeedbackResponse
from services.prediction_service import get_prediction_by_id, update_feedback
from utils.dependencies import get_current_user

router = APIRouter()


@router.patch(
    "/predictions/{record_id}/feedback",
    response_model=FeedbackResponse,
    summary="Submit or update feedback for a prediction",
)
def submit_feedback(
    record_id: int,
    body: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Mark whether the model's prediction was correct.
    If incorrect, optionally supply the correct label.

    Feedback can be updated by calling this endpoint again — later
    submissions overwrite earlier ones.  Only the record's owner can
    submit feedback (ownership is enforced server-side).
    """
    record = get_prediction_by_id(db, record_id=record_id, user_id=current_user.id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction record {record_id} not found.",
        )

    # When marking correct, discard any provided label to keep data clean
    correct_label = None if body.is_correct else body.correct_label

    record = update_feedback(db, record, is_correct=body.is_correct, correct_label=correct_label)

    return FeedbackResponse(
        record_id=record.id,
        user_feedback_correct=record.user_feedback_correct,
        user_feedback_label=record.user_feedback_label,
        message="Feedback recorded.",
    )
