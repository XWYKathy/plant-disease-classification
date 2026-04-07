from datetime import datetime

from pydantic import BaseModel

from config import CLASS_NAMES


class PredictionHistoryItem(BaseModel):
    """One row returned by GET /predictions/history."""
    record_id: int
    original_filename: str
    predicted_class: str
    confidence: float
    uploaded_at: datetime
    user_feedback_correct: bool | None
    user_feedback_label: str | None

    model_config = {"from_attributes": True}   # allow construction from ORM objects 是因为这里是从数据库拿数据


class FeedbackRequest(BaseModel):
    is_correct: bool
    # Required only when is_correct=False; ignored (but accepted) when is_correct=True
    correct_label: str | None = None

    #如果预测是错的 AND 提供了 correct_label → 必须是合法类别
    def model_post_init(self, __context) -> None:  # type: ignore[override]
        if not self.is_correct and self.correct_label is not None:
            if self.correct_label not in CLASS_NAMES:
                raise ValueError(
                    f"'{self.correct_label}' is not a valid class. "
                    f"Choose from: {CLASS_NAMES}"
                )


class FeedbackResponse(BaseModel):
    record_id: int
    user_feedback_correct: bool
    user_feedback_label: str | None
    message: str
