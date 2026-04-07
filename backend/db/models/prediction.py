from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base

if TYPE_CHECKING:
    from db.models.user import User


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(primary_key=True)

    # ── Ownership ─────────────────────────────────────────────────────────────
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    # ── Upload info ───────────────────────────────────────────────────────────
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # Path to saved image, relative to the project root (uploads/{user_id}/{uuid}_{filename})
    image_path: Mapped[str] = mapped_column(String(512), nullable=False)

    # ── Inference result ──────────────────────────────────────────────────────
    predicted_class: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # ── Timestamps ────────────────────────────────────────────────────────────
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ── User feedback (filled in later via PATCH /predictions/{id}/feedback) ──
    user_feedback_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    user_feedback_label: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # ── Relationship ──────────────────────────────────────────────────────────
    user: Mapped["User"] = relationship(back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"<PredictionRecord id={self.id} user_id={self.user_id} "
            f"class={self.predicted_class!r} confidence={self.confidence:.3f}>"
        )
