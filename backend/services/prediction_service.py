"""
PredictionRecord CRUD.

Image files are saved to  uploads/{user_id}/{uuid}_{original_filename}
under the project root (BASE_DIR).  The DB stores the path relative to
BASE_DIR so the app stays portable; callers construct the full path with
config.BASE_DIR / record.image_path when they need to read the file.
"""
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from config import BASE_DIR
from db.models.prediction import PredictionRecord

UPLOADS_ROOT = Path(BASE_DIR) / "uploads"


def save_image_file(user_id: int, filename: str, file_bytes: bytes) -> str:
    """
    Write the raw image bytes to disk and return the path relative to BASE_DIR.

    Layout: uploads/{user_id}/{uuid}_{original_filename}
    """
    user_dir = UPLOADS_ROOT / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{Path(filename).name}"
    dest = user_dir / safe_name
    dest.write_bytes(file_bytes)

    # Store relative to BASE_DIR so the path is environment-independent
    return str(dest.relative_to(BASE_DIR))


def create_prediction_record(
    db: Session,
    *,
    user_id: int,
    original_filename: str,
    image_path: str,
    predicted_class: str,
    confidence: float,
) -> PredictionRecord:
    record = PredictionRecord(
        user_id=user_id,
        original_filename=original_filename,
        image_path=image_path,
        predicted_class=predicted_class,
        confidence=confidence,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_user_predictions(db: Session, user_id: int) -> list[PredictionRecord]:
    """Return all records for a user, newest first."""
    return list(
        db.execute(
            select(PredictionRecord)
            .where(PredictionRecord.user_id == user_id)
            .order_by(PredictionRecord.uploaded_at.desc())
        ).scalars()
    )


def get_prediction_by_id(
    db: Session, record_id: int, user_id: int
) -> PredictionRecord | None:
    """Fetch a single record, enforcing ownership so users can't access each other's data."""
    return db.execute(
        select(PredictionRecord).where(
            PredictionRecord.id == record_id,
            PredictionRecord.user_id == user_id,
        )
    ).scalar_one_or_none() #只能取1或0条，否则报错（因为record_id是主键）


def update_feedback(
    db: Session,
    record: PredictionRecord,
    is_correct: bool,
    correct_label: str | None,
) -> PredictionRecord:
    record.user_feedback_correct = is_correct
    record.user_feedback_label = correct_label
    db.commit()
    db.refresh(record)
    return record
