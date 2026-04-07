# Import models here so Alembic's env.py picks them up via Base.metadata
from db.models.user import User           # noqa: F401
from db.models.prediction import PredictionRecord  # noqa: F401
