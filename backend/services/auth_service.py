"""
Authentication service — DB-backed.

Public interface:
    authenticate_user(db, username, password) -> str | None  (returns JWT on success)

Users are created directly in the database (e.g. via psql or a seed script).
To upgrade to OAuth2 / SSO later: replace only this file.
"""
from sqlalchemy.orm import Session

from services.user_service import get_user_by_username
from utils.security import create_access_token, hash_password, verify_password


def authenticate_user(db: Session, username: str, password: str) -> str | None:
    """
    Validate credentials and return a signed JWT on success, None on failure.
    Timing-safe: verify_password always runs even if the user doesn't exist,
    preventing user enumeration via response time.
    """
    user = get_user_by_username(db, username)
    # Use a dummy hash so verify_password always runs (constant-time guard)
    candidate_hash = user.password_hash if user else hash_password("__dummy__")
    if not verify_password(password, candidate_hash) or user is None:
        return None
    return create_access_token(username=user.username)
