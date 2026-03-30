"""
Authentication service.

Currently uses an in-memory mock store.  To add real auth:
  - Replace `MOCK_USERS` in config.py with a DB lookup.
  - Replace the opaque token with a signed JWT (e.g. python-jose).
  - Replace `_active_tokens` with Redis or a DB session table.

The public interface (authenticate / verify_token) stays the same.
"""
import secrets
import sys
import os

# Allow running from the backend/ directory or from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MOCK_USERS

# In-memory token store.  Not persistent across restarts.
_active_tokens: dict[str, str] = {}   # token → username


def authenticate(username: str, password: str) -> str | None:
    """
    Validate credentials.  Returns an opaque bearer token on success, None on failure.
    """
    expected_password = MOCK_USERS.get(username)
    if expected_password is None or expected_password != password:
        return None
    token = secrets.token_hex(32)
    _active_tokens[token] = username
    return token


def verify_token(token: str) -> str | None:
    """Return the username for a valid token, or None if the token is unknown."""
    return _active_tokens.get(token)
