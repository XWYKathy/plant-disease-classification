"""
One-off script to create a test user for login testing.

Usage (from the backend/ directory, with .env loaded):
    python create_test_user.py
    python create_test_user.py --username alice --password secret123
"""
import argparse
import os
import sys
from pathlib import Path

# Load .env if present
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(Path(__file__).parent))

from db.session import SessionLocal
from services.user_service import create_user, get_user_by_username
from utils.security import hash_password

DEFAULT_USERNAME = "testuser"
DEFAULT_PASSWORD = "testpass123"


def main():
    parser = argparse.ArgumentParser(description="Create a test user")
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        existing = get_user_by_username(db, args.username)
        if existing:
            print(f"User '{args.username}' already exists (id={existing.id}).")
            return

        user = create_user(db, args.username, hash_password(args.password))
        print(f"Created user '{user.username}' (id={user.id}).")
        print(f"  username : {args.username}")
        print(f"  password : {args.password}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
