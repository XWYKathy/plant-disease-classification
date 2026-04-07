"""
Database engine and session factory.

get_db() is the FastAPI dependency injected into every route that needs DB access.
It yields a session and guarantees cleanup even on exceptions.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import DATABASE_URL

#数据库连接池管理器（connection pool manager）
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # reconnects silently after a dropped connection SQLAlchemy 会自动重连
    pool_size=5, #最多保持 5 个长期连接
    max_overflow=10, #最多额外创建 10 个连接
)

#SessionLocal = 创建“会话”的机器
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) #不自动提交、不自动 flush（把内存里的变更同步到 DB）、绑定数据库连接


def get_db():
    """FastAPI dependency — yields a DB session and closes it when the request ends."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
