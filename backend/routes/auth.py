from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from db.session import get_db
from schemas.auth import LoginRequest, LoginResponse
from services.auth_service import authenticate_user

router = APIRouter()


@router.post("/login", response_model=LoginResponse, summary="User login") #FastAPI 会：自动校验返回格式 自动生成 Swagger 文档 自动过滤字段
def login(body: LoginRequest, db: Session = Depends(get_db)): #FastAPI 自动解析请求体
    """
    Authenticate a user and return a bearer token.

    Send the token in subsequent requests:
        Authorization: Bearer <token>
    """
    token = authenticate_user(db, body.username, body.password)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    return LoginResponse(
        success=True,
        message="Login successful.",
        token=token,
        username=body.username,
    )
