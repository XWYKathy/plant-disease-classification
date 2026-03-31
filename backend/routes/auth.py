from fastapi import APIRouter

from schemas.auth import LoginRequest, LoginResponse
from services.auth_service import authenticate

router = APIRouter()


@router.post("/login", response_model=LoginResponse, summary="User login") #FastAPI 会：自动校验返回格式 自动生成 Swagger 文档 自动过滤字段
def login(body: LoginRequest): #FastAPI 自动解析请求体
    """
    Authenticate a user and return a bearer token.

    The token should be sent in the Authorization header for protected routes:
        Authorization: Bearer <token>
    """
    token = authenticate(body.username, body.password)
    if token is None:
        return LoginResponse(success=False, message="Invalid username or password.")
    return LoginResponse(
        success=True,
        message="Login successful.",
        token=token,
        username=body.username,
    )
