from fastapi import APIRouter
from typing import Dict

router = APIRouter(prefix="", tags=["health"])


@router.get('/')
async def read_root() -> Dict[str, str]:
    """루트 엔드포인트"""
    return {"message": "Welcome to the Business Analysis API!"}
