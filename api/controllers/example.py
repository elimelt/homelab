from fastapi import APIRouter
from typing import Dict

router = APIRouter()


@router.get("/example")
async def example() -> Dict[str, str]:
    return {
        "message": "Hello from DevStack API!",
        "timestamp": "2025-11-26T00:00:00Z",
        "status": "success",
    }


