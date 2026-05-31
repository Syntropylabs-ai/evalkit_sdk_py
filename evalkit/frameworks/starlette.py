from __future__ import annotations

# Starlette middleware is the same ASGI pattern as FastAPI.
# Re-export EvalKitMiddleware so users can import from either path.
from .fastapi import EvalKitMiddleware

__all__ = ["EvalKitMiddleware"]
