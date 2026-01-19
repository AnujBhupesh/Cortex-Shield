"""
Rate limiting with slowapi + Redis storage.

Clients are identified by X-Client-Id; defaults to "anonymous".
"""

from __future__ import annotations

from slowapi import Limiter  # type: ignore
from slowapi.errors import RateLimitExceeded  # type: ignore
from slowapi.util import get_remote_address  # type: ignore
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import settings


def _key_func(request: Request) -> str:
    """
    Rate limit key function. Prefer enterprise client identifier; fallback to IP.
    """
    client_id = request.headers.get(settings.client_id_header)
    if client_id and client_id.strip():
        return f"client:{client_id.strip()}"
    return f"ip:{get_remote_address(request)}"


# Redis-backed storage for distributed rate limiting
limiter = Limiter(
    key_func=_key_func,
    storage_uri=settings.redis_url,
    default_limits=[f"{settings.rate_limit_rpm}/minute"],
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    OpenAI-style-ish error response on rate limit exceed.
    """
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": "Rate limit exceeded. Please retry later.",
                "type": "rate_limit_error",
                "param": None,
                "code": "rate_limit_exceeded",
            }
        },
        headers={"Retry-After": str(getattr(exc, "retry_after", 60))},
    )

