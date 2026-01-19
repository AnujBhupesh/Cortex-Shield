"""
HTTP middleware for request logging and correlation.

Captures:
- request_id
- client_id
- latency_ms
- status_code
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response

from app.config import settings


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    """
    FastAPI middleware capturing request metadata for observability.
    """
    start = time.perf_counter()

    request_id = request.headers.get(settings.request_id_header) or str(uuid.uuid4())
    client_id = request.headers.get(settings.client_id_header) or "anonymous"

    request.state.request_id = request_id
    request.state.client_id = client_id

    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log = {
            "event": "request",
            "request_id": request_id,
            "client_id": client_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "latency_ms": latency_ms,
        }
        print(log, flush=True)
