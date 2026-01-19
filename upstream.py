"""
Upstream provider integration (OpenAI-compatible).

Uses httpx.AsyncClient and implements retry logic on transient failures.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from app.config import settings


_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _auth_headers() -> Dict[str, str]:
    """
    Construct upstream auth headers from environment variables.
    """
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    return headers


async def post_chat_completions(
    client: httpx.AsyncClient,
    payload: Dict[str, Any],
    request_id: str,
    client_id: str,
    timeout_seconds: Optional[float] = None,
) -> httpx.Response:
    """
    POST /v1/chat/completions to upstream provider.

    Includes:
    - Async request
    - Simple exponential backoff retries on transient errors
    - Correlation headers for debugging in upstream logs (if supported)
    """
    url = f"{settings.upstream_base_url.rstrip('/')}/v1/chat/completions"
    headers = _auth_headers()
    headers["X-Request-Id"] = request_id
    headers["X-Client-Id"] = client_id

    timeout = timeout_seconds or settings.upstream_timeout_seconds

    # Retry policy: quick, bounded, production-safe defaults
    max_attempts = 3
    backoff = 0.4

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code in _TRANSIENT_STATUS_CODES and attempt < max_attempts:
                await asyncio.sleep(backoff * attempt)
                continue
            return resp
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                await asyncio.sleep(backoff * attempt)
                continue
            raise

    # Should never happen, but keeps type checkers happy
    if last_exc:
        raise last_exc
    raise RuntimeError("Upstream request failed unexpectedly.")
