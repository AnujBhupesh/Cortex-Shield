"""
Observability helpers:
- Token estimation (tiktoken)
- Console logging for billing simulation
- Health checks for upstream + Redis
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx
import redis

from app.config import settings


@dataclass(frozen=True)
class HealthStatus:
    """
    Health status response structure.
    """

    ok: bool
    upstream_ok: bool
    redis_ok: bool
    details: Dict[str, Any]


def estimate_tokens(model: str, text: str) -> int:
    """
    Estimate token count using tiktoken. Falls back to cl100k_base if model is unknown.
    """
    import tiktoken  # type: ignore

    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def log_billing_simulation(
    *,
    request_id: str,
    client_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: Optional[int] = None,
) -> None:
    """
    Log token usage to stdout for billing simulation.
    """
    payload = {
        "event": "billing_simulation",
        "request_id": request_id,
        "client_id": client_id,
        "model": model,
        "prompt_tokens_estimated": prompt_tokens,
        "completion_tokens_estimated": completion_tokens,
        "timestamp_unix": int(time.time()),
    }
    print(payload, flush=True)


async def check_upstream(client: httpx.AsyncClient) -> Dict[str, Any]:
    """
    Verify upstream is reachable by calling an OpenAI-compatible endpoint.
    """
    url = f"{settings.upstream_base_url.rstrip('/')}/v1/models"
    headers = {}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"

    try:
        resp = await client.get(url, headers=headers, timeout=settings.upstream_timeout_seconds)
        return {"reachable": resp.status_code < 500, "status_code": resp.status_code}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def check_redis() -> Dict[str, Any]:
    """
    Verify Redis connectivity.
    """
    try:
        r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
        pong = r.ping()
        return {"reachable": bool(pong)}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


async def build_health_status(http_client: httpx.AsyncClient) -> HealthStatus:
    """
    Build a composite health status including upstream provider + Redis.
    """
    upstream = await check_upstream(http_client)
    redis_status = check_redis()

    upstream_ok = bool(upstream.get("reachable"))
    redis_ok = bool(redis_status.get("reachable"))

    return HealthStatus(
        ok=upstream_ok and redis_ok,
        upstream_ok=upstream_ok,
        redis_ok=redis_ok,
        details={"upstream": upstream, "redis": redis_status},
    )

