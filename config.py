"""
Configuration layer for the Secure AI Inference Gateway.

All secrets are loaded from environment variables to support Twelve-Factor App
deployments and minimize secret sprawl in source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """
    Application settings loaded from environment variables.
    """

    service_name: str = os.getenv("SERVICE_NAME", "secure-ai-inference-gateway")
    environment: str = os.getenv("ENVIRONMENT", "production")

    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Upstream OpenAI-compatible provider configuration
    upstream_base_url: str = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com")
    upstream_api_key: str = os.getenv("UPSTREAM_API_KEY", "")
    upstream_timeout_seconds: float = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "30"))
    upstream_model_default: str = os.getenv("UPSTREAM_MODEL_DEFAULT", "gpt-4o-mini")

    # Redis configuration for rate limiting + health checks
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # Rate limiting
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM", "100"))

    # Security controls
    enable_presidio: bool = os.getenv("ENABLE_PRESIDIO", "true").lower() == "true"
    block_on_prompt_injection: bool = (
        os.getenv("BLOCK_ON_PROMPT_INJECTION", "true").lower() == "true"
    )

    # Logging / tracing
    request_id_header: str = os.getenv("REQUEST_ID_HEADER", "X-Request-Id")
    client_id_header: str = os.getenv("CLIENT_ID_HEADER", "X-Client-Id")


settings = Settings()
