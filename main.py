"""
Secure AI Inference Gateway - FastAPI entry point.

Implements:
- /v1/chat/completions (OpenAI-compatible schema)
- /health (upstream + Redis)
- Middleware logging (request_id, client_id, latency, status)
- Rate limiting (Redis-backed)
- PII redaction + prompt injection shielding before upstream call
- Token estimation (tiktoken) for billing simulation logs
"""

from __future__ import annotations

from typing import Any, Dict, List, Union, cast

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from slowapi.middleware import SlowAPIMiddleware  # type: ignore

from app.config import settings
from app.middleware import request_logging_middleware
from app.models import ChatCompletionsRequest, ErrorResponse
from app.monitoring import build_health_status, estimate_tokens, log_billing_simulation
from app.rate_limit import limiter, rate_limit_exceeded_handler
from app.security import normalize_messages_to_text, run_guardrails_on_text
from app.upstream import post_chat_completions


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    app = FastAPI(title=settings.service_name)

    # Middleware: request logging + slowapi
    app.middleware("http")(request_logging_middleware)
    app.state.limiter = limiter
    app.add_exception_handler(429, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Reusable async HTTP client
    app.state.http_client = httpx.AsyncClient(http2=True)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.http_client.aclose()

    @app.get("/health")
    async def health() -> JSONResponse:
        """
        Health endpoint checks upstream provider and Redis connectivity.
        """
        status = await build_health_status(app.state.http_client)
        return JSONResponse(
            status_code=200 if status.ok else 503,
            content={
                "ok": status.ok,
                "upstream_ok": status.upstream_ok,
                "redis_ok": status.redis_ok,
                "details": status.details,
            },
        )

    def _extract_message_texts(req: ChatCompletionsRequest) -> List[str]:
        """
        Extract string contents from messages, ignoring non-string content blocks.
        """
        texts: List[str] = []
        for m in req.messages:
            if isinstance(m.content, str):
                texts.append(m.content)
            else:
                # If multimodal blocks exist, include any {"type":"text","text": "..."} content
                for block in cast(List[Dict[str, Any]], m.content):
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text")
                        if isinstance(t, str):
                            texts.append(t)
        return texts

    def _apply_redaction_to_request(req: ChatCompletionsRequest) -> ChatCompletionsRequest:
        """
        Redact PII from message contents (string fields and text blocks).
        """
        # Create a deep-ish transformed copy (Pydantic model is immutable-ish by design patterns)
        data = req.model_dump()

        for msg in data.get("messages", []):
            content = msg.get("content")
            if isinstance(content, str):
                result = run_guardrails_on_text(content)
                msg["content"] = result.redacted_text
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block.get("text")
                        if isinstance(t, str):
                            result = run_guardrails_on_text(t)
                            block["text"] = result.redacted_text

        return ChatCompletionsRequest.model_validate(data)

    async def _validated_body(request: Request) -> ChatCompletionsRequest:
        """
        Strictly validate incoming JSON against Pydantic v2 models.
        """
        try:
            raw = await request.json()
            return ChatCompletionsRequest.model_validate(raw)
        except ValidationError as exc:
            err = ErrorResponse(
                error={
                    "message": "Invalid request payload.",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "validation_error",
                    "details": exc.errors(),
                }
            )
            raise ValueError(err.model_dump())

    @app.post("/v1/chat/completions")
    @limiter.limit(f"{settings.rate_limit_rpm}/minute")
    async def chat_completions(request: Request, body: ChatCompletionsRequest = Depends(_validated_body)) -> JSONResponse:
        """
        OpenAI-compatible Chat Completions gateway endpoint.
        Applies:
        - prompt injection scanning (block if configured)
        - PII redaction
        - token estimation logging for cost tracking
        - async upstream call
        """
        request_id = getattr(request.state, "request_id", "unknown")
        client_id = getattr(request.state, "client_id", "anonymous")

        # Prompt injection scanning on normalized text
        message_texts = _extract_message_texts(body)
        joined = normalize_messages_to_text(message_texts)
        inj_check = run_guardrails_on_text(joined)

        if inj_check.injection_detected and settings.block_on_prompt_injection:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": "Prompt injection detected. Request blocked by security policy.",
                        "type": "invalid_request_error",
                        "param": "messages",
                        "code": "prompt_injection_detected",
                        "signatures": inj_check.injection_signatures,
                        "request_id": request_id,
                    }
                },
            )

        # Redact PII before forwarding upstream
        redacted_req = _apply_redaction_to_request(body)

        # Token estimation on redacted prompt (billing simulation)
        prompt_text = normalize_messages_to_text(_extract_message_texts(redacted_req))
        prompt_tokens = estimate_tokens(redacted_req.model, prompt_text)
        log_billing_simulation(
            request_id=request_id,
            client_id=client_id,
            model=redacted_req.model,
            prompt_tokens=prompt_tokens,
        )

        # Forward to upstream (OpenAI-compatible)
        upstream_payload: Dict[str, Any] = redacted_req.model_dump()

        # If caller didn't specify model (shouldn't happen due to validation),
        # ensure a safe default.
        if not upstream_payload.get("model"):
            upstream_payload["model"] = settings.upstream_model_default

        http_client: httpx.AsyncClient = app.state.http_client
        try:
            upstream_resp = await post_chat_completions(
                client=http_client,
                payload=upstream_payload,
                request_id=request_id,
                client_id=client_id,
            )
        except ValueError as exc:
            # Raised by our validation wrapper with serialized OpenAI-style error
            return JSONResponse(status_code=400, content=cast(dict, exc.args[0]))
        except Exception as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "Upstream provider request failed.",
                        "type": "upstream_error",
                        "param": None,
                        "code": "upstream_unreachable",
                        "details": str(exc),
                        "request_id": request_id,
                    }
                },
            )

        # Pass-through response body with upstream status code
        try:
            data = upstream_resp.json()
        except Exception:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "Upstream returned non-JSON response.",
                        "type": "upstream_error",
                        "param": None,
                        "code": "invalid_upstream_response",
                        "status_code": upstream_resp.status_code,
                        "request_id": request_id,
                    }
                },
            )

        # Optional: estimate completion tokens if upstream includes content
        # (We do not “mock” token usage; we only estimate if text is present.)
        completion_text = ""
        try:
            choices = data.get("choices", [])
            if choices and isinstance(choices, list):
                msg = choices[0].get("message", {})
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str):
                        completion_text = c
        except Exception:
            completion_text = ""

        if completion_text:
            completion_tokens = estimate_tokens(redacted_req.model, completion_text)
            log_billing_simulation(
                request_id=request_id,
                client_id=client_id,
                model=redacted_req.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        return JSONResponse(status_code=upstream_resp.status_code, content=data)

    return app


app = create_app()
