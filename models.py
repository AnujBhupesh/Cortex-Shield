"""
Pydantic v2 models for OpenAI-compatible /v1/chat/completions.

These models use strict validation to reduce malformed payload risks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """
    Represents a single chat message in OpenAI-compatible format.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    role: Literal["system", "user", "assistant", "tool"] = Field(...)
    content: Union[str, List[Dict[str, Any]]] = Field(..., min_length=1)


class ResponseFormat(BaseModel):
    """
    Optional response_format field used in some OpenAI-compatible APIs.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["text", "json_object"]


class ChatCompletionsRequest(BaseModel):
    """
    OpenAI-compatible chat completions request schema (subset + common fields).
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    model: str = Field(..., min_length=1, max_length=200)
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=200)

    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)

    stream: Optional[bool] = Field(default=False)
    n: Optional[int] = Field(default=1, ge=1, le=10)

    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)

    response_format: Optional[ResponseFormat] = None

    user: Optional[str] = Field(default=None, max_length=200)


class ErrorResponse(BaseModel):
    """
    OpenAI-style error payload.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    error: Dict[str, Any]

