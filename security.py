"""
Security & hardening controls:
- PII Redaction (Presidio preferred, regex fallback)
- Prompt Injection Shield (signature-based)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from app.config import settings

# Regex patterns (fast path). These are intentionally pragmatic rather than perfect.
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
)
# Basic credit card detection: 13-19 digits with optional separators; Luhn validated later.
_CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


@dataclass(frozen=True)
class GuardrailResult:
    """
    Outcome of running security guardrails.
    """

    redacted_text: str
    was_redacted: bool
    injection_detected: bool
    injection_signatures: List[str]


def _luhn_check(number: str) -> bool:
    """
    Luhn checksum validation for credit card number candidates.
    """
    digits = [int(ch) for ch in number if ch.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False

    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _regex_redact(text: str) -> Tuple[str, bool]:
    """
    Redact emails, IPv4 addresses, and credit card numbers using regex + Luhn validation.
    """
    original = text

    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _IPV4_RE.sub("[REDACTED_IP]", text)

    def cc_replacer(match: re.Match) -> str:
        candidate = match.group(0)
        compact = "".join(ch for ch in candidate if ch.isdigit())
        if _luhn_check(compact):
            return "[REDACTED_CREDIT_CARD]"
        return candidate

    text = _CC_RE.sub(cc_replacer, text)

    return text, text != original


def _presidio_redact(text: str) -> Tuple[str, bool]:
    """
    Redact PII using Presidio Analyzer + Anonymizer.

    This is an actual integration. If Presidio is enabled but fails due to missing
    NLP model dependencies, we fall back to regex redaction.
    """
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        from presidio_anonymizer import AnonymizerEngine  # type: ignore
        from presidio_anonymizer.entities import OperatorConfig  # type: ignore
    except Exception:
        return _regex_redact(text)

    try:
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        # Analyze for: EMAIL_ADDRESS, IP_ADDRESS, and CREDIT_CARD
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=["EMAIL_ADDRESS", "IP_ADDRESS", "CREDIT_CARD"],
        )

        if not results:
            return text, False

        operators = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED_EMAIL]"}),
            "IP_ADDRESS": OperatorConfig("replace", {"new_value": "[REDACTED_IP]"}),
            "CREDIT_CARD": OperatorConfig(
                "replace", {"new_value": "[REDACTED_CREDIT_CARD]"}
            ),
        }

        redacted = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
        return redacted.text, redacted.text != text
    except Exception:
        # Safety: never fail open due to a redaction runtime issue.
        return _regex_redact(text)


_PROMPT_INJECTION_SIGNATURES: List[Tuple[str, re.Pattern]] = [
    ("ignore_previous_instructions", re.compile(r"\bignore\b.*\bprevious\b.*\binstructions\b", re.IGNORECASE)),
    ("system_override", re.compile(r"\bsystem\b.*\boverride\b", re.IGNORECASE)),
    ("dan", re.compile(r"\bDAN\b", re.IGNORECASE)),
    ("jailbreak", re.compile(r"\bjailbreak\b|\bdo anything now\b", re.IGNORECASE)),
]


def detect_prompt_injection(text: str) -> List[str]:
    """
    Signature-based prompt injection detection.

    Returns a list of signature IDs that were detected in the text.
    """
    hits: List[str] = []
    for sig_id, pattern in _PROMPT_INJECTION_SIGNATURES:
        if pattern.search(text):
            hits.append(sig_id)
    return hits


def run_guardrails_on_text(text: str) -> GuardrailResult:
    """
    Apply redaction + injection checks to a text blob.
    """
    injection_hits = detect_prompt_injection(text)
    injection_detected = len(injection_hits) > 0

    if settings.enable_presidio:
        redacted, was_redacted = _presidio_redact(text)
    else:
        redacted, was_redacted = _regex_redact(text)

    return GuardrailResult(
        redacted_text=redacted,
        was_redacted=was_redacted,
        injection_detected=injection_detected,
        injection_signatures=injection_hits,
    )


def normalize_messages_to_text(message_contents: Iterable[str]) -> str:
    """
    Normalize message contents into a single text blob for guardrail scanning.
    """
    return "\n".join([c for c in message_contents if c is not None and str(c).strip()])
