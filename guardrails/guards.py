"""
Guardrails & PII protection layer.
- Detects and anonymises PII (names, phone numbers, emails, SSNs, credit cards)
  using Microsoft Presidio.
- Validates input length and content safety.
- Returns flags so the pipeline can decide whether to block or continue.
"""

import re
from typing import Optional

# Presidio imports — graceful fallback if not installed
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    _presidio_available = True
except ImportError:
    _presidio_available = False

# Regex-based fallback patterns for common PII
_EMAIL_RE    = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE    = re.compile(r"\b(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})\b")
_SSN_RE      = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_RE     = re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")

# Content safety — block prompt injection / jailbreak attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak|unrestricted)",
    r"act\s+as\s+(?:an?\s+)?(?:evil|malicious|unethical)",
    r"system\s*:\s*you\s+are",
    r"<\s*system\s*>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

MAX_QUERY_LENGTH = 2000

# Initialise Presidio engines once
_analyzer  = AnalyzerEngine()  if _presidio_available else None
_anonymizer = AnonymizerEngine() if _presidio_available else None


def _regex_redact(text: str) -> tuple[str, list[str]]:
    """Fallback regex-based PII redaction."""
    flags = []
    if _EMAIL_RE.search(text):
        text = _EMAIL_RE.sub("[EMAIL]", text)
        flags.append("PII_EMAIL")
    if _PHONE_RE.search(text):
        text = _PHONE_RE.sub("[PHONE]", text)
        flags.append("PII_PHONE")
    if _SSN_RE.search(text):
        text = _SSN_RE.sub("[SSN]", text)
        flags.append("PII_SSN")
    if _CARD_RE.search(text):
        text = _CARD_RE.sub("[CARD]", text)
        flags.append("PII_CARD")
    return text, flags


def _presidio_redact(text: str) -> tuple[str, list[str]]:
    """Presidio-based PII detection and anonymisation."""
    results = _analyzer.analyze(text=text, language="en")
    if not results:
        return text, []
    flags = list({r.entity_type for r in results})
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text, [f"PII_{f}" for f in flags]


def validate_and_sanitize(text: str) -> dict:
    """
    Main guard function called by the pipeline before any LLM processing.

    Returns:
        {
            "sanitized_text": str,
            "flags": list[str],   # e.g. ["PII_EMAIL", "INJECTION_DETECTED"]
            "blocked": bool,
        }
    """
    flags: list[str] = []

    # 1. Length check
    if len(text) > MAX_QUERY_LENGTH:
        text = text[:MAX_QUERY_LENGTH]
        flags.append("TRUNCATED")

    # 2. Prompt injection detection → block
    if _INJECTION_RE.search(text):
        flags.append("INJECTION_DETECTED")
        flags.append("BLOCKED")
        return {"sanitized_text": "[BLOCKED: unsafe content]", "flags": flags, "blocked": True}

    # 3. PII redaction
    if _presidio_available:
        text, pii_flags = _presidio_redact(text)
    else:
        text, pii_flags = _regex_redact(text)
    flags.extend(pii_flags)

    return {"sanitized_text": text, "flags": flags, "blocked": False}
