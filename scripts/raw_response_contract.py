"""Fail-closed raw-response archival contract for the formal E1 runner.

The provider text is evidence.  It is never repaired, truncated, or replaced
with a parsed/normalized envelope before it is archived.  E1 accepts the
legacy ``json_object`` wire format, so the contract checks only that the raw
payload is a single complete JSON object with no non-whitespace suffix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "E1-RAW-RESPONSE-CONTRACT-2"
CONTRACT_NAME = "json_object_legacy_compat_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_lf(value: str) -> str:
    """Return deterministic UTF-8 text with LF line endings."""

    return value.replace("\r\n", "\n").replace("\r", "\n")


def canonical_json_sha256(payload: object) -> str:
    """Hash the parsed JSON object without changing the archived raw bytes."""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def classify_raw_response(raw_output: object) -> dict[str, Any]:
    """Classify untouched provider text; never apply compatibility repair."""

    if isinstance(raw_output, bytes):
        try:
            text = raw_output.decode("utf-8")
        except UnicodeDecodeError as exc:
            return {
                "class": "INVALID_UTF8",
                "present": bool(raw_output),
                "utf8_valid": False,
                "json_syntax_valid": False,
                "strict_contract_valid": False,
                "reason": f"UnicodeDecodeError: {exc}",
            }
    elif isinstance(raw_output, str):
        text = raw_output
    else:
        text = ""

    normalized = normalize_lf(text)
    if not normalized.strip():
        return {
            "class": "EMPTY_RESPONSE",
            "present": False,
            "utf8_valid": True,
            "json_syntax_valid": False,
            "strict_contract_valid": False,
            "reason": "response is empty",
        }

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    decoder = json.JSONDecoder(parse_constant=reject_nonstandard_constant)
    try:
        payload, decoded_end = decoder.raw_decode(normalized.lstrip())
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "class": "MALFORMED_JSON",
            "present": True,
            "utf8_valid": True,
            "json_syntax_valid": False,
            "strict_contract_valid": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    stripped = normalized.lstrip()
    trailing = stripped[decoded_end:].strip()
    if trailing:
        return {
            "class": "TRAILING_DATA",
            "present": True,
            "utf8_valid": True,
            "json_syntax_valid": False,
            "strict_contract_valid": False,
            "reason": "TRAILING_NON_WHITESPACE",
        }
    if not isinstance(payload, dict):
        return {
            "class": "NON_OBJECT_JSON",
            "present": True,
            "utf8_valid": True,
            "json_syntax_valid": True,
            "strict_contract_valid": False,
            "reason": "ROOT_NOT_OBJECT",
            "canonical_json_sha256": canonical_json_sha256(payload),
        }
    return {
        "class": "JSON_OBJECT",
        "present": True,
        "utf8_valid": True,
        "json_syntax_valid": True,
        "strict_contract_valid": True,
        "reason": "VALID_JSON_OBJECT",
        "canonical_json_sha256": canonical_json_sha256(payload),
    }


def archive_raw_response(path: Path, raw_output: object) -> dict[str, Any]:
    """Write the raw response with fixed UTF-8/LF bytes and return its receipt."""

    if isinstance(raw_output, bytes):
        try:
            text = raw_output.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_output.decode("utf-8", errors="surrogateescape")
    else:
        text = str(raw_output or "")
    normalized = normalize_lf(text)
    raw_bytes = normalized.encode("utf-8", errors="surrogateescape")
    contract = classify_raw_response(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)
    return {
        "contract_version": CONTRACT_VERSION,
        "contract": CONTRACT_NAME,
        **contract,
        "encoding": "utf-8",
        "newline": "lf",
        "byte_length": len(raw_bytes),
        "sha256": _sha256_bytes(raw_bytes),
    }
