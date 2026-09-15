"""Opt-in adapter for a single fenced JSON response in opt-in model runs.

The formal E1 wire contract remains a raw JSON object.  This module exists
only for an explicitly labelled replay path where a provider wrapped the
object in exactly one `````json```` fence.  The untouched provider text must
still be archived and counted as a raw-contract violation.
"""

from __future__ import annotations

import re
from typing import Any

from scripts.raw_response_contract import classify_raw_response, normalize_lf


ADAPTER_NAME = "single_fenced_json"
ADAPTER_VERSION = "1"

_FENCED_JSON_RE = re.compile(
    r"\A[ \t]*```json[ \t]*\n(?P<body>[\s\S]*?)\n```[ \t]*\Z"
)


def _decode_text(raw_output: object) -> tuple[str | None, dict[str, Any]]:
    if isinstance(raw_output, bytes):
        try:
            text = raw_output.decode("utf-8")
        except UnicodeDecodeError as exc:
            return None, {
                "class": "INVALID_UTF8",
                "reason": f"UnicodeDecodeError: {exc}",
            }
    elif isinstance(raw_output, str):
        text = raw_output
    else:
        return None, {
            "class": "NON_STRING_RESPONSE",
            "reason": "response is not text or UTF-8 bytes",
        }
    return normalize_lf(text), {}


def adapt_single_fenced_json(raw_output: object) -> dict[str, Any]:
    """Return a receipt for the narrow, deterministic fenced-JSON adapter.

    ``accepted`` is true only when the complete response consists of one
    `````json```` block whose body is a syntactically valid JSON object.  No
    prose, multiple fences, trailing data, or JSON repair is accepted.
    """

    raw_contract = classify_raw_response(raw_output)
    text, decode_info = _decode_text(raw_output)
    receipt: dict[str, Any] = {
        "adapter_name": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "applied": False,
        "accepted": False,
        "raw_contract": raw_contract,
    }
    if text is None:
        receipt["reason"] = decode_info.get("reason", "NON_STRING_RESPONSE")
        receipt["normalized_contract"] = decode_info
        return receipt

    stripped = text.strip()
    if not stripped:
        receipt["reason"] = "EMPTY_RESPONSE"
        receipt["normalized_contract"] = classify_raw_response("")
        return receipt

    # A provider that already obeys the raw JSON-object contract is scoreable
    # without using this adapter.  Preserve that distinction for mixed-model
    # retry batches.
    if raw_contract.get("strict_contract_valid") is True:
        receipt.update(
            {
                "reason": "RAW_JSON_OBJECT_ALREADY_VALID",
                "normalized_contract": raw_contract,
                "normalized_text": stripped,
            }
        )
        return receipt

    # A body containing another fence is a nested/multiple-fence response,
    # not the single provider wrapper this adapter is allowed to remove.
    if stripped.count("```") != 2:
        receipt["reason"] = "EXPECTED_EXACTLY_ONE_FENCE_PAIR"
        receipt["normalized_contract"] = None
        return receipt

    match = _FENCED_JSON_RE.fullmatch(stripped)
    if match is None:
        receipt["reason"] = "FENCE_MUST_BE_THE_ONLY_OUTER_CONTENT"
        receipt["normalized_contract"] = None
        return receipt

    normalized_text = match.group("body").strip()
    normalized_contract = classify_raw_response(normalized_text)
    receipt["normalized_contract"] = normalized_contract
    if normalized_contract.get("strict_contract_valid") is not True:
        receipt["reason"] = (
            "NORMALIZED_CONTRACT_INVALID:"
            f"{normalized_contract.get('class', 'UNKNOWN')}"
        )
        return receipt

    receipt.update(
        {
            "applied": True,
            "accepted": True,
            "reason": "SINGLE_JSON_FENCE_UNWRAPPED",
            "normalized_text": normalized_text,
        }
    )
    return receipt


def classify_fenced(raw_output: object, classify_json_object) -> dict[str, Any]:
    """Classify a provider response for the opt-in replay boundary.

    The callback is the unmodified detector classifier.  The returned
    classification remains a normal JSON-object classification after the
    adapter, while the receipt records that the raw stream was wrapped.
    """

    receipt = adapt_single_fenced_json(raw_output)
    if not receipt.get("accepted"):
        return classify_json_object(raw_output)
    classification = dict(classify_json_object(receipt["normalized_text"]))
    classification["fenced_json_adapter"] = {
        "adapter_name": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "raw_contract_class": receipt["raw_contract"].get("class"),
        "normalized_contract_class": receipt["normalized_contract"].get("class"),
    }
    return classification
