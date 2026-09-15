"""Frozen response-format contracts for E2-B model calls."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any


DEFAULT_RESPONSE_FORMAT_MODE = "json_object"
STRICT_RESPONSE_FORMAT_MODE = "json_schema_strict"
FOUR_FIELD_STRICT_RESPONSE_FORMAT_MODE = "json_schema_strict_four_field"
HARDENED_FOUR_FIELD_STRICT_RESPONSE_FORMAT_MODE = "json_schema_strict_four_field_v2"
COMPACT_RESPONSE_FORMAT_MODE = "json_schema_compact_findings"
CANDIDATE_ID_RESPONSE_FORMAT_MODE = "json_schema_selected_candidate_ids"
FORMAL_E2_LABELS = (
    "access_control",
    "arithmetic",
    "front_running",
    "reentrancy",
    "time_manipulation",
    "unchecked_low_level_calls",
)
RESPONSE_FORMAT_MODES = frozenset((
    DEFAULT_RESPONSE_FORMAT_MODE,
    STRICT_RESPONSE_FORMAT_MODE,
    FOUR_FIELD_STRICT_RESPONSE_FORMAT_MODE,
    HARDENED_FOUR_FIELD_STRICT_RESPONSE_FORMAT_MODE,
    COMPACT_RESPONSE_FORMAT_MODE,
    CANDIDATE_ID_RESPONSE_FORMAT_MODE,
))


def _closed_object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [{"type": "null"}, schema]}


CONFIDENCE = {"type": "number"}

STATE_VARIABLE = _closed_object({"name": {"type": "string"}, "type": {"type": "string"}})
PRE_CONDITION = _closed_object({
    "variable": {"type": "string"},
    "operator": {"type": "string"},
    "value": {"type": "string"},
    "must_be": {"type": "string"},
})
STATE_TRANSITION = _closed_object({
    "variable": {"type": "string"},
    "operator": {"type": "string"},
    "value": {"type": "string"},
})
Z3_SCHEMA = _closed_object({
    "state_variables": {"type": "array", "items": STATE_VARIABLE},
    "pre_conditions": {"type": "array", "items": PRE_CONDITION},
    "state_transitions": {"type": "array", "items": STATE_TRANSITION},
})
CONSTRAINT = _closed_object({
    "id": {"type": "string"},
    "description": {"type": "string"},
    "expression": {"type": "string"},
    "must_be": {"type": "string"},
    "related_line": {"type": "string"},
    "z3_schema": Z3_SCHEMA,
    "llm_reasoning": {"type": "string"},
})
IDENTIFIED_RISK = _closed_object({
    "risk_type": {"type": "string"},
    "function_name": {"type": "string"},
    "triggering_data_flow": {"type": "string"},
    "terminal_action_check": {"type": "string"},
})
FINDING = _closed_object({
    "vulnerability_type": {"type": "string"},
    "function_name": {"type": "string"},
    "attack_path": {"type": "string"},
    "temporal_pattern": {"type": "string"},
    "ast_match_confidence": CONFIDENCE,
    "constraint_violation_confidence": CONFIDENCE,
    "constraints": {"type": "array", "items": CONSTRAINT},
})

LOW_LEVEL_CALL = _closed_object({
    "location": {"type": "string"},
    "return_captured": {"type": "boolean"},
    "return_checked": {"type": "boolean"},
    "verdict": {"type": "string"},
})
LOW_LEVEL_CALL_CHECKLIST = _closed_object({
    "calls": {"type": "array", "items": LOW_LEVEL_CALL},
    "verdict": {"type": "string"},
})
ARITHMETIC_GUARDRAIL = _closed_object({
    "solidity_version": {"type": "string"},
    "safemath_used": {"type": "boolean"},
    "explicit_overflow_guards": {"type": "boolean"},
    "unchecked_blocks": {"type": "boolean"},
    "verdict": {"type": "string"},
})
LIFECYCLE_GUARDRAIL = _closed_object({
    "is_initialization_function": {"type": "boolean"},
    "sets_critical_state": {"type": "boolean"},
    "has_reinit_guard": {"type": "boolean"},
    "has_access_check": {"type": "boolean"},
    "verdict": {"type": "string"},
})
TIME_MANIPULATION_GUARDRAIL = _closed_object({
    "timestamp_usage": {"type": "string"},
    "temporal_subtype": {"type": "string"},
    "producer": {"type": "string"},
    "consumer": {"type": "string"},
    "invariant_violated": {"type": "string"},
    "financial_impact": {"type": "string"},
    "verdict": {"type": "string"},
})
DOS_PREFLIGHT = _closed_object({
    "is_gas_exhaustion_possible": {"type": "boolean"},
    "can_attacker_infinitely_inflate_loop": {"type": "boolean"},
    "is_ether_permanently_trapped_without_profit": {"type": "boolean"},
    "loop_variable": {"type": "string"},
    "is_attacker_controlled": {"type": "boolean"},
    "veto_dos": {"type": "boolean"},
})
DOS_INFLATION_PROOF = _closed_object({
    "loop_location": {"type": "string"},
    "loop_variable": {"type": "string"},
    "is_attacker_controlled": {"type": "boolean"},
    "termination_condition": {"type": "string"},
    "attacker_can_inflate": {"type": "boolean"},
    "inflation_mechanism": {"type": "string"},
    "blocks_all_users": {"type": "boolean"},
    "verdict": {"type": "string"},
})
FRONT_RUNNING_GUARDRAIL = _closed_object({
    "critical_state_read": {"type": "boolean"},
    "attacker_can_front_run": {"type": "boolean"},
    "has_slippage_check": {"type": "boolean"},
    "has_commit_reveal": {"type": "boolean"},
    "financial_impact": {"type": "string"},
    "verdict": {"type": "string"},
})

STRICT_OUTPUT_SCHEMA = _closed_object({
    "identified_risks": {"type": "array", "items": IDENTIFIED_RISK},
    "root_cause_analysis": {"type": "string"},
    "primary_vulnerabilities": {"type": "array", "items": {"type": "string"}},
    "findings": {"type": "array", "items": FINDING},
    "dos_pre_flight_check": DOS_PREFLIGHT,
    "lifecycle_guardrail": LIFECYCLE_GUARDRAIL,
})

STRICT_JSON_SCHEMA = {
    "name": "fusedaudit_e2b_output_v2",
    "strict": True,
    "schema": STRICT_OUTPUT_SCHEMA,
}

FOUR_FIELD_STRICT_OUTPUT_SCHEMA = _closed_object({
    "identified_risks": {"type": "array", "items": IDENTIFIED_RISK},
    "root_cause_analysis": {"type": "string"},
    "primary_vulnerabilities": {"type": "array", "items": {"type": "string"}},
    "findings": {"type": "array", "items": FINDING},
})

FOUR_FIELD_STRICT_JSON_SCHEMA = {
    "name": "dappscan_audit_output",
    "strict": True,
    "schema": FOUR_FIELD_STRICT_OUTPUT_SCHEMA,
}

HARDENED_IDENTIFIED_RISK = _closed_object({
    "risk_type": {"type": "string", "enum": list(FORMAL_E2_LABELS)},
    "function_name": {"type": "string"},
    "triggering_data_flow": {"type": "string"},
    "terminal_action_check": {"type": "string"},
})
HARDENED_FINDING = _closed_object({
    "vulnerability_type": {"type": "string", "enum": list(FORMAL_E2_LABELS)},
    "function_name": {"type": "string"},
    "attack_path": {"type": "string"},
    "temporal_pattern": {"type": "string"},
    "ast_match_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "constraint_violation_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "constraints": {"type": "array", "items": CONSTRAINT},
})
HARDENED_FOUR_FIELD_OUTPUT_SCHEMA = _closed_object({
    "identified_risks": {"type": "array", "items": HARDENED_IDENTIFIED_RISK},
    "root_cause_analysis": {"type": "string"},
    "primary_vulnerabilities": {
        "type": "array",
        "items": {"type": "string", "enum": list(FORMAL_E2_LABELS)},
    },
    "findings": {"type": "array", "items": HARDENED_FINDING},
})
HARDENED_FOUR_FIELD_STRICT_JSON_SCHEMA = {
    "name": "dappscan_audit_output_v2",
    "strict": True,
    "schema": HARDENED_FOUR_FIELD_OUTPUT_SCHEMA,
}

COMPACT_FINDING = _closed_object({
    "category": {
        "type": "string",
        "enum": [
            "access_control",
            "arithmetic",
            "front_running",
            "reentrancy",
            "time_manipulation",
            "unchecked_low_level_calls",
        ],
    },
    "function_name": {"type": "string"},
    "primary_line": {"type": "integer", "minimum": 1},
    "evidence_lines": {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
    },
    "reason": {"type": "string"},
})

COMPACT_OUTPUT_SCHEMA = _closed_object({
    "findings": {"type": "array", "items": COMPACT_FINDING},
})

COMPACT_JSON_SCHEMA = {
    "name": "dappscan_compact_findings_output",
    "strict": True,
    "schema": COMPACT_OUTPUT_SCHEMA,
}

CANDIDATE_ID_JSON_SCHEMA = {
    "name": "dappscan_selected_candidate_ids",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_candidate_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            }
        },
        "required": ["selected_candidate_ids"],
    },
}


def response_format_for_mode(mode: str) -> dict[str, Any]:
    """Return the exact provider payload for a frozen response-format mode."""
    if mode == DEFAULT_RESPONSE_FORMAT_MODE:
        return {"type": "json_object"}
    if mode == STRICT_RESPONSE_FORMAT_MODE:
        return {"type": "json_schema", "json_schema": deepcopy(STRICT_JSON_SCHEMA)}
    if mode == FOUR_FIELD_STRICT_RESPONSE_FORMAT_MODE:
        return {"type": "json_schema", "json_schema": deepcopy(FOUR_FIELD_STRICT_JSON_SCHEMA)}
    if mode == HARDENED_FOUR_FIELD_STRICT_RESPONSE_FORMAT_MODE:
        return {"type": "json_schema", "json_schema": deepcopy(HARDENED_FOUR_FIELD_STRICT_JSON_SCHEMA)}
    if mode == COMPACT_RESPONSE_FORMAT_MODE:
        return {"type": "json_schema", "json_schema": deepcopy(COMPACT_JSON_SCHEMA)}
    if mode == CANDIDATE_ID_RESPONSE_FORMAT_MODE:
        return {"type": "json_schema", "json_schema": deepcopy(CANDIDATE_ID_JSON_SCHEMA)}
    raise ValueError(f"unsupported FUSEDAUDIT_RESPONSE_FORMAT_MODE: {mode!r}")


def strict_response_schema_sha256() -> str:
    payload = json.dumps(STRICT_JSON_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
