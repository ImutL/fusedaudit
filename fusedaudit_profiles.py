"""Runtime profiles used to isolate legacy E1 from E2 development behavior.

The evaluator historically received E2 controls as independent environment
variables.  This module adds one explicit boundary without changing the
values of any existing controls: E1 is the default profile, and execution
entry points must select ``e2_dappscan_vnext`` explicitly.  A deprecated
unprofiled read is retained for direct legacy unit helpers; production runners
call :func:`assert_execution_profile` and therefore cannot use that escape.
"""

from __future__ import annotations

import os
from typing import Mapping


LEGACY_E1_PROFILE = "legacy_e1"
E1_OPTIMIZED_V1_PROFILE = "e1_optimized_v1"
E2_DAPPSCAN_VNEXT_PROFILE = "e2_dappscan_vnext"
PROFILE_ENV = "FUSEDAUDIT_PROFILE"
E1_ABLATION_ENV = "FUSEDAUDIT_E1_ABLATION_ARM"
E1_ABLATION_ARMS = frozenset(
    {
        "full",
        "without_slicing",
        "retrieval_off",
        "without_z3",
    }
)

E1_ABLATION_SEMANTICS_VERSION = "E1-ABLATION-SEMANTICS-2"

# Keep the historical arm names stable for existing checkpoints, while making
# the disabled component explicit in receipts and runtime gating.
E1_ABLATION_COMPONENTS = {
    "full": {
        "ast_slicing": True,
        "retrieval": True,
        "z3_solver": True,
    },
    "without_slicing": {
        "ast_slicing": False,
        "retrieval": True,
        "z3_solver": True,
    },
    "retrieval_off": {
        "ast_slicing": True,
        "retrieval": False,
        "z3_solver": True,
    },
    "without_z3": {
        "ast_slicing": True,
        "retrieval": True,
        "z3_solver": False,
    },
}


def e1_ablation_components(arm: str) -> dict[str, bool]:
    """Return the explicit component switches for one E1 ablation arm."""

    try:
        return dict(E1_ABLATION_COMPONENTS[arm])
    except KeyError as exc:
        raise ValueError(
            f"unsupported E1 ablation arm {arm!r}; expected one of "
            f"{sorted(E1_ABLATION_ARMS)}"
        ) from exc

# E1 keeps the frozen legacy JSON-object wire contract. The provider may use
# one of the historical compatible envelopes inside that object; the raw
# response itself must still be a complete JSON object before local parsing.
E1_RAW_RESPONSE_CONTRACT = "json_object_legacy_compat_v1"

PROFILE_OVERLAYS = {
    LEGACY_E1_PROFILE: {
        "prompt": "legacy_default",
        "schema": "legacy_default",
        "ast_admission": "legacy_default",
        "threshold": "legacy_default",
        "retrieval": "legacy_default",
    },
    E1_OPTIMIZED_V1_PROFILE: {
        "prompt": "legacy_default",
        "schema": "legacy_default",
        "ast_admission": "e1_source_grounded_v1",
        "threshold": "legacy_default",
        "retrieval": "e1_sanitized_v1",
    },
    E2_DAPPSCAN_VNEXT_PROFILE: {
        "prompt": "e2_dappscan_vnext",
        "schema": "e2_dappscan_vnext",
        "ast_admission": "e2_dappscan_vnext",
        "threshold": "e2_dappscan_vnext_env_only",
        "retrieval": "e2_dappscan_vnext_isolated_readonly",
    },
}

E2_DEVELOPMENT_FLAGS = frozenset(
    {
        "FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE",
        "FUSEDAUDIT_E2_IDENTIFIED_RISK_BRIDGE",
        "FUSEDAUDIT_E2_FULL_NAMESPACE_RETRIEVAL",
        "FUSEDAUDIT_E2_STRICT_OUTPUT_COVERAGE",
        "FUSEDAUDIT_E2_STRICT_PROVIDER_CONTRACT",
        "FUSEDAUDIT_E2_PROVIDER_CONTRACT_HARDENING",
        "FUSEDAUDIT_E2_PROVIDER_TRANSPORT_HARDENING",
        "FUSEDAUDIT_E2_PROVIDER_COMPATIBILITY_REPAIR",
        "FUSEDAUDIT_E2_COMPACT_PROMPT",
        "FUSEDAUDIT_E2_TRAILING_CLOSER_REPAIR",
        "FUSEDAUDIT_E2_RETRIEVAL_ABLATION_ARM",
        "FUSEDAUDIT_E2_CONTEXT_RETRIEVAL",
        # Arithmetic proof admission is intentionally opt-in.  The default
        # E2 replay path keeps this disabled and publishes locator-only
        # behavior until a paired replay proves zero regression.
        "FUSEDAUDIT_E2_ARITHMETIC_PROOF_GATE",
        "FUSEDAUDIT_E2_PRECISION_GATES",
        "FUSEDAUDIT_E2_SOURCE_CANDIDATES",
        "FUSEDAUDIT_E2_ARITHMETIC_LOCATOR_RELEASE",
        # Replay-only continuation for sealed historical Dev responses.  The
        # replay entry point is the only current caller that enables it.
        "FUSEDAUDIT_E2_HISTORICAL_SOURCE_CONTINUATION",
        "FUSEDAUDIT_E2_CONTEXT_VARIANT",
        "FUSEDAUDIT_E2_CACHE_ROOT",
        "FUSEDAUDIT_E2_CHECKPOINT_ROOT",
        "FUSEDAUDIT_RETRIEVAL_READ_ONLY",
    }
)


def resolve_profile(environment: Mapping[str, str] | None = None) -> str:
    """Return the selected profile, defaulting to the immutable E1 path."""

    env = os.environ if environment is None else environment
    value = str(env.get(PROFILE_ENV, LEGACY_E1_PROFILE) or LEGACY_E1_PROFILE).strip()
    if value not in {
        LEGACY_E1_PROFILE,
        E1_OPTIMIZED_V1_PROFILE,
        E2_DAPPSCAN_VNEXT_PROFILE,
    }:
        raise ValueError(
            f"unsupported {PROFILE_ENV}={value!r}; expected "
            f"{LEGACY_E1_PROFILE!r}, {E1_OPTIMIZED_V1_PROFILE!r}, or "
            f"{E2_DAPPSCAN_VNEXT_PROFILE!r}"
        )
    return value


def is_e2_profile(environment: Mapping[str, str] | None = None) -> bool:
    return resolve_profile(environment) == E2_DAPPSCAN_VNEXT_PROFILE


def is_e1_optimized_profile(environment: Mapping[str, str] | None = None) -> bool:
    return resolve_profile(environment) == E1_OPTIMIZED_V1_PROFILE


def profile_overlay(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    return dict(PROFILE_OVERLAYS[resolve_profile(environment)])


def e2_flag_enabled(name: str, environment: Mapping[str, str] | None = None) -> bool:
    """Read an E2 flag through the explicit E2 profile boundary."""

    if name not in E2_DEVELOPMENT_FLAGS:
        raise ValueError(f"unknown E2 development flag: {name}")
    env = os.environ if environment is None else environment
    # Existing unit helpers historically set the flag directly.  Keep those
    # calls deterministic while making all execution entry points explicit.
    if PROFILE_ENV not in env:
        return str(env.get(name, "")) == "1"
    return is_e2_profile(env) and str(env.get(name, "")) == "1"


def assert_e1_isolated(environment: Mapping[str, str] | None = None) -> None:
    """Fail closed when E1 is asked to run with an E2 development flag."""

    env = os.environ if environment is None else environment
    profile = resolve_profile(env)
    if profile == E2_DAPPSCAN_VNEXT_PROFILE:
        return
    enabled = sorted(name for name in E2_DEVELOPMENT_FLAGS if str(env.get(name, "")) == "1")
    if enabled:
        raise RuntimeError(
            "E1 isolation violation: E2 development flags are enabled under "
            f"{LEGACY_E1_PROFILE}: {', '.join(enabled)}"
        )


def assert_execution_profile(expected: str, environment: Mapping[str, str] | None = None) -> None:
    """Require an execution entry point to declare its profile explicitly."""

    env = os.environ if environment is None else environment
    raw = env.get(PROFILE_ENV)
    if raw != expected:
        raise RuntimeError(
            f"execution requires explicit {PROFILE_ENV}={expected!r}; got {raw!r}"
        )
