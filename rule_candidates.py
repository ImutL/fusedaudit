"""Rule-based candidate additions from source structure.

The helpers in this module deliberately return *candidates*, not findings.
They are enabled only by an explicit opt-in environment flag and use source
structure plus the already parsed function inventory.  No dataset/source id
is consulted here.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


_READ_METHODS = {
    "allowance",
    "balanceof",
    "decimals",
    "factory",
    "getpair",
    "getreserves",
    "hasrole",
    "latestanswer",
    "owner",
    "quote",
    "supportsinterface",
    "totalsupply",
    "weth",
}
_CALL_METHODS = {
    "approve",
    "burn",
    "call",
    "delegatecall",
    "deposit",
    "execute",
    "flashloan",
    "liquidate",
    "mint",
    "repay",
    "send",
    "settle",
    "stake",
    "staticcall",
    "swap",
    "transfer",
    "transferfrom",
    "unstake",
    "withdraw",
}
_KNOWN_TYPED_TOKEN_METHODS = {"approve", "transfer", "transferfrom"}
_AUTH_WORDS = re.compile(
    r"\b(?:only|auth|admin|owner|guardian|governance|operator|manager|role|pauser|controller)",
    re.I,
)
_SENSITIVE_WORDS = re.compile(
    r"\b(?:admin|allowance|balance|collateral|controller|credit|debt|delegat|factory|fee|guardian|implementation|initialized?|limit|operator|owner|proxy|registry|reserve|reward|share|stake|supply|token|treasury|validator)\w*\b",
    re.I,
)
_STATE_ASSIGN = re.compile(
    r"(?:\b[A-Za-z_]\w*(?:\s*\[[^\n\]]+\])+(?:\.[A-Za-z_]\w+)?|\b[A-Za-z_]\w*)\s*(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)",
)
_MEMBER_ASSIGN = re.compile(
    r"\b(?P<base>[A-Za-z_]\w*)\s*(?:\[[^\n\]]+\]\s*)?\.\s*[A-Za-z_]\w+\s*(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)",
)
_LOW_LEVEL = re.compile(r"\.\s*(call|send|delegatecall|staticcall)\s*(?:\.|\{)?\s*\(", re.I)
_QUALIFIED_CALL = re.compile(
    r"\b[A-Za-z_]\w*\s*\.\s*(?P<method>[A-Za-z_]\w*)\s*(?:\{[^}\n]*\}\s*)?\(",
    re.I,
)
_BARE_CALL = re.compile(r"(?<![.\w])(?P<name>[A-Za-z_]\w*)\s*\(")
_SAFE_MINT_CALL = re.compile(
    r"(?<![A-Za-z0-9_.])_?safeMint\s*\(",
    re.I,
)
_ASSET_ACTION_CALL = re.compile(
    r"\.\s*(?:approve|burn|call|deposit|mint|send|safeTransfer|"
    r"safeTransferFrom|settle|transfer|transferFrom|withdraw)\s*"
    r"(?:\{[^}\n]*\}\s*)?\(",
    re.I,
)
_USER_CONTROLLED_RECEIVER = re.compile(
    r"\b(?:account|beneficiary|msg\s*\.\s*sender|owner|recipient|"
    r"sender|to|_msgSender)\b",
    re.I,
)


def _lines(source: str) -> list[str]:
    return (source or "").splitlines()


def _func_name(func: Any) -> str:
    return str(getattr(func, "name", "") or "").strip()


def _func_visibility(func: Any) -> str:
    return str(getattr(func, "visibility", "") or "").casefold()


def _func_span(func: Any, lines: list[str]) -> tuple[int, int]:
    start = max(int(getattr(func, "line_number", 0) or 0), 1)
    end = int(getattr(func, "end_line_number", 0) or 0)
    if end < start:
        end = min(len(lines), start + 80)
    return start, min(end, len(lines))


def _func_text(func: Any, lines: list[str]) -> tuple[int, int, str]:
    start, end = _func_span(func, lines)
    return start, end, "\n".join(lines[start - 1 : end])


def _modifiers(func: Any) -> list[str]:
    return [str(value).strip() for value in (getattr(func, "modifiers", []) or []) if str(value).strip()]


def _has_authorization_guard(func: Any, body: str) -> bool:
    """Return whether the entry point has an access-control guard.

    Authorization and reentrancy protection are intentionally separate.  An
    ``onlyOwner``/``onlyRole`` modifier does not establish a mutex and must
    not suppress an external-call-before-state-write candidate.
    """

    if any(_AUTH_WORDS.search(value) for value in _modifiers(func)):
        return True
    return bool(
        re.search(
            r"\b(?:require|assert|if)\s*\([^\n;{}]*(?:msg\s*\.\s*sender|hasrole|owner|admin|guardian|operator)[^\n;{}]*\)",
            body,
            re.I,
        )
    )


def _has_mutex_guard(func: Any, body: str) -> bool:
    """Return whether the function visibly uses a reentrancy mutex guard."""

    if any(
        re.search(r"\b(?:nonreentrant|reentrancyguard|mutex|locked)\b", value, re.I)
        for value in _modifiers(func)
    ):
        return True
    return bool(
        re.search(r"\b(?:nonreentrant|reentrancyguard|mutex)\b", body, re.I)
        or re.search(
            r"\b(?:require|assert|if)\s*\([^\n;{}]*(?:_?entered|_?locked|_?status)[^\n;{}]*\)",
            body,
            re.I,
        )
    )


def _has_guard(func: Any, body: str) -> bool:
    """Backward-compatible combined guard predicate for external callers."""

    return _has_authorization_guard(func, body) or _has_mutex_guard(func, body)


def _storage_aliases(func: Any, lines: list[str]) -> set[str]:
    """Return local names that alias storage-backed structs or mappings."""

    _, _, body = _func_text(func, lines)
    return {
        match.group("name")
        for match in re.finditer(
            r"\b[A-Za-z_]\w*(?:\s*<[^\n;{}>]+>)?\s+storage\s+(?P<name>[A-Za-z_]\w*)\b",
            body,
            re.I,
        )
    }


def _state_names(contracts: Iterable[Any]) -> set[str]:
    names: set[str] = set()
    for contract in contracts:
        for name in getattr(contract, "state_variables", []) or []:
            text = str(name).strip()
            if text:
                names.add(text)
    return names


def _line_has_state_write(
    line: str,
    state_names: set[str],
    storage_aliases: set[str] | None = None,
) -> bool:
    if re.search(r"\b(?:emit|return|require|assert)\b", line, re.I):
        return False
    assignment = _STATE_ASSIGN.search(line)
    member_assignment = _MEMBER_ASSIGN.search(line)
    if not assignment and not member_assignment:
        return False
    if state_names and any(
        re.search(rf"\b{re.escape(name)}\b", line) for name in state_names
    ):
        return True
    if member_assignment and member_assignment.group("base") in (storage_aliases or set()):
        return True
    return bool(
        re.search(r"\b(?:owner|admin|guardian|controller|factory|implementation|registry|treasury)\b", line, re.I)
        or re.search(r"\[[^\]]*\]\s*(?:=|\+=|-=)", line)
    )


def _critical_write_lines(func: Any, lines: list[str], state_names: set[str]) -> list[int]:
    start, end, _ = _func_text(func, lines)
    storage_aliases = _storage_aliases(func, lines)
    result: list[int] = []
    for number in range(start, end + 1):
        line = lines[number - 1]
        if _line_has_state_write(line, state_names, storage_aliases) and _SENSITIVE_WORDS.search(line):
            result.append(number)
    return result


def _asset_sink_lines(body_start: int, body_lines: list[str]) -> list[int]:
    result: list[int] = []
    for offset, line in enumerate(body_lines):
        if re.search(
            r"\.(?:approve|burn|call|delegatecall|deposit|execute|liquidate|mint|repay|send|settle|stake|swap|transfer|transferfrom|unstake|withdraw)\s*(?:\.|\{)?\s*\(",
            line,
            re.I,
        ):
            result.append(body_start + offset)
    return result


def _internal_edges(funcs: list[Any], lines: list[str]) -> dict[str, list[tuple[int, str]]]:
    names = {_func_name(func) for func in funcs if _func_name(func)}
    edges: dict[str, list[tuple[int, str]]] = {name: [] for name in names}
    for func in funcs:
        caller = _func_name(func)
        if not caller:
            continue
        start, end, _ = _func_text(func, lines)
        for number in range(start, end + 1):
            line = lines[number - 1]
            for match in _BARE_CALL.finditer(line):
                callee = match.group("name")
                if callee in names and callee != caller and not re.search(r"\b(?:function|modifier|if|for|while|require|assert|return)\s*$", line[: match.start()], re.I):
                    edges[caller].append((number, callee))
    return edges


def _reachable_functions(root: str, edges: dict[str, list[tuple[int, str]]]) -> set[str]:
    result: set[str] = set()
    stack = [root]
    while stack:
        name = stack.pop()
        if name in result:
            continue
        result.add(name)
        stack.extend(callee for _, callee in edges.get(name, []))
    return result


def _ordered_reachable_names(
    reachable: Iterable[str], funcs_by_name: dict[str, Any]
) -> list[str]:
    """Return reachable functions in stable source order."""

    return sorted(
        (name for name in reachable if name in funcs_by_name),
        key=lambda name: (
            int(getattr(funcs_by_name[name], "line_number", 0) or 0),
            int(getattr(funcs_by_name[name], "end_line_number", 0) or 0),
            name.casefold(),
            name,
        ),
    )


def _external_event_lines(func: Any, lines: list[str]) -> list[int]:
    start, end, _ = _func_text(func, lines)
    result: list[int] = []
    for number in range(start, end + 1):
        line = lines[number - 1]
        if _LOW_LEVEL.search(line):
            result.append(number)
            continue
        match = _QUALIFIED_CALL.search(line)
        if not match:
            continue
        method = match.group("method").casefold()
        if method in _READ_METHODS or method.startswith(("get", "is", "has", "quote", "calc", "current", "latest")):
            continue
        if method in _CALL_METHODS:
            result.append(number)
    return result


def _typed_return_methods(source: str) -> dict[str, dict[str, Any]]:
    """Return source-declared methods that expose a typed return value."""

    methods: dict[str, dict[str, Any]] = {}
    for match in re.finditer(
        r"\bfunction\s+(?P<name>[A-Za-z_]\w*)\s*\(",
        source or "",
        re.I,
    ):
        end_candidates = [
            value
            for value in (source.find("{", match.end()), source.find(";", match.end()))
            if value >= 0
        ]
        if not end_candidates:
            continue
        header = source[match.start() : min(end_candidates)]
        return_match = re.search(
            r"\breturns\s*\((?P<types>[^)]*)\)",
            header,
            re.I | re.S,
        )
        if return_match is None or not return_match.group("types").strip():
            continue
        methods[match.group("name").casefold()] = {
            "returns": return_match.group("types").strip(),
            "read_only": bool(re.search(r"\b(?:view|pure)\b", header, re.I)),
        }
    return methods


def _modifier_candidates(contracts: Iterable[Any], lines: list[str], funcs_by_name: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for contract in contracts:
        for modifier in getattr(contract, "modifiers", []) or []:
            body = str(getattr(modifier, "body_text", "") or "")
            if not re.search(r"\btx\s*\.\s*origin\b", body, re.I):
                continue
            if re.search(r"\bmsg\s*\.\s*sender\s*(?:==|!=)\s*tx\s*\.\s*origin\b|\btx\s*\.\s*origin\s*(?:==|!=)\s*msg\s*\.\s*sender\b", body, re.I):
                continue
            if not re.search(r"\b(?:require|assert|if)\s*\([^\n;{}]*tx\s*\.\s*origin", body, re.I):
                continue
            modifier_name = str(getattr(modifier, "name", "") or "")
            protected = [
                func for func in funcs_by_name.values()
                if modifier_name in _modifiers(func)
                and _func_visibility(func) in {"public", "external"}
                and str(getattr(func, "state_mutability", "") or "").casefold() not in {"view", "pure"}
            ]
            for func in protected:
                writes = _critical_write_lines(func, lines, _state_names(contracts))
                if not writes:
                    writes = _asset_sink_lines(_func_span(func, lines)[0], lines[_func_span(func, lines)[0] - 1 : _func_span(func, lines)[1]])
                if not writes:
                    continue
                modifier_line = max(int(getattr(modifier, "line_number", 0) or 0), 1)
                origin_line = modifier_line
                for offset, text in enumerate(body.splitlines()):
                    if re.search(r"\btx\s*\.\s*origin\b", text, re.I):
                        origin_line = modifier_line + offset
                        break
                results.append({
                    "risk_type": "access_control",
                    "submechanism": "tx_origin_modifier_authorization",
                    "source_evidence_kind": "tx_origin_modifier_authorization",
                    "source_grounded": True,
                    "confidence": 0.93,
                    "reason": f"modifier {modifier_name}() uses tx.origin authorization before { _func_name(func) }() critical state/asset operation",
                    "function_name": _func_name(func),
                    "entrypoint_function_name": _func_name(func),
                    "entrypoint_lines": [int(getattr(func, "line_number", 0) or 0)],
                    "line": origin_line,
                    "evidence_lines": sorted(set([origin_line, *writes[:4]])),
                    "state_write_lines": writes[:4],
                    "modifier_name": modifier_name,
                    "modifier_line": modifier_line,
                    "modifier_body": body,
                    "source_call_path": [modifier_name, _func_name(func)],
                    "sink_function_name": _func_name(func),
                })
    return results


def access_control_candidates(contracts: list[Any], source: str, existing: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    lines = _lines(source)
    if not lines:
        return []
    funcs = [func for contract in contracts for func in getattr(contract, "functions", []) or []]
    funcs_by_name = {_func_name(func): func for func in funcs if _func_name(func)}
    state_names = _state_names(contracts)
    results = _modifier_candidates(contracts, lines, funcs_by_name)
    edges = _internal_edges(funcs, lines)
    existing_keys = {
        (str(row.get("risk_type") or ""), str(row.get("function_name") or ""), int(row.get("line") or 0))
        for row in existing
        if isinstance(row, dict)
    }
    for func in funcs:
        if _func_visibility(func) not in {"public", "external"} or bool(getattr(func, "is_constructor", False)):
            continue
        start, end, body = _func_text(func, lines)
        name = _func_name(func)
        if not name or _has_authorization_guard(func, body):
            continue
        reachable = _reachable_functions(name, edges)
        ordered_reachable = _ordered_reachable_names(reachable, funcs_by_name)
        reachable_funcs = [funcs_by_name[item] for item in ordered_reachable]
        critical: list[int] = []
        for target in reachable_funcs:
            critical.extend(_critical_write_lines(target, lines, state_names))
        lower_name = name.casefold()
        initializer = lower_name in {"initialize", "init"} and not re.search(r"\b(?:initializer|reinitializer|onlyinitializing)\b", body, re.I)
        admin_named = bool(re.search(r"(?:set|update|change|replace|assign|transfer)(?:owner|admin|guardian|operator|controller|factory|implementation|registry|treasury|fee|limit|reward|reserve)", name, re.I))
        own_critical = _critical_write_lines(func, lines, state_names)
        if initializer or admin_named:
            if own_critical:
                line = own_critical[0]
                results.append({
                    "risk_type": "access_control",
                    "submechanism": "unguarded_initializer_or_admin_state_write",
                    "source_evidence_kind": "unprotected_critical_state_write",
                    "source_grounded": True,
                    "confidence": 0.91,
                    "reason": f"{name}() is externally reachable without an effective authorization guard and writes critical administrative state",
                    "function_name": name,
                    "line": line,
                    "evidence_lines": sorted(set([start, *own_critical[:6]])),
                    "state_write_lines": own_critical[:6],
                })
                continue
        if not critical:
            continue
        # General critical-state propagation: a public root that reaches an
        # unguarded helper writer is itself an access-control candidate.
        target = next((item for item in reachable_funcs if _critical_write_lines(item, lines, state_names)), None)
        if target is None:
            continue
        target_name = _func_name(target)
        if target_name == name and not (_SENSITIVE_WORDS.search(name) or re.search(r"\b(?:owner|admin|guardian|controller|factory|registry|treasury|reward|reserve|fee|limit)\b", body, re.I)):
            # Ordinary user/accounting operations are negative controls unless
            # the source explicitly exposes an administrative sink.
            continue
        target_lines = _critical_write_lines(target, lines, state_names)
        line = target_lines[0]
        results.append({
            "risk_type": "access_control",
            "submechanism": "cross_function_critical_state_write" if target_name != name else "unprotected_critical_state_write",
            "source_evidence_kind": "unprotected_critical_state_write",
            "source_grounded": True,
            "confidence": 0.88,
            "reason": f"{name}() reaches unprotected critical state write in {target_name}() without an effective authorization path",
            "function_name": name,
            "entrypoint_function_name": name,
            "entrypoint_lines": [start],
            "line": line,
            "evidence_lines": sorted(set([start, *target_lines[:6]])),
            "state_write_lines": target_lines[:6],
            "source_call_path": [name, target_name] if target_name != name else [name],
        })
    return results


def reentrancy_candidates(contracts: list[Any], source: str, existing: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    lines = _lines(source)
    funcs = [func for contract in contracts for func in getattr(contract, "functions", []) or []]
    funcs_by_name = {_func_name(func): func for func in funcs if _func_name(func)}
    state_names = _state_names(contracts)
    edges = _internal_edges(funcs, lines)
    results: list[dict[str, Any]] = []
    existing_keys = {
        (str(row.get("risk_type") or ""), str(row.get("function_name") or ""), int(row.get("line") or 0))
        for row in existing
        if isinstance(row, dict)
    }
    for root in funcs:
        if _func_visibility(root) not in {"public", "external"} or bool(getattr(root, "is_constructor", False)):
            continue
        root_name = _func_name(root)
        _, _, root_body = _func_text(root, lines)
        if not root_name or _has_mutex_guard(root, root_body):
            continue
        reachable = _reachable_functions(root_name, edges)
        ordered_reachable = _ordered_reachable_names(reachable, funcs_by_name)
        ordered_writes: list[tuple[int, str]] = []
        for name in ordered_reachable:
            target = funcs_by_name.get(name)
            if target is None:
                continue
            storage_aliases = _storage_aliases(target, lines)
            start, end, _ = _func_text(target, lines)
            for number in range(start, end + 1):
                if _line_has_state_write(lines[number - 1], state_names, storage_aliases):
                    ordered_writes.append((number, name))
        ordered_writes.sort(key=lambda item: (item[0], item[1].casefold(), item[1]))
        if not ordered_writes:
            continue
        for name in ordered_reachable:
            target = funcs_by_name.get(name)
            if target is None or _has_mutex_guard(target, _func_text(target, lines)[2]):
                continue
            for call_line in _external_event_lines(target, lines):
                later = [item for item in ordered_writes if item[0] > call_line]
                # When the call is in a helper, only accept writes in the same
                # helper or a caller/helper reachable after that call.  The
                # source ordering plus call graph keeps this conservative.
                if not later:
                    continue
                write_line, write_function = later[0]
                key = ("reentrancy", root_name, call_line)
                if key in existing_keys:
                    continue
                results.append({
                    "risk_type": "reentrancy",
                    "submechanism": "interprocedural_external_call_before_state_write" if name != root_name or write_function != root_name else "external_call_before_state_write",
                    "source_evidence_kind": "interprocedural_external_call_state_write",
                    "source_grounded": True,
                    "confidence": 0.84,
                    "reason": f"{root_name}() reaches external call @L{call_line} before persistent state write @L{write_line} through {name}()",
                    "function_name": root_name,
                    "entry_function_names": [root_name],
                    "entrypoint_lines": [int(getattr(root, "line_number", 0) or 0)],
                    "execution_order": ["callback", "state_write"],
                    "interprocedural": name != root_name or write_function != root_name,
                    "line": call_line,
                    "evidence_lines": [call_line, write_line],
                    "source_callback_type": "external_call",
                    "function_visibility": _func_visibility(root),
                    "function_modifiers": _modifiers(root),
                    "callback_line_text": lines[call_line - 1].strip(),
                    "state_write_line_text": lines[write_line - 1].strip(),
                })
                existing_keys.add(key)
                break
    return results


def _safe_mint_callback_candidates(
    contracts: list[Any], source: str, existing: Iterable[dict[str, Any]] = ()
) -> list[dict[str, Any]]:
    """Find a narrow post-state-update ERC-721 callback before settlement."""

    lines = _lines(source)
    state_names = _state_names(contracts)
    results: list[dict[str, Any]] = []
    existing_keys = {
        (str(row.get("risk_type") or ""), str(row.get("function_name") or ""), int(row.get("line") or 0))
        for row in existing
        if isinstance(row, dict)
    }
    for contract in contracts:
        for func in getattr(contract, "functions", []) or []:
            if _func_visibility(func) not in {"public", "external"}:
                continue
            if bool(getattr(func, "is_constructor", False)):
                continue
            start, end, body = _func_text(func, lines)
            if _has_mutex_guard(func, body):
                continue
            prior_writes = [
                number
                for number in _critical_write_lines(func, lines, state_names)
                if start < number < end
            ]
            if not prior_writes:
                continue
            for callback_line in range(start, end + 1):
                line = lines[callback_line - 1]
                callback_match = _SAFE_MINT_CALL.search(line)
                if callback_match is None:
                    continue
                receiver_text = line[callback_match.end() :].split(",", 1)[0]
                if not _USER_CONTROLLED_RECEIVER.search(receiver_text):
                    continue
                if not any(write_line < callback_line for write_line in prior_writes):
                    continue
                action = next(
                    (
                        number
                        for number in range(callback_line + 1, end + 1)
                        if _ASSET_ACTION_CALL.search(lines[number - 1])
                    ),
                    None,
                )
                if action is None:
                    continue
                key = ("reentrancy", _func_name(func), callback_line)
                if key in existing_keys:
                    continue
                results.append(
                    {
                        "risk_type": "reentrancy",
                        "submechanism": "erc721_safe_mint_callback_before_external_asset_action",
                        "source_evidence_kind": "post_state_update_receiver_callback",
                        "source_grounded": True,
                        "confidence": 0.86,
                        "reason": (
                            f"{_func_name(func)}() updates persistent state before user-controlled "
                            f"safeMint callback @L{callback_line}, then performs external asset action @L{action}"
                        ),
                        "function_name": _func_name(func),
                        "entry_function_names": [_func_name(func)],
                        "entrypoint_lines": [start],
                        "execution_order": ["callback", "asset_action"],
                        "interprocedural": False,
                        "line": callback_line,
                        "evidence_lines": [callback_line, action],
                        "source_callback_type": "erc721_safe_mint_callback",
                        "function_visibility": _func_visibility(func),
                        "function_modifiers": _modifiers(func),
                        "callback_line_text": line.strip(),
                        "callback_call_text": line.strip(),
                        "asset_action_line_text": lines[action - 1].strip(),
                        "state_write_line_text": lines[max(prior_writes) - 1].strip(),
                        "pre_callback_state_write_lines": [max(prior_writes)],
                    }
                )
                existing_keys.add(key)
    return results


def _typed_bool_methods(source: str) -> set[str]:
    return {
        name
        for name, metadata in _typed_return_methods(source).items()
        if re.search(
            r"\bbool(?:\s+[A-Za-z_]\w*)?\b",
            str(metadata.get("returns") or ""),
            re.I,
        )
    }


def _call_result_is_used(line: str, call_start: int) -> bool:
    """Return whether a qualified call contributes to a surrounding expression."""

    prefix = line[:call_start]
    stripped = prefix.strip()
    if not stripped:
        return False
    if re.search(r"\b(?:assert|emit|if|require|return|revert|try|while)\b", prefix, re.I):
        return True
    if re.search(r"(?:=|,|\(|\[|[+\-*/%&|^<>])\s*$", prefix):
        return True
    return False


def _looks_like_token_receiver(line: str, call_start: int) -> bool:
    """Recognize common token-like receivers when the interface is imported."""

    receiver_prefix = line[:call_start]
    identifiers = re.findall(r"\b[A-Za-z_]\w*\b", receiver_prefix)
    token_words = (
        "token",
        "asset",
        "gift",
        "reward",
        "coin",
        "currency",
        "collateral",
        "underlying",
        "stable",
        "weth",
        "usdc",
        "usdt",
        "dai",
    )
    return any(
        any(word in identifier.casefold() for word in token_words)
        for identifier in identifiers
    )


def unchecked_candidates(contracts: list[Any], source: str, existing: Iterable[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    lines = _lines(source)
    funcs = [func for contract in contracts for func in getattr(contract, "functions", []) or []]
    bool_methods = _typed_bool_methods(source)
    typed_methods = _typed_return_methods(source)
    results: list[dict[str, Any]] = []
    for func in funcs:
        if _func_visibility(func) not in {"public", "external", "internal", "private"}:
            continue
        start, end, body = _func_text(func, lines)
        name = _func_name(func)
        for number in range(start, end + 1):
            line = lines[number - 1]
            low = _LOW_LEVEL.search(line)
            if low:
                before = "\n".join(lines[max(start - 1, number - 4) : number])
                after = "\n".join(lines[number : min(end, number + 6)])
                assigned = bool(re.search(r"(?:bool\s+\w+|\(\s*bool\b|\w+\s*=)\s*[^;]*\.\s*(?:call|send|delegatecall|staticcall)", line, re.I))
                checked = bool(re.search(r"\b(?:require|assert|if|while|try)\s*\([^\n;{}]*(?:success|result|call|send|delegatecall|staticcall)", before + "\n" + after, re.I))
                if checked:
                    continue
                results.append({
                    "risk_type": "unchecked_low_level_calls",
                    "source_evidence_kind": "source_unchecked_low_level_call",
                    "source_grounded": True,
                    "confidence": 0.93,
                    "reason": f"{name}() discards or fails to check low-level call success at @L{number}",
                    "function_name": name,
                    "line": number,
                    "evidence_lines": [number],
                })
                continue
            match = _QUALIFIED_CALL.search(line)
            if not match:
                continue
            method = match.group("method").casefold()
            typed_metadata = typed_methods.get(method)
            imported_token_return = (
                method in _KNOWN_TYPED_TOKEN_METHODS
                and _looks_like_token_receiver(line, match.start())
            )
            if method not in bool_methods and typed_metadata is None and not imported_token_return:
                continue
            if typed_metadata and typed_metadata.get("read_only"):
                continue
            prefix = line[: match.start()]
            if re.search(r"\b(?:require|assert|if|while|try)\b", prefix, re.I):
                continue
            if _call_result_is_used(line, match.start()):
                continue
            if re.search(r"\b(?:safeerc20|transferhelper)\b", line, re.I):
                continue
            results.append({
                "risk_type": "unchecked_low_level_calls",
                "source_evidence_kind": (
                    "typed_return_discard"
                    if method in bool_methods or imported_token_return
                    else "external_return_discard"
                ),
                "typed_return_discard": True,
                "source_typed_return_discard": True,
                "source_grounded": True,
                "confidence": 0.90,
                "reason": f"{name}() discards typed return value from external call {method}() at @L{number}",
                "function_name": name,
                "line": number,
                "evidence_lines": [number],
            })
    # Inline assembly success values: a call in assembly is unchecked unless
    # the same block visibly branches on the returned status.
    for func in funcs:
        name = _func_name(func)
        start, end, body = _func_text(func, lines)
        if not re.search(r"\bassembly\s*\{", body, re.I):
            continue
        for offset, line in enumerate(lines[start - 1 : end]):
            if not re.search(r"\bcall\s*\(", line, re.I):
                continue
            window = "\n".join(lines[max(start - 1, start + offset - 1) : min(end, start + offset + 5)])
            if re.search(r"\b(?:if|switch)\s+iszero\s*\(|\b(?:if|switch)\s+success\b", window, re.I):
                continue
            number = start + offset
            results.append({
                "risk_type": "unchecked_low_level_calls",
                "source_evidence_kind": "assembly_call_status_discard",
                "source_grounded": True,
                "confidence": 0.92,
                "reason": f"{name}() invokes assembly call() without a visible success-status check at @L{number}",
                "function_name": name,
                "line": number,
                "evidence_lines": [number],
            })
    return results


def append_candidates(contracts: list[Any], source: str, existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append deduplicated C2 candidates to an existing AST risk list."""

    additions = [
        *access_control_candidates(contracts, source, existing),
        *reentrancy_candidates(contracts, source, existing),
        *_safe_mint_callback_candidates(contracts, source, existing),
        *unchecked_candidates(contracts, source, existing),
    ]
    seen = {
        (str(row.get("risk_type") or ""), str(row.get("function_name") or ""), int(row.get("line") or 0))
        for row in existing
        if isinstance(row, dict)
    }
    for row in additions:
        key = (str(row.get("risk_type") or ""), str(row.get("function_name") or ""), int(row.get("line") or 0))
        if key in seen:
            continue
        existing.append(row)
        seen.add(key)
    return existing
