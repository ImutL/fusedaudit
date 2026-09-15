"""Source-local candidate decisions for the optimized E1 path.

This module deliberately has no model, database, or benchmark dependency.  It
answers one narrow question: what does the current source prove about one
low-level call's return-status path?
"""

from __future__ import annotations

import re


CONFIRMED = "confirmed"
REFUTED = "refuted"
UNRESOLVED = "unresolved"

_LOW_LEVEL_CALL_PATTERN = re.compile(
    r"\.\s*(?P<method>send|delegatecall|staticcall|call)\b"
    r"(?:\s*\.\s*value\s*\([^()\r\n]*\)\s*)?"
    r"(?:\{[^}\r\n]*\}\s*)?\(",
    re.IGNORECASE,
)
_CONTROL_PATTERN = re.compile(r"\b(?P<kind>require|assert|if|while)\s*\(", re.IGNORECASE)
_CHECK_HELPER_PATTERN = re.compile(
    r"\b(?:verify|check)(?:[A-Za-z_0-9]*)\s*\(", re.IGNORECASE
)
_TYPE_NAMES = {
    "address",
    "bool",
    "bytes",
    "calldata",
    "fixed",
    "int",
    "mapping",
    "memory",
    "string",
    "uint",
    "var",
    "storage",
}


def _strip_comments_preserve_lines(source: str) -> str:
    """Remove comments without changing offsets or line count."""

    result: list[str] = []
    index = 0
    state = "code"
    quote = ""
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if state == "line_comment":
            if char == "\n":
                result.append(char)
                state = "code"
            else:
                result.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                result.extend((" ", " "))
                index += 2
                state = "code"
            else:
                result.append("\n" if char == "\n" else " ")
                index += 1
            continue
        if state == "string":
            result.append(char)
            if char == "\\" and next_char:
                result.append(next_char)
                index += 2
                continue
            if char == quote:
                state = "code"
                quote = ""
            index += 1
            continue
        if char == "/" and next_char == "/":
            result.extend((" ", " "))
            index += 2
            state = "line_comment"
            continue
        if char == "/" and next_char == "*":
            result.extend((" ", " "))
            index += 2
            state = "block_comment"
            continue
        if char in {"\"", "'"}:
            state = "string"
            quote = char
        result.append(char)
        index += 1
    return "".join(result)


def _matching_delimiter(text: str, opening: int, left: str = "(", right: str = ")") -> int:
    if opening < 0 or opening >= len(text) or text[opening] != left:
        return -1
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == left:
            depth += 1
        elif text[index] == right:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _line_start(text: str, offset: int) -> int:
    return text.rfind("\n", 0, max(0, offset)) + 1


def _line_end(text: str, offset: int) -> int:
    end = text.find("\n", max(0, offset))
    return len(text) if end < 0 else end


def _statement_bounds(text: str, call_start: int, call_end: int) -> tuple[int, int]:
    start = max(
        text.rfind(";", 0, call_start),
        text.rfind("{", 0, call_start),
        text.rfind("}", 0, call_start),
    ) + 1
    semicolon = text.find(";", max(call_end, call_start))
    closing_brace = text.find("}", max(call_end, call_start))
    ends = [value for value in (semicolon, closing_brace) if value >= 0]
    end = min(ends) + 1 if ends else len(text)
    return start, end


def _extract_assignment_names(prefix: str) -> list[str]:
    """Return identifiers on the left of the last assignment operator."""

    equal = prefix.rfind("=")
    if equal < 0:
        return []
    if equal > 0 and prefix[equal - 1] in "=!<>":
        return []
    if equal + 1 < len(prefix) and prefix[equal + 1] == "=":
        return []
    lhs = prefix[:equal].rsplit(";", 1)[-1].strip().strip("() ")
    if not lhs:
        return []
    names: list[str] = []
    for part in lhs.split(","):
        identifiers = re.findall(r"\b([A-Za-z_]\w*)\b", part)
        if not identifiers:
            continue
        name = identifiers[-1]
        if name.casefold() in _TYPE_NAMES:
            continue
        if name not in names:
            names.append(name)
    return names


def _control_consuming_call(text: str, statement_start: int, call_start: int) -> dict | None:
    """Find a control predicate that directly consumes the call result."""

    controls = list(_CONTROL_PATTERN.finditer(text, statement_start, call_start))
    for control in reversed(controls):
        opening = text.find("(", control.start(), control.end())
        closing = _matching_delimiter(text, opening)
        if closing >= call_start:
            return {
                "kind": control.group("kind").casefold(),
                "line": _line_number(text, control.start()),
            }

    try_match = re.search(r"\btry\b", text[statement_start:call_start], re.IGNORECASE)
    if try_match:
        return {
            "kind": "try",
            "line": _line_number(text, statement_start + try_match.start()),
        }
    return None


def _assigned_return_checks(
    text: str,
    names: list[str],
    call_end: int,
    function_end: int,
) -> list[dict]:
    """Find explicit status checks for assigned return values."""

    tail = text[call_end:function_end]
    checks: list[dict] = []
    for control in _CONTROL_PATTERN.finditer(tail):
        opening = tail.find("(", control.start(), control.end())
        closing = _matching_delimiter(tail, opening)
        if closing < 0:
            continue
        predicate = tail[opening + 1:closing]
        if not any(re.search(rf"\b{re.escape(name)}\b", predicate) for name in names):
            continue
        absolute = call_end + control.start()
        checks.append({
            "kind": control.group("kind").casefold(),
            "line": _line_number(text, absolute),
        })

    for helper in _CHECK_HELPER_PATTERN.finditer(tail):
        opening = tail.find("(", helper.start(), helper.end())
        closing = _matching_delimiter(tail, opening)
        if closing < 0:
            continue
        arguments = tail[opening + 1:closing]
        if not any(re.search(rf"\b{re.escape(name)}\b", arguments) for name in names):
            continue
        absolute = call_end + helper.start()
        checks.append({
            "kind": helper.group(0).split("(", 1)[0].strip().casefold(),
            "line": _line_number(text, absolute),
        })
    checks.sort(key=lambda item: (int(item["line"]), str(item["kind"])))
    return checks


def _function_ranges(text: str) -> list[dict]:
    """Find coarse function ranges for callers that do not already have one."""

    ranges: list[dict] = []
    declaration = re.compile(
        r"\bfunction\s+(?P<name>[A-Za-z_]\w*)\b|"
        r"\b(?P<special>fallback|receive|constructor)\b",
        re.IGNORECASE,
    )
    for match in declaration.finditer(text):
        opening = text.find("{", match.end())
        terminator = text.find(";", match.end())
        if opening < 0 or (terminator >= 0 and terminator < opening):
            continue
        closing = _matching_delimiter(text, opening, "{", "}")
        if closing < 0:
            continue
        ranges.append({
            "name": (match.group("name") or match.group("special") or "").strip(),
            "start": match.start(),
            "end": closing + 1,
        })
    return ranges


def _select_function_range(
    text: str,
    line_number: int,
    function_name: str,
    start_line: int | None,
    end_line: int | None,
) -> dict:
    lines = text.splitlines()
    if start_line is not None and end_line is not None:
        start_line = max(1, int(start_line))
        end_line = min(len(lines), max(start_line, int(end_line)))
        start_offset = sum(len(line) + 1 for line in lines[: start_line - 1])
        end_offset = sum(len(line) + 1 for line in lines[:end_line])
        return {
            "name": function_name or "source",
            "start": start_offset,
            "end": min(len(text), end_offset),
            "start_line": start_line,
            "end_line": end_line,
        }

    for item in _function_ranges(text):
        start = _line_number(text, item["start"])
        end = _line_number(text, item["end"])
        if start <= line_number <= end:
            return {
                **item,
                "start_line": start,
                "end_line": end,
            }
    return {
        "name": function_name or "source",
        "start": 0,
        "end": len(text),
        "start_line": 1,
        "end_line": len(lines),
    }


def _candidate_decision(
    text: str,
    match: re.Match,
    call_end: int,
    function_end: int,
) -> dict:
    call_start = match.start()
    statement_start, statement_end = _statement_bounds(text, call_start, call_end)
    control = _control_consuming_call(text, statement_start, call_start)
    line = _line_number(text, call_start)
    method = match.group("method").casefold()
    if control is not None:
        return {
            "state": REFUTED,
            "line": line,
            "method": method,
            "call_text": text[call_start:call_end].strip(),
            "binding_names": [],
            "check_lines": [int(control["line"])],
            "check_kinds": [str(control["kind"])],
            "evidence_lines": sorted({line, int(control["line"])}),
            "reason": f"The {method} return is consumed directly by {control['kind']}().",
        }

    prefix = text[statement_start:call_start]
    names = _extract_assignment_names(prefix)
    if not names:
        before = prefix.strip().casefold()
        if re.search(r"\breturn\b", before) or re.search(r"[A-Za-z_]\w*\s*\([^;{}]*$", prefix):
            return {
                "state": UNRESOLVED,
                "line": line,
                "method": method,
                "call_text": text[call_start:call_end].strip(),
                "binding_names": [],
                "check_lines": [],
                "check_kinds": [],
                "evidence_lines": [line],
                "reason": "The low-level return is propagated or passed to another expression; its final consumer is outside this local candidate path.",
            }
        return {
            "state": CONFIRMED,
            "line": line,
            "method": method,
            "call_text": text[call_start:call_end].strip(),
            "binding_names": [],
            "check_lines": [],
            "check_kinds": [],
            "evidence_lines": [line],
            "reason": "The low-level call is an expression statement and its return status is discarded.",
        }

    checks = _assigned_return_checks(text, names, call_end, function_end)
    if checks:
        check_lines = [int(item["line"]) for item in checks]
        check_kinds = [str(item["kind"]) for item in checks]
        return {
            "state": REFUTED,
            "line": line,
            "method": method,
            "call_text": text[call_start:call_end].strip(),
            "binding_names": names,
            "check_lines": check_lines,
            "check_kinds": check_kinds,
            "evidence_lines": sorted({line, *check_lines}),
            "reason": "The assigned low-level return status is consumed by an explicit control or verification check.",
        }

    tail = text[call_end:function_end]
    non_control_use = any(
        re.search(rf"\b{re.escape(name)}\b", tail) for name in names
    )
    if non_control_use and re.search(
        rf"\b(?:return|emit)\b[^;]*\b(?:{'|'.join(re.escape(name) for name in names)})\b",
        tail,
        re.IGNORECASE,
    ):
        state = UNRESOLVED
        reason = "The assigned low-level return is propagated without a locally visible status check."
    else:
        state = CONFIRMED
        reason = "The assigned low-level return has no locally visible control or verification consumer."
    return {
        "state": state,
        "line": line,
        "method": method,
        "call_text": text[call_start:call_end].strip(),
        "binding_names": names,
        "check_lines": [],
        "check_kinds": [],
        "evidence_lines": [line],
        "reason": reason,
    }


def analyze_unchecked_low_level_call_candidates(
    source: str,
    *,
    function_name: str = "",
    start_line: int | None = None,
    end_line: int | None = None,
) -> list[dict]:
    """Return source-local decisions for every low-level call in a function."""

    original = str(source or "")
    if not original:
        return []
    text = _strip_comments_preserve_lines(original)
    line_hint = int(start_line or 1)
    function = _select_function_range(
        text,
        line_hint,
        function_name,
        start_line,
        end_line,
    )
    search_start = int(function["start"])
    search_end = int(function["end"])
    rows: list[dict] = []
    for match in _LOW_LEVEL_CALL_PATTERN.finditer(text, search_start, search_end):
        opening = text.find("(", match.start(), match.end())
        closing = _matching_delimiter(text, opening)
        if closing < 0 or closing >= search_end:
            continue
        row = _candidate_decision(text, match, closing + 1, search_end)
        row["function_name"] = str(function["name"] or "source")
        row["call_end_line"] = _line_number(text, closing)
        row["function_start_line"] = int(function["start_line"])
        row["function_end_line"] = int(function["end_line"])
        rows.append(row)
    return rows


def candidate_at_line(
    source: str,
    line_number: int,
    *,
    function_name: str = "",
) -> dict | None:
    """Return a low-level call candidate whose expression contains a line."""

    rows = analyze_unchecked_low_level_call_candidates(
        source,
        function_name=function_name,
        start_line=None,
        end_line=None,
    )
    exact = [row for row in rows if int(row.get("line", 0)) == int(line_number)]
    if exact:
        return exact[0]
    containing = [
        row
        for row in rows
        if int(row.get("line", 0)) <= int(line_number) <= int(row.get("call_end_line", 0))
    ]
    return containing[0] if len(containing) == 1 else None


__all__ = [
    "CONFIRMED",
    "REFUTED",
    "UNRESOLVED",
    "analyze_unchecked_low_level_call_candidates",
    "candidate_at_line",
]
