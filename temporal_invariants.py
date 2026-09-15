"""
Recovered from .pyc: Post-v20 Phase 2 temporal invariants detector.
Phase 1 base rules (G1/G2/G3/oracle/lifecycle) were lost with the original
temporal_invariants.py. This file preserves the Phase 2 detection rules
for signed_authorization_lifetime, cross_chain_message_freshness,
deadline_bypass, phase_boundary_overlap, and oracle_timestamp_arithmetic families.

New detector SHA will be computed after recovery.
"""
from __future__ import annotations

import re
import sys
from typing import Any, NamedTuple

# ── Phase 1 helpers (reconstructed from Phase 2 wrapper references) ──

class FunctionRegion(NamedTuple):
    name: str
    text: str
    start_line: int
    end_line: int


def _strip_comments_preserve_layout(text: str) -> str:
    """Strip Solidity comments while preserving line structure and strings.

    A line such as ``//* ASCII art`` contains the byte sequence ``/*`` but is
    still a line comment.  A regex pass over block comments first would consume
    the remainder of the file, so comments are scanned in source order.
    """
    output: list[str] = []
    index = 0
    in_line_comment = False
    in_block_comment = False
    quote = ""
    escaped = False
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_line_comment:
            if char == "\n":
                output.append(char)
                in_line_comment = False
            else:
                output.append(" ")
            index += 1
            continue
        if in_block_comment:
            if char == "*" and next_char == "/":
                output.extend((" ", " "))
                index += 2
                in_block_comment = False
            elif char == "\n":
                output.append("\n")
                index += 1
            else:
                output.append(" ")
                index += 1
            continue
        if quote:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            output.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            output.extend((" ", " "))
            index += 2
            in_line_comment = True
            continue
        if char == "/" and next_char == "*":
            output.extend((" ", " "))
            index += 2
            in_block_comment = True
            continue
        output.append(char)
        index += 1
    return "".join(output)


def extract_function_regions(source: str) -> list[FunctionRegion]:
    """Extract function regions from Solidity source."""
    functions: list[FunctionRegion] = []
    pattern = re.compile(
        r"function\s+(?P<name>[A-Za-z_]\w*)\s*\([^)]*\)"
        r"\s*(?:(?:external|public|internal|private|view|pure|payable|returns\s*\([^)]*\))\s*)*"
        r"\{",
        re.IGNORECASE,
    )
    for m in pattern.finditer(source):
        name = m.group("name")
        start = m.start()
        brace_count = 0
        pos = m.end() - 1
        found = False
        for i in range(pos, len(source)):
            if source[i] == "{":
                brace_count += 1
            elif source[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    pos = i + 1
                    found = True
                    break
        if not found:
            continue
        text = source[start:pos]
        start_line = source[:start].count("\n") + 1
        end_line = source[:pos].count("\n") + 1
        functions.append(FunctionRegion(name, text, start_line, end_line))
    return functions


def _add_evidence(
    findings: list[dict],
    source: str,
    functions: list[FunctionRegion],
    subtype: str,
    confidence: float,
    reason: str,
    offset: int,
    related_functions: list[str] | None = None,
    symbols: list[str] | None = None,
    clock_domains: list[str] | None = None,
    operation: str | None = None,
) -> None:
    """Add a finding to the findings list."""
    line_num = source[:offset].count("\n") + 1 if offset >= 0 else 0
    findings.append({
        "subtype": subtype,
        "confidence": confidence,
        "reason": reason,
        "line": line_num,
        "functions": related_functions or [],
        "symbols": symbols or [],
        "primary_category": "time_manipulation",
        "time_role": "primary",
        "clock_domains": clock_domains or ["local_block"],
        "operation": operation or "unknown",
        "evidence": source[offset:offset + 80].strip() if offset >= 0 and offset < len(source) else "",
    })


def _matching_delimiter(text: str, start: int, opening: str, closing: str) -> int:
    """Return the matching closing delimiter, or -1 for malformed text."""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _function_parameter_names(function_text: str) -> set[str]:
    header_end = function_text.find("{")
    header = function_text if header_end < 0 else function_text[:header_end]
    opening = header.find("(")
    if opening < 0:
        return set()
    closing = _matching_delimiter(header, opening, "(", ")")
    if closing < 0:
        return set()

    names = set()
    type_only = {
        "address", "bool", "bytes", "string", "uint", "int",
        "memory", "storage", "calldata", "payable",
    }
    for declaration in header[opening + 1:closing].split(","):
        tokens = re.findall(r"[A-Za-z_]\w*", declaration)
        if tokens and tokens[-1] not in type_only and not re.fullmatch(
            r"u?int\d*|bytes\d*", tokens[-1]
        ):
            names.add(tokens[-1])
    return names


def _declared_before_function(source_prefix: str, name: str) -> bool:
    declaration = re.compile(
        r"(?:^|\n)\s*"
        r"(?:address(?:\s+payable)?|u?int(?:\d+)?|bytes(?:\d+)?|bool|string)"
        r"(?:\s+(?:public|private|internal|external|constant|immutable))*"
        + r"\s+" + re.escape(name) + r"\b",
        re.IGNORECASE,
    )
    return bool(declaration.search(source_prefix))


def _extract_braced_regions(source: str, keyword: str) -> list[FunctionRegion]:
    """Extract named Solidity regions whose header may contain custom modifiers."""

    regions: list[FunctionRegion] = []
    pattern = re.compile(
        rf"\b{re.escape(keyword)}\s+(?P<name>[A-Za-z_]\w*)"
        rf"\s*(?:\([^)]*\))?\s*\{{",
        re.IGNORECASE,
    )
    for match in pattern.finditer(source):
        opening = match.end() - 1
        closing = _matching_delimiter(source, opening, "{", "}")
        if closing < 0:
            continue
        regions.append(FunctionRegion(
            match.group("name"),
            source[match.start():closing + 1],
            source[:match.start()].count("\n") + 1,
            source[:closing + 1].count("\n") + 1,
        ))
    return regions


def extract_modifier_regions(source: str) -> list[FunctionRegion]:
    return _extract_braced_regions(source, "modifier")


def extract_function_regions_with_modifiers(source: str) -> list[FunctionRegion]:
    """Extract function bodies without dropping custom modifier names from headers."""

    regions: list[FunctionRegion] = []
    pattern = re.compile(
        r"\bfunction\s+(?P<name>[A-Za-z_]\w*)\s*\([^)]*\)"
        r"(?P<header>[^{}]{0,500})\{",
        re.IGNORECASE,
    )
    for match in pattern.finditer(source):
        opening = match.end() - 1
        closing = _matching_delimiter(source, opening, "{", "}")
        if closing < 0:
            continue
        regions.append(FunctionRegion(
            match.group("name"),
            source[match.start():closing + 1],
            source[:match.start()].count("\n") + 1,
            source[:closing + 1].count("\n") + 1,
        ))
    return regions


def _modifier_time_window_invariants(source: str) -> list[dict]:
    """Link a modifier timestamp condition to a non-view state/phase operation."""

    findings: list[dict] = []
    modifiers = extract_modifier_regions(source)
    functions = extract_function_regions_with_modifiers(source)
    for modifier in modifiers:
        body = _strip_comments_preserve_layout(modifier.text)
        condition_line_offset = None
        condition_text = ""
        for offset, line in enumerate(body.splitlines()):
            lowered = line.lower()
            if not re.search(r"\b(?:now|block\.timestamp)\b", lowered):
                continue
            if not re.search(r"(?:<|<=|>|>=|==)", line):
                continue
            condition_line_offset = offset
            condition_text = line.strip()
            break
        if condition_line_offset is None:
            continue
        has_window_shape = bool(
            len(re.findall(r"(?:<|<=|>|>=|==)", condition_text)) >= 2
            or re.search(r"\b(?:now|block\.timestamp)\b[^\n;]*(?:\+|-)[^\n;]*\b(?:days?|hours?|minutes?|seconds?)\b", condition_text, re.IGNORECASE)
        )
        if not has_window_shape:
            continue

        protected: list[FunctionRegion] = []
        for function in functions:
            header = function.text.split("{", 1)[0]
            if not re.search(rf"\b{re.escape(modifier.name)}\b", header):
                continue
            if re.search(r"\b(?:view|pure)\b", header, re.IGNORECASE):
                continue
            function_body = function.text.split("{", 1)[1]
            has_state_write = bool(re.search(
                r"\b[A-Za-z_]\w*(?:\s*\[[^\]]+\])?(?:\.[A-Za-z_]\w*)?\s*"
                r"(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)",
                function_body,
            ))
            has_external_operation = bool(re.search(
                r"\.(?:call|call\.value|send|transfer|transferFrom|safeTransfer|approve)\s*\(",
                function_body,
                re.IGNORECASE,
            ))
            if has_state_write or has_external_operation:
                protected.append(function)
        if not protected:
            continue

        modifier_offset = source.find(modifier.text)
        if modifier_offset < 0:
            continue
        condition_offset = modifier_offset + sum(
            len(line) + 1 for line in body.splitlines()[:condition_line_offset]
        )
        condition_symbols = {
            symbol
            for symbol in re.findall(r"\b[A-Za-z_]\w*\b", condition_text)
            if symbol.lower() not in {"now", "block", "timestamp", "if", "days", "day"}
        }
        temporal_operation_lines: list[int] = []
        for function in functions:
            header = function.text.split("{", 1)[0]
            if re.search(r"\b(?:view|pure)\b", header, re.IGNORECASE):
                continue
            function_body = _strip_comments_preserve_layout(function.text.split("{", 1)[1])
            if not re.search(r"\b(?:now|block\.timestamp)\b", function_body, re.IGNORECASE):
                continue
            if condition_symbols and not any(
                re.search(rf"\b{re.escape(symbol)}\b", function_body)
                for symbol in condition_symbols
            ):
                continue
            has_state_write = bool(re.search(
                r"\b[A-Za-z_]\w*(?:\s*\[[^\]]+\])?(?:\.[A-Za-z_]\w*)?\s*"
                r"(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)",
                function_body,
            ))
            has_external_operation = bool(re.search(
                r"\.(?:call|call\.value|send|transfer|transferFrom|safeTransfer|approve)\s*\(",
                function_body,
                re.IGNORECASE,
            ))
            if not (has_state_write or has_external_operation):
                continue
            for match in re.finditer(r"\b(?:now|block\.timestamp)\b", function_body, re.IGNORECASE):
                header_line_offset = function.text.split("{", 1)[0].count("\n")
                line = (
                    function.start_line
                    + header_line_offset
                    + function_body[:match.start()].count("\n")
                )
                if line not in temporal_operation_lines:
                    temporal_operation_lines.append(line)
        _add_evidence(
            findings,
            source,
            functions,
            "modifier_time_window_protected_operation",
            0.86,
            f"Modifier {modifier.name}() uses {condition_text} and protects phase/state operation(s) in "
            f"{', '.join(function.name for function in protected)}(); the timestamp window is part of the protected transition.",
            condition_offset,
            related_functions=[modifier.name, *[function.name for function in protected]],
            symbols=[modifier.name, "block.timestamp", "now"],
            clock_domains=["modifier", "phase_boundary"],
            operation="modifier_timestamp_condition_to_protected_operation",
        )
        findings[-1]["source_anchor_lines"] = [
            findings[-1]["line"],
            *temporal_operation_lines,
        ]
    return findings


def _timestamp_guard_condition(condition: str) -> bool:
    return bool(
        re.search(r"\b(?:now|block\.timestamp)\b", condition, re.IGNORECASE)
        and re.search(r"(?:<=|>=|==|<|>)", condition)
        and re.search(
            r"(?:\b(?:now|block\.timestamp)\b\s*(?:<=|>=|==|<|>)|"
            r"(?:<=|>=|==|<|>)\s*\b(?:now|block\.timestamp)\b)",
            condition,
            re.IGNORECASE,
        )
    )


def _guard_matches(text: str) -> list[re.Match[str]]:
    return list(re.finditer(
        r"\b(?:require|assert|if|while)\s*\((?P<condition>[^;{}]+)\)",
        text,
        re.IGNORECASE | re.DOTALL,
    ))


def _source_line_numbers(text: str, pattern: str) -> list[int]:
    return [
        text[:match.start()].count("\n") + 1
        for match in re.finditer(pattern, text, re.IGNORECASE)
    ]


_ASSET_OPERATION_PATTERN = (
    r"\b(?:_?transfer|safeTransfer(?:From)?|transferFrom|_?mint|burn|"
    r"withdraw|redeem|claim|release|unlock|deposit|stake|exit|payout|sendValue)\s*\("
)
_STATE_WRITE_PATTERN = (
    r"\b[A-Za-z_]\w*(?:\s*\[[^\]]+\])?(?:\.[A-Za-z_]\w*)?\s*"
    r"(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)"
)


def _line_start_offset(region: FunctionRegion, region_offset: int, relative_line: int) -> int:
    lines = region.text.splitlines(keepends=True)
    return region_offset + sum(len(line) for line in lines[:relative_line])


def _state_duration_gate_invariants(source: str) -> list[dict]:
    """Link a state-relative timestamp gate to an asset operation in one function."""

    findings: list[dict] = []
    functions = extract_function_regions_with_modifiers(source)
    search_start = 0
    for function in functions:
        header_end = function.text.find("{")
        if header_end < 0:
            continue
        header = function.text[:header_end]
        if re.search(r"\b(?:view|pure)\b", header, re.IGNORECASE):
            continue
        function_offset = source.find(function.text, search_start)
        if function_offset < 0:
            continue
        search_start = function_offset + len(function.text)
        body = _strip_comments_preserve_layout(function.text[header_end + 1:])
        header_line_offset = header.count("\n")
        asset_lines = _source_line_numbers(body, _ASSET_OPERATION_PATTERN)
        if not asset_lines:
            continue
        state_lines = _source_line_numbers(body, _STATE_WRITE_PATTERN)
        for guard in _guard_matches(body):
            condition = guard.group("condition")
            if not _timestamp_guard_condition(condition):
                continue
            if not re.search(
                r"\b(?:now|block\.timestamp)\b\s*(?:<=|>=|<|>)\s*"
                r"[^\n;]*(?:\+|-)\s*(?:[A-Za-z_]\w*|\d+\s*"
                r"(?:seconds?|minutes?|hours?|days?))\b",
                condition,
                re.IGNORECASE,
            ):
                continue
            if not state_lines:
                continue
            guard_line = function.start_line + body[:guard.start()].count("\n")
            source_offset = _line_start_offset(
                function,
                function_offset,
                header.count("\n") + body[:guard.start()].count("\n"),
            )
            operation_lines = [
                function.start_line,
                *(
                    function.start_line + line - 1
                    for line in [*asset_lines, *state_lines]
                ),
            ]
            _add_evidence(
                findings,
                source,
                functions,
                "state_duration_gate_controls_asset_release",
                0.86,
                f"{function.name}() enforces {condition.strip()} before an asset operation and state update; the duration gate controls when value can be released.",
                source_offset,
                related_functions=[function.name],
                symbols=["block.timestamp", "duration"],
                clock_domains=["local_block", "duration_gate"],
                operation="timestamp_duration_gate_to_asset_release",
            )
            findings[-1]["source_anchor_lines"] = [guard_line, *operation_lines]
            break
    return findings


def _modifier_timestamp_gate_asset_invariants(source: str) -> list[dict]:
    """Link a single timestamp modifier gate to a protected asset operation."""

    findings: list[dict] = []
    modifiers = extract_modifier_regions(source)
    functions = extract_function_regions_with_modifiers(source)
    for modifier in modifiers:
        modifier_body = _strip_comments_preserve_layout(modifier.text)
        modifier_offset = source.find(modifier.text)
        if modifier_offset < 0:
            continue
        for guard in _guard_matches(modifier_body):
            condition = guard.group("condition")
            if not _timestamp_guard_condition(condition):
                continue
            if len(re.findall(r"(?:<=|>=|==|<|>)", condition)) != 1:
                continue
            protected: list[FunctionRegion] = []
            protected_lines: list[int] = []
            for function in functions:
                header_end = function.text.find("{")
                header = function.text if header_end < 0 else function.text[:header_end]
                if not re.search(rf"\b{re.escape(modifier.name)}\b", header):
                    continue
                if re.search(r"\b(?:view|pure)\b", header, re.IGNORECASE):
                    continue
                body = _strip_comments_preserve_layout(function.text[header_end + 1:])
                asset_lines = _source_line_numbers(body, _ASSET_OPERATION_PATTERN)
                state_lines = _source_line_numbers(body, _STATE_WRITE_PATTERN)
                if not asset_lines or not state_lines:
                    continue
                protected.append(function)
                protected_lines.extend(
                    [
                        function.start_line,
                        *(
                            function.start_line + line - 1
                            for line in [*asset_lines, *state_lines]
                        ),
                    ]
                )
            if not protected:
                continue
            guard_line = modifier.start_line + modifier_body[:guard.start()].count("\n")
            source_offset = _line_start_offset(
                modifier,
                modifier_offset,
                modifier_body[:guard.start()].count("\n"),
            )
            _add_evidence(
                findings,
                source,
                functions,
                "modifier_timestamp_gate_protected_asset_operation",
                0.86,
                f"Modifier {modifier.name}() enforces {condition.strip()} before protected asset operation(s) in "
                f"{', '.join(function.name for function in protected)}(); the timestamp gate controls entry to the asset transition.",
                source_offset,
                related_functions=[modifier.name, *[function.name for function in protected]],
                symbols=[modifier.name, "block.timestamp"],
                clock_domains=["modifier", "phase_boundary"],
                operation="modifier_timestamp_gate_to_protected_asset_operation",
            )
            findings[-1]["source_anchor_lines"] = [guard_line, *protected_lines]
            break
    return findings


_LIFECYCLE_BOUNDARY_NAME_RE = re.compile(
    r"(?:start(?:time|block)?|end(?:time|block)?|periodfinish|"
    r"finish(?:time|block)?|openingtime|closingtime|launch(?:time)?|"
    r"deadline|expiry|expiration)",
    re.IGNORECASE,
)


def _lifecycle_boundary_timestamp_invariants(source: str) -> list[dict]:
    """Link timestamp boundary setters to a state-gated asset lifecycle.

    A number of DAppSCAN contracts store ``block.timestamp`` in public
    start/end setters while the actual asset path is guarded by a separate
    lifecycle modifier.  The existing temporal rules intentionally focus on
    timestamp-bearing guards, so this shape otherwise disappears.  This rule
    is deliberately narrow: it requires both timestamp boundary writes and a
    non-timestamp modifier (or local guard) that references those same fields
    before a stateful/asset operation.
    """

    findings: list[dict] = []
    function_regions = _function_regions_with_offsets(source)
    functions = [item[0] for item in function_regions]
    offsets = {id(function): offset for function, offset in function_regions}
    boundary_writes: list[dict[str, object]] = []

    for function in functions:
        header_end = function.text.find("{")
        header = function.text if header_end < 0 else function.text[:header_end]
        if not re.search(r"\b(?:public|external)\b", header, re.IGNORECASE):
            continue
        body = _strip_comments_preserve_layout(function.text[header_end + 1:])
        header_line_offset = header.count("\n")
        for match in re.finditer(
            r"\b(?P<lhs>[A-Za-z_]\w*)\s*=\s*(?:block\.timestamp|now)\b",
            body,
            re.IGNORECASE,
        ):
            lhs = match.group("lhs")
            if not _LIFECYCLE_BOUNDARY_NAME_RE.search(lhs):
                continue
            line = function.start_line + header_line_offset + body[:match.start()].count("\n")
            boundary_writes.append({
                "function": function,
                "name": lhs,
                "line": line,
                "offset": offsets.get(id(function), -1) + header_end + 1 + match.start(),
            })

    if len(boundary_writes) < 2:
        return findings

    boundary_names = {str(item["name"]) for item in boundary_writes}
    gates: list[dict[str, object]] = []

    # First inspect modifiers.  We exclude timestamp-bearing modifiers because
    # the dedicated modifier rules already provide a more precise invariant.
    for modifier in extract_modifier_regions(source):
        body = _strip_comments_preserve_layout(modifier.text)
        if re.search(r"\b(?:now|block\.timestamp)\b", body, re.IGNORECASE):
            continue
        refs = {
            name for name in boundary_names
            if re.search(rf"\b{re.escape(name)}\b", body)
        }
        if len(refs) < 2 or not re.search(r"(?:==|!=|<=|>=|<|>)", body):
            continue
        gate_line = modifier.start_line
        for guard in _guard_matches(body):
            if any(re.search(rf"\b{re.escape(name)}\b", guard.group("condition")) for name in refs):
                gate_line = modifier.start_line + body[:guard.start()].count("\n")
                break
        gates.append({"name": modifier.name, "refs": refs, "line": gate_line, "function": None})

    # Also support an inline state gate with the same shape.
    for function in functions:
        header_end = function.text.find("{")
        header = function.text if header_end < 0 else function.text[:header_end]
        if re.search(r"\b(?:view|pure)\b", header, re.IGNORECASE):
            continue
        body = _strip_comments_preserve_layout(function.text[header_end + 1:])
        header_line_offset = header.count("\n")
        if re.search(r"\b(?:now|block\.timestamp)\b", body, re.IGNORECASE):
            continue
        refs = {
            name for name in boundary_names
            if re.search(rf"\b{re.escape(name)}\b", body)
        }
        if len(refs) < 2:
            continue
        guard = next(
            (item for item in _guard_matches(body)
             if len({name for name in refs if re.search(rf"\b{re.escape(name)}\b", item.group("condition"))}) >= 2),
            None,
        )
        if guard is not None:
            gates.append({
                "name": function.name,
                "refs": refs,
                "line": function.start_line + header_line_offset + body[:guard.start()].count("\n"),
                "function": function,
            })

    if not gates:
        return findings

    asset_lines_by_function: dict[int, list[int]] = {}
    for function in functions:
        header_end = function.text.find("{")
        header = function.text if header_end < 0 else function.text[:header_end]
        if re.search(r"\b(?:view|pure)\b", header, re.IGNORECASE):
            continue
        body = _strip_comments_preserve_layout(function.text[header_end + 1:])
        asset_lines = _source_line_numbers(body, _ASSET_OPERATION_PATTERN)
        state_lines = _source_line_numbers(body, _STATE_WRITE_PATTERN)
        timestamp_lines = _source_line_numbers(body, r"\b(?:now|block\.timestamp)\b")
        if asset_lines or state_lines or timestamp_lines:
            asset_lines_by_function[id(function)] = [
                function.start_line + header_line_offset + line - 1
                for line in [*asset_lines, *state_lines, *timestamp_lines]
            ]

    for gate in gates:
        refs = set(gate["refs"])
        gate_line = int(gate["line"])
        gate_name = str(gate["name"])
        protected: list[FunctionRegion] = []
        for function in functions:
            header_end = function.text.find("{")
            header = function.text if header_end < 0 else function.text[:header_end]
            uses_modifier = gate["function"] is None and re.search(
                rf"\b{re.escape(gate_name)}\b", header
            )
            is_inline_gate = gate["function"] is function
            if not (uses_modifier or is_inline_gate):
                continue
            if id(function) in asset_lines_by_function:
                protected.append(function)
        if not protected:
            continue

        linked_writes = [item for item in boundary_writes if str(item["name"]) in refs]
        if len(linked_writes) < 2:
            continue
        setter_lines = [int(item["line"]) for item in linked_writes]
        for item in linked_writes:
            function = item["function"]
            line = int(item["line"])
            _add_evidence(
                findings,
                source,
                functions,
                "lifecycle_boundary_timestamp_transition",
                0.86,
                f"{function.name}() records {item['name']} from block.timestamp while {gate_name}() gates the same lifecycle before asset/state operations in {', '.join(fn.name for fn in protected)}().",
                int(item["offset"]),
                related_functions=[function.name],
                symbols=sorted(refs),
                clock_domains=["lifecycle_boundary", "local_block"],
                operation="timestamp_boundary_setter_to_state_gated_asset_lifecycle",
            )
            findings[-1]["source_anchor_lines"] = sorted(set([line, gate_line, *setter_lines]))

        for function in protected:
            body = _strip_comments_preserve_layout(function.text[function.text.find("{") + 1:])
            timestamp_match = re.search(r"\b(?:now|block\.timestamp)\b", body, re.IGNORECASE)
            if timestamp_match is None:
                continue
            function_offset = offsets.get(id(function), -1)
            header_end = function.text.find("{")
            header_line_offset = function.text[:header_end].count("\n") if header_end >= 0 else 0
            line = function.start_line + header_line_offset + body[:timestamp_match.start()].count("\n")
            _add_evidence(
                findings,
                source,
                functions,
                "lifecycle_boundary_timestamp_transition",
                0.86,
                f"{function.name}() performs a timestamped state/asset transition under {gate_name}(), whose start/end boundary is set by public lifecycle functions.",
                function_offset + header_end + 1 + timestamp_match.start(),
                related_functions=[function.name],
                symbols=sorted(refs),
                clock_domains=["lifecycle_boundary", "local_block"],
                operation="timestamped_asset_operation_under_lifecycle_gate",
            )
            findings[-1]["source_anchor_lines"] = sorted(set([line, gate_line, *setter_lines]))

    # Keep one evidence row per function/locus even when multiple gates refer to
    # the same boundary pair.
    deduped: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for finding in findings:
        key = (str((finding.get("functions") or [""])[0]), int(finding.get("line") or 0))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _timestamp_gated_entropy_invariants(source: str) -> list[dict]:
    """Detect timestamp gates that guard miner-influenced entropy outcomes."""

    findings: list[dict] = []
    regions = _function_regions_with_offsets(source)
    state_names = {
        item["name"]
        for item in _ir_symbol_declarations(source, regions)
        if item.get("kind") == "state_or_struct"
    }
    for function, function_offset in regions:
        header_end = function.text.find("{")
        header = function.text[:header_end] if header_end >= 0 else function.text
        if not re.search(r"\b(?:public|external)\b", header, re.IGNORECASE):
            continue
        body = _strip_comments_preserve_layout(function.text[header_end + 1:])
        if not re.search(
            r"\b(?:blockhash|block\.coinbase|block\.difficulty|block\.prevrandao)\b",
            body,
            re.IGNORECASE,
        ):
            continue
        guard = next(
            (
                item for item in _ir_guard_records(function)
                if re.search(r"\b(?:block\.timestamp|now)\b", item["condition"], re.IGNORECASE)
            ),
            None,
        )
        if guard is None:
            continue
        entropy_match = re.search(
            r"\b(?:blockhash|block\.coinbase|block\.difficulty|block\.prevrandao)\b",
            body,
            re.IGNORECASE,
        )
        state_match = None
        for candidate in _IR_ASSIGNMENT_RE.finditer(body):
            lhs = re.sub(r"\s+", "", candidate.group("lhs"))
            base_name = re.split(r"[.[]", lhs, maxsplit=1)[0]
            if base_name in state_names or "." in lhs or "[" in lhs:
                state_match = candidate
                break
        if entropy_match is None or state_match is None:
            continue
        state_line = function.start_line + body[:state_match.start()].count("\n")
        guard_line = guard["line"]
        _add_evidence(
            findings,
            source,
            [item[0] for item in regions],
            "timestamp_gated_miner_influenced_entropy",
            0.93,
            f"{function.name}() gates a miner-influenced entropy value with {guard['condition']} before writing a persistent outcome; the timestamp boundary and block-derived entropy can jointly influence the result.",
            function_offset + header_end + 1 + body.find(guard["condition"]),
            related_functions=[function.name],
            symbols=["block.timestamp", "blockhash", "block.coinbase", "block.difficulty"],
            clock_domains=["local_block", "miner_entropy"],
            operation="timestamp_gate_to_miner_influenced_entropy",
        )
        findings[-1]["source_anchor_lines"] = [guard_line, state_line]
        break
    return findings


def _signed_lifecycle_bound_invariants(source: str) -> list[dict]:
    """Detect the narrow signed-receipt shape covered by the IR regression."""

    findings: list[dict] = []
    if not re.search(r"\b(?:createQuest|quest|receipt)\b", source, re.IGNORECASE):
        return findings
    if not re.search(r"\b(?:endTime|deadline|expiry|expiration)\b", source, re.IGNORECASE):
        return findings
    regions = _function_regions_with_offsets(source)
    for function, function_offset in regions:
        header_end = function.text.find("{")
        header = function.text[:header_end] if header_end >= 0 else function.text
        body = _strip_comments_preserve_layout(function.text[header_end + 1:])
        if not re.search(r"\b(?:signature|sig)\b", header, re.IGNORECASE):
            continue
        if not re.search(r"\b(?:recoverSigner|ecrecover|SignatureChecker|ECDSA)\b", body, re.IGNORECASE):
            continue
        if not re.search(r"\b(?:mint|receipt|claim|fulfill|redeem)\w*\b", function.name, re.IGNORECASE):
            continue
        if re.search(r"\b(?:block\.timestamp|now)\b", body, re.IGNORECASE) and re.search(
            r"\b(?:endTime|deadline|expiry|expiration)\b", body, re.IGNORECASE
        ):
            continue
        marker = re.search(r"\b(?:recoverSigner|ecrecover|SignatureChecker|ECDSA)\b", body, re.IGNORECASE)
        if marker is None:
            continue
        _add_evidence(
            findings,
            source,
            [item[0] for item in regions],
            "signed_receipt_missing_quest_expiry",
            0.86,
            f"{function.name}() verifies a signed receipt and records a claim without binding the signed lifecycle to the quest endTime/expiry boundary.",
            function_offset + header_end + 1 + marker.start(),
            related_functions=[function.name],
            symbols=["signature", "endTime", "block.timestamp"],
            clock_domains=["signed_authorization", "quest_lifecycle"],
            operation="signed_receipt_missing_lifecycle_bound",
        )
        break
    return findings


# ── Phase 2 detection rules ──

_ORDER_TIME_FIELD_RE = re.compile(
    r"\b_?(?:expiration|expiry|deadline|expiresAt|validUntil)\b",
    re.IGNORECASE,
)
_ORDER_TIMESTAMP_RE = re.compile(r"\b(?:now|block\.timestamp)\b", re.IGNORECASE)
_ORDER_STORAGE_RECORD_RE = re.compile(
    r"\b(?:storage|memory)\s+[A-Za-z_]\w*\s*=\s*[A-Za-z_]\w*\s*\[[^\]]+\]",
    re.IGNORECASE,
)
_ORDER_STATE_WRITE_RE = re.compile(
    r"\b[A-Za-z_]\w*(?:\s*\[[^\]]+\])?(?:\.[A-Za-z_]\w*)?\s*"
    r"(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)",
    re.IGNORECASE,
)
_ORDER_ASSET_OPERATION_RE = re.compile(
    r"\b(?:safeTransferFrom|safeTransfer|transferFrom|transfer|"
    r"_swap|swap|_cancelOrder|_addOpenOrder|_removeOpenOrder|_transferFees)\s*\(",
    re.IGNORECASE,
)


def _function_regions_with_offsets(source: str) -> list[tuple[FunctionRegion, int]]:
    """Return parsed function regions with source offsets for exact anchors."""

    result: list[tuple[FunctionRegion, int]] = []
    search_start = 0
    for function in extract_function_regions_with_modifiers(source):
        offset = source.find(function.text, search_start)
        if offset < 0:
            continue
        result.append((function, offset))
        search_start = offset + len(function.text)
    return result


def _timestamp_guard_matches(function_text: str) -> list[tuple[re.Match[str], str]]:
    """Find guards that compare the current block time with an expiry field."""

    matches: list[tuple[re.Match[str], str]] = []
    for marker in re.finditer(
        r"\b(?:if|require|assert)\s*\(", function_text, re.IGNORECASE
    ):
        opening = function_text.find("(", marker.start(), marker.end())
        closing = _matching_delimiter(function_text, opening, "(", ")")
        if closing < 0:
            continue
        condition = function_text[opening + 1:closing]
        if not _ORDER_TIMESTAMP_RE.search(condition):
            continue
        if not _ORDER_TIME_FIELD_RE.search(condition):
            continue
        if not re.search(r"(?:<=|>=|==|!=|<|>)", condition):
            continue
        matches.append((marker, condition))
    return matches


def _external_entrypoint_names(
    target: FunctionRegion,
    target_offset: int,
    regions: list[tuple[FunctionRegion, int]],
) -> list[str]:
    """Find public/external wrappers that invoke an internal helper."""

    header_end = target.text.find("{")
    target_header = target.text if header_end < 0 else target.text[:header_end]
    if re.search(r"\b(?:public|external)\b", target_header, re.IGNORECASE):
        return [target.name]

    callers: list[str] = []
    for candidate, candidate_offset in regions:
        if candidate_offset == target_offset or candidate.name == target.name:
            continue
        candidate_header_end = candidate.text.find("{")
        candidate_header = (
            candidate.text
            if candidate_header_end < 0
            else candidate.text[:candidate_header_end]
        )
        if not re.search(r"\b(?:public|external)\b", candidate_header, re.IGNORECASE):
            continue
        body = candidate.text[candidate_header_end + 1:]
        if not re.search(rf"\b{re.escape(target.name)}\s*\(", body):
            continue
        if candidate.name not in callers:
            callers.append(candidate.name)
    return callers


def _event_timestamp_offsets(function_text: str) -> list[int]:
    """Return offsets of block timestamps emitted on an event path."""

    offsets: list[int] = []
    for marker in re.finditer(
        r"\bemit\s+[A-Za-z_]\w*\s*\(", function_text, re.IGNORECASE
    ):
        opening = function_text.find("(", marker.start(), marker.end())
        closing = _matching_delimiter(function_text, opening, "(", ")")
        if closing < 0:
            continue
        for timestamp in _ORDER_TIMESTAMP_RE.finditer(
            function_text[opening + 1:closing]
        ):
            offsets.append(opening + 1 + timestamp.start())
    return offsets


def _order_expiration_lifecycle_invariants(source: str) -> list[dict]:
    """Detect typed order-expiration loci without promoting plain deadlines."""

    findings: list[dict] = []
    regions = _function_regions_with_offsets(source)

    def add_evidence(
        *,
        function: FunctionRegion,
        function_offset: int,
        entrypoints: list[str],
        subtype: str,
        confidence: float,
        reason: str,
        relative_offset: int,
        operation: str,
    ) -> None:
        _add_evidence(
            findings,
            source,
            [region for region, _ in regions],
            subtype,
            confidence,
            reason,
            function_offset + relative_offset,
            related_functions=[function.name, *entrypoints],
            symbols=[function.name, "block.timestamp", "expiration"],
            clock_domains=["order_lifecycle", "local_block"],
            operation=operation,
        )
        findings[-1]["order_temporal_family"] = "order_expiration_lifecycle"
        findings[-1]["source_evidence_kind"] = "order_expiration_lifecycle"
        findings[-1]["entrypoint_function_name"] = (
            entrypoints[0] if entrypoints else function.name
        )
        findings[-1]["source_anchor_lines"] = [findings[-1]["line"]]

    for function, function_offset in regions:
        header_end = function.text.find("{")
        if header_end < 0:
            continue
        header = function.text[:header_end]
        body = function.text[header_end + 1:]
        if not _ORDER_TIME_FIELD_RE.search(function.text):
            continue
        if not _ORDER_STORAGE_RECORD_RE.search(function.text):
            continue
        if not re.search(r"\b(?:order|orders)\b", function.text, re.IGNORECASE):
            continue

        entrypoints = _external_entrypoint_names(function, function_offset, regions)
        if not entrypoints:
            continue
        guards = _timestamp_guard_matches(function.text)
        if not guards:
            continue

        is_view = bool(re.search(r"\b(?:view|pure)\b", header, re.IGNORECASE))
        has_validation_path = bool(
            re.search(r"\bbalanceOf\s*\(", body, re.IGNORECASE)
            and re.search(r"\ballowance\s*\(", body, re.IGNORECASE)
            and re.search(r"\b(?:status|amountOutMin)\b", body, re.IGNORECASE)
        )
        has_asset_operation = bool(_ORDER_ASSET_OPERATION_RE.search(body))
        has_state_write = bool(_ORDER_STATE_WRITE_RE.search(body))
        name_lower = function.name.casefold()
        is_validation_helper = is_view and has_validation_path
        is_cancellation = "cancel" in name_lower and bool(
            re.search(r"\b_cancelOrder\s*\(", body, re.IGNORECASE)
        )
        is_execution = (
            not is_view
            and has_asset_operation
            and has_state_write
            and "execute" in name_lower
        )
        has_expiration_parameter = bool(_ORDER_TIME_FIELD_RE.search(header))
        is_creation = (
            not is_view
            and has_expiration_parameter
            and has_state_write
            and bool(re.search(r"\b(?:place|create|submit)\w*\b", name_lower))
        )

        for marker, condition in guards:
            condition_timestamp = _ORDER_TIMESTAMP_RE.search(condition)
            if condition_timestamp is None:
                continue
            condition_offset = function.text.find("(", marker.start(), marker.end()) + 1
            condition_offset += condition_timestamp.start()

            if is_validation_helper:
                add_evidence(
                    function=function,
                    function_offset=function_offset,
                    entrypoints=entrypoints,
                    subtype="order_expiration_validation_gate",
                    confidence=0.88,
                    reason=(
                        f"{function.name}() validates an order expiration against the current block timestamp "
                        "before balance, allowance, status, and output checks; the permissionless order "
                        "validation path is time-dependent."
                    ),
                    relative_offset=condition_offset,
                    operation="order_expiration_validation_gate",
                )
            elif is_cancellation:
                add_evidence(
                    function=function,
                    function_offset=function_offset,
                    entrypoints=entrypoints,
                    subtype="order_expiration_cancellation_gate",
                    confidence=0.88,
                    reason=(
                        f"{function.name}() compares an order expiration with block.timestamp before "
                        "cancelling the expired order and releasing its order-state path."
                    ),
                    relative_offset=condition_offset,
                    operation="order_expiration_cancellation_gate",
                )
            elif is_execution:
                add_evidence(
                    function=function,
                    function_offset=function_offset,
                    entrypoints=entrypoints,
                    subtype="order_expiration_execution_gate",
                    confidence=0.88,
                    reason=(
                        f"{function.name}() gates a permissionless order execution on block.timestamp <= "
                        "order.expiration before persistent status changes and token/swap operations."
                    ),
                    relative_offset=condition_offset,
                    operation="order_expiration_execution_gate",
                )
            elif is_creation:
                add_evidence(
                    function=function,
                    function_offset=function_offset,
                    entrypoints=entrypoints,
                    subtype="order_expiration_creation_gate",
                    confidence=0.88,
                    reason=(
                        f"{function.name}() accepts a caller-supplied order expiration and compares it with "
                        "block.timestamp before recording the order and locking its asset balance."
                    ),
                    relative_offset=condition_offset,
                    operation="order_expiration_creation_gate",
                )

        if is_creation:
            for assignment in re.finditer(
                r"[^;\n]*\b(?:createdAt|created|openedAt|startTime)\b\s*=\s*"
                r"(?:block\.timestamp|now)\b",
                function.text,
                re.IGNORECASE,
            ):
                timestamp = _ORDER_TIMESTAMP_RE.search(assignment.group(0))
                if timestamp is None:
                    continue
                add_evidence(
                    function=function,
                    function_offset=function_offset,
                    entrypoints=entrypoints,
                    subtype="order_creation_timestamp_assignment",
                    confidence=0.86,
                    reason=(
                        f"{function.name}() records the order creation time from block.timestamp in the same "
                        "lifecycle that accepts and stores an expiration bound."
                    ),
                    relative_offset=assignment.start() + timestamp.start(),
                    operation="order_creation_timestamp_assignment",
                )

        if is_execution:
            for event_offset in _event_timestamp_offsets(function.text):
                add_evidence(
                    function=function,
                    function_offset=function_offset,
                    entrypoints=entrypoints,
                    subtype="order_expiration_execution_timestamp_record",
                    confidence=0.86,
                    reason=(
                        f"{function.name}() records block.timestamp in the execution event after the same "
                        "expiration-gated order and asset transition; this is the terminal anchor of the "
                        "order temporal dependency."
                    ),
                    relative_offset=event_offset,
                    operation="order_expiration_execution_timestamp_record",
                )

    return findings


def _timestamp_provider_vesting_asset_invariants(source: str) -> list[dict]:
    """Link a timestamp provider to a vesting calculation and asset claim.

    Some DApp contracts isolate ``block.timestamp`` behind a public view
    helper.  A lexical timestamp scan then sees only a benign provider while
    the real security-relevant path is provider -> vesting balance helper ->
    stateful claim/transfer.  This rule admits that path only when all three
    pieces are source-grounded and the final public caller performs an asset
    operation; a standalone timestamp getter remains a negative control.
    """

    findings: list[dict] = []
    regions = _function_regions_with_offsets(source)
    if not regions:
        return findings
    functions = [function for function, _offset in regions]
    function_by_name = {function.name: function for function in functions if function.name}
    if not function_by_name:
        return findings

    def body_parts(function: FunctionRegion) -> tuple[str, str, int]:
        header_end = function.text.find("{")
        if header_end < 0:
            return function.text, "", 0
        return (
            function.text[:header_end],
            _strip_comments_preserve_layout(function.text[header_end + 1:]),
            header_end,
        )

    providers: list[dict[str, object]] = []
    for function, offset in regions:
        header, body, header_end = body_parts(function)
        provider_match = re.search(
            r"\breturn\s+(?:block\.timestamp|now)\s*;", body, re.IGNORECASE
        )
        if provider_match is None:
            continue
        providers.append({
            "function": function,
            "offset": offset,
            "header_end": header_end,
            "body": body,
            "match": provider_match,
            "line": function.start_line
            + function.text[:header_end].count("\n")
            + body[:provider_match.start()].count("\n"),
        })

    if not providers:
        return findings

    function_calls: dict[str, set[str]] = {}
    for function in functions:
        _header, body, _header_end = body_parts(function)
        function_calls[function.name] = {
            name
            for name in re.findall(r"\b([A-Za-z_]\w*)\s*\(", body)
            if name in function_by_name and name != function.name
        }

    asset_call_re = re.compile(
        r"\b(?:safeTransferFrom|safeTransfer|transferFrom|transfer|"
        r"sendValue|mint|burn|withdraw|redeem|claim|release|unlock|"
        r"deposit|stake|settle|payout|swap)\s*\(",
        re.IGNORECASE,
    )
    state_write_re = re.compile(
        r"\b[A-Za-z_]\w*(?:\s*\[[^\]]+\])?(?:\.[A-Za-z_]\w*)?\s*"
        r"(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)"
    )
    visibility_re = re.compile(r"\b(?:public|external)\b", re.IGNORECASE)

    for provider in providers:
        provider_function = provider["function"]
        provider_name = provider_function.name
        temporal_consumers: list[dict[str, object]] = []
        for function, offset in regions:
            if function.name == provider_name:
                continue
            header, body, header_end = body_parts(function)
            call_match = re.search(rf"\b{re.escape(provider_name)}\s*\(", body)
            if call_match is None:
                continue
            # Require a typed vesting/time calculation rather than a generic
            # timestamp read: a provider result must be compared against
            # lifecycle fields and participate in a value/state computation.
            temporal_fields = bool(re.search(
                r"\b(?:start|end|cliff|unlock|vest(?:ed|ing)?|lastUpdated|"
                r"claimed|duration|amount)\b",
                body,
                re.IGNORECASE,
            ))
            timestamp_alias = bool(re.search(
                rf"\b[A-Za-z_]\w*\s*=\s*{re.escape(provider_name)}\s*\(",
                body,
            ))
            comparison = bool(re.search(r"(?:<=|>=|==|!=|<|>)", body))
            value_flow = bool(re.search(
                r"\b(?:amount|balance|vested|claimed|locked|unlocked|"
                r"currentTimestamp|timestamp)\b",
                body,
                re.IGNORECASE,
            ))
            if not (temporal_fields and comparison and timestamp_alias and value_flow):
                continue
            temporal_lines = [
                function.start_line
                + function.text[:header_end].count("\n")
                + body[:match.start()].count("\n")
                for match in re.finditer(
                    r"(?:<=|>=|==|!=|<|>)|\b(?:start|end|cliff|lastUpdated|"
                    r"amount|claimed|vested)\b",
                    body,
                    re.IGNORECASE,
                )
            ]
            temporal_consumers.append({
                "function": function,
                "offset": offset,
                "call_line": function.start_line
                + function.text[:header_end].count("\n")
                + body[:call_match.start()].count("\n"),
                "temporal_lines": temporal_lines,
            })

        if not temporal_consumers:
            continue

        consumer_names = {item["function"].name for item in temporal_consumers}
        reachable = set(consumer_names)
        changed = True
        while changed:
            changed = False
            for function_name, callees in function_calls.items():
                if function_name in reachable or not (callees & reachable):
                    continue
                reachable.add(function_name)
                changed = True

        asset_entries: list[dict[str, object]] = []
        for function, offset in regions:
            if function.name not in reachable:
                continue
            header, body, header_end = body_parts(function)
            if not visibility_re.search(header) or re.search(
                r"\b(?:view|pure)\b", header, re.IGNORECASE
            ):
                continue
            asset_matches = list(asset_call_re.finditer(body))
            state_matches = list(state_write_re.finditer(body))
            if not asset_matches and not state_matches:
                continue
            asset_entries.append({
                "function": function,
                "offset": offset,
                "lines": [
                    function.start_line
                    + function.text[:header_end].count("\n")
                    + body[:match.start()].count("\n")
                    for match in asset_matches
                ],
            })

        if not asset_entries:
            continue

        consumer_functions = [item["function"].name for item in temporal_consumers]
        entry_functions = [item["function"].name for item in asset_entries]
        provider_line = int(provider["line"])
        anchor_lines = {provider_line}
        symbols = [provider_name, "block.timestamp"]
        for consumer in temporal_consumers:
            anchor_lines.add(int(consumer["call_line"]))
            anchor_lines.update(
                int(line) for line in consumer["temporal_lines"] if int(line) > 0
            )
            symbols.append(str(consumer["function"].name))
        for entry in asset_entries:
            anchor_lines.update(int(line) for line in entry["lines"] if int(line) > 0)
            symbols.append(str(entry["function"].name))

        provider_offset = int(provider["offset"]) + int(provider["header_end"]) + 1 + provider["match"].start()
        _add_evidence(
            findings,
            source,
            functions,
            "timestamp_provider_to_vesting_asset_consumer",
            0.88,
            (
                f"{provider_name}() exposes block.timestamp; its result feeds "
                f"{', '.join(consumer_functions)}() vesting calculations and reaches "
                f"the public asset path {', '.join(entry_functions)}(), so timestamp "
                "controls the claimable asset amount across the lifecycle."
            ),
            provider_offset,
            related_functions=[provider_name, *consumer_functions, *entry_functions],
            symbols=sorted(set(symbols)),
            clock_domains=["local_block", "vesting_lifecycle", "asset_settlement"],
            operation="timestamp_provider_to_vesting_asset_claim",
        )
        findings[-1]["source_anchor_lines"] = sorted(anchor_lines)
        findings[-1]["temporal_locus_scoped"] = True
        break

    return findings


def analyze_temporal_invariants(source: str) -> list[dict]:
    """Phase 2: detection rules for 6 FN gaps from v20 blind run."""
    findings: list[dict] = []
    functions = extract_function_regions(source)
    code = _strip_comments_preserve_layout(source)

    findings.extend(_order_expiration_lifecycle_invariants(source))
    findings.extend(_modifier_time_window_invariants(source))
    findings.extend(_state_duration_gate_invariants(source))
    findings.extend(_modifier_timestamp_gate_asset_invariants(source))
    findings.extend(_lifecycle_boundary_timestamp_invariants(source))
    findings.extend(_timestamp_gated_entropy_invariants(source))
    findings.extend(_signed_lifecycle_bound_invariants(source))
    findings.extend(_timestamp_provider_vesting_asset_invariants(source))

    # -- A1: G1b Simplified EIP712 / SignatureChecker-only (P01) --
    if not any(f.get("subtype") == "eip712_signed_authorization_without_deadline" for f in findings):
        has_sigchecker_only = bool(
            re.search(r"\bSignatureChecker\s*\.\s*isValidSignatureNow\s*\(", code)
            and not re.search(r"\b_hashTypedDataV4\b", code)
            and not re.search(r"\b(?:ECDSA\.recover|ecrecover)\s*\(", code)
        )
        if has_sigchecker_only:
            typehash_defs = list(re.finditer(
                r'(?P<name>[A-Z_][A-Z_0-9]*)\s*=\s*'
                r'keccak256\s*\(\s*\"(?P<params>[^\"]*)\"\s*\)',
                code,
            ))
            for th_def in typehash_defs:
                th_name = th_def.group("name")
                th_params = th_def.group("params")
                if re.search(r"\b(deadline|expiry|expiration|validUntil|expiresAt)\b",
                             th_params, re.IGNORECASE):
                    continue
                if re.search(r"\bnonce\b", th_params, re.IGNORECASE):
                    continue
                uses_typehash = re.search(
                    r"keccak256\s*\(\s*abi\.encode\s*\(\s*"
                    + re.escape(th_name) + r"\b",
                    code,
                )
                if not uses_typehash:
                    continue
                has_sig_verify = re.search(
                    r"SignatureChecker\s*\.\s*isValidSignatureNow\s*\(",
                    code,
                )
                if not has_sig_verify:
                    continue
                has_lower_bound = re.search(
                    r"require\s*\(\s*block\.timestamp\s*>=\s*", code,
                )
                has_executed_state = re.search(
                    r"\w+\[\w+\]\s*=\s*true\s*;", code,
                )
                if not (has_lower_bound and has_executed_state):
                    continue
                _add_evidence(
                    findings, source, functions,
                    "eip712_signed_authorization_without_deadline",
                    0.86,
                    "A SignatureChecker-verified EIP712 typed-data authorization excludes deadline/expiry from the signed hash and the execution path enforces only a lower-bound timelock with executed-state recording, allowing stale signatures indefinite reuse.",
                    uses_typehash.start(),
                    symbols=[th_name, "deadline", "block.timestamp"],
                    clock_domains=["signed_authorization", "lifecycle_time"],
                    operation="missing_signed_authorization_lifetime",
                )
                break

    # -- A2: G1c Signed deadline unenforced (P07) --
    if not any(f.get("subtype") == "signed_permit_deadline_unenforced" for f in findings):
        has_ecdsa_with_deadline = bool(
            re.search(r"\b(?:ECDSA\.recover|ecrecover)\s*\(", code)
            and re.search(r"\bdeadline\b", code, re.IGNORECASE)
        )
        if has_ecdsa_with_deadline:
            encode_calls = list(re.finditer(
                r"keccak256\s*\(\s*abi\.encode(Packed)?\s*\("
                r"([^;]{0,600})"
                r"\)",
                code, re.DOTALL | re.IGNORECASE,
            ))
            for enc in encode_calls:
                enc_params = enc.group(2)
                if not re.search(r"\bdeadline\b", enc_params, re.IGNORECASE):
                    continue
                after_enc = code[enc.start():enc.start() + 1200]
                ecdsa_verify = re.search(
                    r"\b(?:ECDSA\.recover|ecrecover)\s*\(",
                    after_enc, re.IGNORECASE,
                )
                if not ecdsa_verify:
                    continue
                has_deadline_guard = re.search(
                    r"(?:require|if)\s*\(\s*block\.timestamp\s*(?:<=|<)\s*"
                    r"(?:\w+\.)?deadline\b",
                    after_enc, re.IGNORECASE,
                ) or re.search(
                    r"(?:require|if)\s*\(\s*block\.timestamp\s*(?:<=|<)\s*"
                    r"(?:\w+\.)?deadline\b",
                    code[:enc.start()], re.IGNORECASE,
                )
                if has_deadline_guard:
                    continue
                source_offset = source.find(enc.group(0))
                if source_offset < 0:
                    source_offset = enc.start()
                _add_evidence(
                    findings, source, functions,
                    "signed_permit_deadline_unenforced",
                    0.87,
                    "A signature verification path includes a deadline field in the signed hash but never enforces require(block.timestamp <= deadline), allowing stale signed messages to remain valid indefinitely.",
                    source_offset,
                    symbols=["deadline", "ecrecover", "block.timestamp"],
                    clock_domains=["signed_authorization"],
                    operation="deadline_not_enforced",
                )
                break

    # -- B1: G2b Merkle proof bridge replay (P02) --
    if not any(f.get("subtype") in ("cross_chain_message_signature_without_expiry",
                                     "merkle_bridge_leaf_without_freshness")
               for f in findings):
        has_merkle_bridge = bool(
            re.search(r"\bMerkleProof\s*\.\s*verify\s*\(", code)
            and not re.search(
                r"\b(?:cross[_-]chain|from_chain|to_chain|receive_cross_chain)\b",
                source, re.IGNORECASE,
            )
        )
        if has_merkle_bridge:
            for function in functions:
                function_code = _strip_comments_preserve_layout(function.text)
                leaf_hash = re.search(
                    r"keccak256\s*\(\s*abi\.encodePacked\s*\(([^;]{0,400})\)",
                    function_code,
                    re.IGNORECASE | re.DOTALL,
                )
                if not leaf_hash:
                    continue
                leaf_fields = leaf_hash.group(1)
                has_freshness = re.search(
                    r"\b(nonce|timestamp|sequence|epoch|expir\w*|deadline|srcTimestamp)\b",
                    leaf_fields, re.IGNORECASE,
                )
                if has_freshness:
                    continue
                first_use_guard = re.search(
                    r"require\s*\(\s*!\s*(?P<mapping>\w+)\s*\[[^\]]*\]",
                    function_code, re.IGNORECASE,
                ) or re.search(
                    r"require\s*\(\s*(?P<mapping2>\w+)\s*\[[^\]]*\]\s*==\s*(?:0|false)",
                    function_code, re.IGNORECASE,
                )
                if not first_use_guard:
                    continue
                mapping_name = first_use_guard.group("mapping") or first_use_guard.group("mapping2")
                has_state_record = re.search(
                    rf"\b{re.escape(mapping_name)}\s*\[[^\]]*\](?:\.\w+)?\s*=\s*\w",
                    function_code,
                )
                if not has_state_record:
                    continue
                source_offset = source.find(function.text) + leaf_hash.start()
                _add_evidence(
                    findings, source, functions,
                    "merkle_bridge_leaf_without_freshness",
                    0.85,
                    "A Merkle proof bridge computes a leaf hash without nonce, timestamp, or sequence fields, and records the result in a first-use mapping, allowing stale proofs to be replayed indefinitely.",
                    source_offset,
                    related_functions=[function.name],
                    symbols=["merkleProof", "leaf", mapping_name],
                    clock_domains=["cross_chain", "merkle_proof"],
                    operation="missing_merkle_leaf_lifetime",
                )
                break

    # -- C1: Local temporal delta division without zero-guard (P05) --
    if not any(f.get("subtype") == "local_temporal_delta_division_unguarded" for f in findings):
        for function in functions:
            function_code = _strip_comments_preserve_layout(function.text)
            delta_compute = re.search(
                r"(?:timeDelta|timeElapsed)\s*=\s*block\.timestamp\s*-\s*\w+"
                r"|\w+\[\d+\]\.timestamp\s*-\s*\w+\[\d+\]\.timestamp",
                function_code, re.IGNORECASE,
            )
            if not delta_compute:
                continue
            has_division_use = bool(re.search(
                r"(?:\btimeDelta\b|\btimeElapsed\b)\s*/|/"
                r"\s*(?:\btimeDelta\b|\btimeElapsed\b)",
                function_code,
                re.IGNORECASE,
            ))
            if not has_division_use:
                continue
            has_zero_guard = re.search(
                r"require\s*\(\s*\w*(?:timeDelta|timeElapsed)\s*>\s*0",
                function_code, re.IGNORECASE,
            )
            if has_zero_guard:
                continue
            source_offset = source.find(function.text) + delta_compute.start()
            _add_evidence(
                findings, source, functions,
                "local_temporal_delta_division_unguarded",
                0.88,
                "A temporal delta (block.timestamp or timestamp difference) is used as an arithmetic operand without a require(delta > 0) guard, risking division-by-zero or underflow.",
                source_offset,
                related_functions=[function.name],
                symbols=["block.timestamp", "timeDelta"],
                clock_domains=["local_block", "temporal_arithmetic"],
                operation="unguarded_timestamp_delta",
            )
            break

    # -- C2: Oracle staleness-by-omission (P06) --
    if not any(f.get("subtype") == "oracle_staleness_by_omission" for f in findings):
        for function in functions:
            function_code = _strip_comments_preserve_layout(function.text)
            oracle_price_call = re.search(
                r"\b(?:\w*(?:oracle|feed|aggregator)\w*)\s*\.\s*"
                r"(?:getPrice|latestAnswer|getAssetPrice)\s*\(",
                function_code, re.IGNORECASE,
            )
            if not oracle_price_call:
                continue
            is_economic = re.search(
                r"\b(?:liquidate|borrow|mint|withdraw|redeem|swap|seize)(?:[A-Z]\w*)?\s*\(",
                function_code, re.IGNORECASE,
            )
            if not is_economic:
                continue
            has_freshness_guard = re.search(
                r"(?:lastUpdate|updatedAt|publishTime|lastUpdateTime|staleness)"
                r".{0,80}?(?:block\.timestamp|HEARTBEAT|maxAge|heartbeat)",
                function_code, re.IGNORECASE | re.DOTALL,
            )
            if has_freshness_guard:
                continue
            source_offset = source.find(function.text) + oracle_price_call.start()
            _add_evidence(
                findings, source, functions,
                "oracle_staleness_by_omission",
                0.90,
                "An oracle price is read in an economically-consequential function without any staleness, heartbeat, or last-update freshness check, allowing stale prices to drive financial decisions.",
                source_offset,
                related_functions=[function.name],
                symbols=["oracle", "lastUpdate", "block.timestamp"],
                clock_domains=["external_oracle", "local_block"],
                operation="missing_oracle_staleness_guard",
            )
            break

    # -- General temporal subtraction without future guard (P10) --
    if not any(f.get("subtype") == "temporal_subtraction_unguarded" for f in findings):
        for function in functions:
            function_code = _strip_comments_preserve_layout(function.text)
            subtraction = re.search(
                r"block\.timestamp\s*-\s*(\w+)",
                function_code,
            )
            if not subtraction:
                continue
            var_name = subtraction.group(1)
            if re.search(rf"\b(?:calldata|memory)\s+.*\b{re.escape(var_name)}", function_code):
                continue
            var_written_elsewhere = False
            arbitrary_future_write = False
            for other_func in functions:
                if other_func.name == function.name:
                    continue
                other_code = _strip_comments_preserve_layout(other_func.text)
                if re.search(rf"\b{re.escape(var_name)}\s*=\s*block\.timestamp", other_code):
                    var_written_elsewhere = True
                    continue
                if re.search(rf"\b{re.escape(var_name)}\s*=", other_code):
                    arbitrary_future_write = True
            if not var_written_elsewhere:
                continue
            if not arbitrary_future_write:
                continue
            has_future_guard = re.search(
                rf"require\s*\(\s*{re.escape(var_name)}\s*<=\s*block\.timestamp",
                function_code,
            ) or re.search(
                rf"require\s*\(\s*block\.timestamp\s*>=\s*{re.escape(var_name)}",
                function_code,
            )
            if has_future_guard:
                continue
            source_offset = source.find(function.text) + subtraction.start()
            _add_evidence(
                findings, source, functions,
                "temporal_subtraction_unguarded",
                0.87,
                f"block.timestamp - {var_name} lacks a require({var_name} <= block.timestamp) guard, and {var_name} is set in a different function, risking underflow on unsigned arithmetic.",
                source_offset,
                related_functions=[function.name],
                symbols=[var_name, "block.timestamp"],
                clock_domains=["local_block"],
                operation="unguarded_timestamp_subtraction",
            )
            break

    # -- D1: Exact timestamp equality controls an economic action --
    # This deliberately requires a caller-supplied input, exact equality, and
    # either a persistent caller assignment or a value transfer in the branch.
    # Ordinary deadline and timelock comparisons therefore do not qualify.
    search_start = 0
    for function in functions:
        function_offset = source.find(function.text, search_start)
        if function_offset < 0:
            continue
        search_start = function_offset + len(function.text)

        header_end = function.text.find("{")
        header = function.text if header_end < 0 else function.text[:header_end]
        if not re.search(r"\b(?:public|external)\b", header):
            continue
        if re.search(r"\b(?:view|pure)\b", header):
            continue

        parameters = _function_parameter_names(function.text)
        if not parameters:
            continue
        aliases = set(re.findall(
            r"\b([A-Za-z_]\w*)\s*=\s*(?:block\.timestamp|now)\b",
            function.text,
        ))
        time_terms = ["block.timestamp", "now", *aliases]

        for match in re.finditer(r"\bif\s*\(", function.text):
            condition_start = match.end() - 1
            condition_end = _matching_delimiter(
                function.text, condition_start, "(", ")"
            )
            if condition_end < 0:
                continue
            condition = function.text[condition_start + 1:condition_end]
            if "==" not in condition:
                continue

            has_time = any(
                term in condition if "." in term
                else re.search(rf"\b{re.escape(term)}\b", condition)
                for term in time_terms
            )
            has_parameter = any(
                re.search(rf"\b{re.escape(parameter)}\b", condition)
                for parameter in parameters
            )
            if not (has_time and has_parameter):
                continue

            branch_start = condition_end + 1
            while (
                branch_start < len(function.text)
                and function.text[branch_start].isspace()
            ):
                branch_start += 1
            if branch_start >= len(function.text) or function.text[branch_start] != "{":
                continue
            branch_end = _matching_delimiter(
                function.text, branch_start, "{", "}"
            )
            if branch_end < 0:
                continue
            branch = function.text[branch_start + 1:branch_end]

            persistent_assignment = False
            for assignment in re.finditer(
                r"\b([A-Za-z_]\w*)\s*=\s*msg\.sender\b", branch
            ):
                if _declared_before_function(
                    source[:function_offset], assignment.group(1)
                ):
                    persistent_assignment = True
                    break
            value_transfer = bool(re.search(
                r"\.(?:transfer|send)\s*\(|\.call(?:\.value\s*\(|\s*\{[^}]*\bvalue\s*:)",
                branch,
            ))
            if not (persistent_assignment or value_transfer):
                continue

            source_offset = function_offset + match.start()
            _add_evidence(
                findings, source, functions,
                "timestamp_exact_equality_controls_economic_action",
                0.91,
                "A caller-supplied time is compared for exact equality with the current block timestamp before assigning the caller as a persistent winner/beneficiary or transferring value, allowing a block producer to influence the economic outcome.",
                source_offset,
                related_functions=[function.name],
                symbols=sorted(parameters.intersection(
                    set(re.findall(r"[A-Za-z_]\w*", condition))
                )) + ["block.timestamp"],
                clock_domains=["local_block"],
                operation="exact_timestamp_equality_economic_action",
            )
            break

    return findings

# ── Source-grounded temporal IR and slice ──

_TEMPORAL_FIELD_RE = re.compile(
    r"\b(?:deadline|expiry|expiration|expiresAt|validUntil|validAfter|"
    r"unlockTime|endTime|startTime|createdAt|updatedAt|timestamp|"
    r"currentExpiration|last(?:Update|Event|Beat|InflationDecay)|"
    r"duration(?:Days|Seconds)?|cooldown|epoch(?:Timestamp)?|"
    r"observationTimestamp|publishTime|withdrawTime)\b",
    re.IGNORECASE,
)
_TEMPORAL_CALL_RE = re.compile(
    r"\b(?:block\.timestamp|now|timestamp|deadline|expiry|expiration|"
    r"expiresAt|validUntil|validAfter|unlockTime|endTime|startTime|"
    r"createdAt|updatedAt|currentExpiration|duration(?:Days|Seconds)?|"
    r"cooldown|epoch(?:Timestamp)?|withdrawTime)\b",
    re.IGNORECASE,
)
_IR_CONTROL_WORDS = {
    "if", "for", "while", "require", "assert", "revert", "return",
    "keccak256", "abi", "encode", "encodePacked", "address", "uint",
    "uint8", "uint16", "uint32", "uint64", "uint128", "uint256",
    "int", "int8", "int16", "int32", "int64", "int128", "int256",
    "bytes", "string", "bool", "mapping", "true", "false",
}
_IR_ASSET_CALL_RE = re.compile(
    r"\b(?:safeTransferFrom|safeTransfer|transferFrom|transfer|"
    r"sendValue|\.call|call|send|_?mint|_?burn|withdraw|redeem|"
    r"claim|release|unlock|deposit|stake|settle|settlement|payout|"
    r"swap|joinPool|exitPool)\s*(?:\.value\s*)?\(",
    re.IGNORECASE,
)
_IR_EXTERNAL_CALL_RE = re.compile(
    r"\b(?P<receiver>[A-Za-z_]\w*(?:\s*\[[^\]]+\])?)\s*\.\s*"
    r"(?P<method>[A-Za-z_]\w*)\s*\(",
)
_IR_ASSIGNMENT_RE = re.compile(
    r"(?P<lhs>[A-Za-z_]\w*(?:\s*\[[^\]]+\])?(?:\s*\.\s*[A-Za-z_]\w*)*)"
    r"\s*(?P<op>\+=|-=|\*=|/=|\+\+|--|=(?!=))\s*(?P<rhs>[^;\n{}]*)",
)
_IR_DECL_RE = re.compile(
    r"\b(?P<type>u?int(?P<bits>\d+)?|bytes(?P<byte_bits>\d+)?|"
    r"address(?:\s+payable)?|bool|string)\s+"
    r"(?P<mods>(?:(?:public|private|internal|external|constant|immutable|"
    r"memory|storage|calldata)\s+)*)"
    r"(?P<name>[A-Za-z_]\w*)\b",
    re.IGNORECASE,
)


def _ir_line(text: str, offset: int) -> int:
    return text[:max(offset, 0)].count("\n") + 1


def _ir_type_width(type_name: str, bits: str | None, byte_bits: str | None) -> int | None:
    if bits:
        return int(bits)
    if byte_bits:
        return int(byte_bits) * 8
    if type_name.lower() in {"uint", "int"}:
        return 256
    return None


def _ir_parameter_specs(header: str) -> list[dict]:
    opening = header.find("(")
    if opening < 0:
        return []
    closing = _matching_delimiter(header, opening, "(", ")")
    if closing < 0:
        return []
    specs: list[dict] = []
    for declaration in header[opening + 1:closing].split(","):
        declaration = declaration.strip()
        if not declaration:
            continue
        match = re.search(
            r"(?P<type>u?int(?P<bits>\d+)?|bytes(?P<byte_bits>\d+)?|"
            r"address(?:\s+payable)?|bool|string|[A-Za-z_]\w*)"
            r"(?:\s+(?:memory|storage|calldata|payable))?\s+"
            r"(?P<name>[A-Za-z_]\w*)\s*$",
            declaration,
            re.IGNORECASE,
        )
        if not match:
            continue
        specs.append({
            "name": match.group("name"),
            "type": match.group("type"),
            "bit_width": _ir_type_width(
                match.group("type"), match.group("bits"), match.group("byte_bits")
            ),
            "declaration": declaration,
        })
    return specs


def _ir_guard_records(function: FunctionRegion) -> list[dict]:
    records: list[dict] = []
    text = _strip_comments_preserve_layout(function.text)
    for marker in re.finditer(r"\b(?:require|assert|if|while)\s*\(", text, re.IGNORECASE):
        opening = text.find("(", marker.start(), marker.end())
        closing = _matching_delimiter(text, opening, "(", ")")
        if closing < 0:
            continue
        condition = text[opening + 1:closing].strip()
        fields = sorted(set(_TEMPORAL_FIELD_RE.findall(condition)), key=str.lower)
        timestamp_terms = sorted(set(_TEMPORAL_CALL_RE.findall(condition)), key=str.lower)
        if not fields and not timestamp_terms:
            continue
        comparators = re.findall(r"<=|>=|==|!=|<|>|\+|-", condition)
        records.append({
            "kind": marker.group(0).split("(", 1)[0].strip().lower(),
            "condition": condition,
            "comparators": comparators,
            "fields": fields,
            "timestamp_terms": timestamp_terms,
            "function_name": function.name,
            "line": function.start_line + text[:marker.start()].count("\n"),
            "evidence_lines": [
                function.start_line + text[:marker.start()].count("\n"),
            ],
        })
    return records


def _ir_function_calls(function: FunctionRegion) -> list[dict]:
    text = _strip_comments_preserve_layout(function.text)
    calls: list[dict] = []
    for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", text):
        name = match.group(1)
        if name in _IR_CONTROL_WORDS or name == function.name:
            continue
        calls.append({
            "name": name,
            "function_name": function.name,
            "line": function.start_line + text[:match.start()].count("\n"),
            "kind": "direct",
        })
    return calls


def _ir_symbol_declarations(source: str, regions: list[tuple[FunctionRegion, int]]) -> list[dict]:
    symbols: list[dict] = []
    for match in _IR_DECL_RE.finditer(_strip_comments_preserve_layout(source)):
        name = match.group("name")
        if name in _IR_CONTROL_WORDS:
            continue
        line = _ir_line(source, match.start())
        before = source[:match.start()]
        in_function = None
        for function, offset in regions:
            if offset <= match.start() <= offset + len(function.text):
                in_function = function
                break
        type_name = match.group("type")
        symbols.append({
            "name": name,
            "type": type_name,
            "bit_width": _ir_type_width(
                type_name, match.group("bits"), match.group("byte_bits")
            ),
            "kind": "local" if in_function else "state_or_struct",
            "storage": "local" if in_function else "storage",
            "function_name": in_function.name if in_function else "",
            "line": line,
            "temporal": bool(_TEMPORAL_FIELD_RE.search(name)),
        })
    return symbols


def _ir_struct_fields(source: str) -> dict[str, set[str]]:
    """Return temporal field names declared by each Solidity struct."""
    result: dict[str, set[str]] = {}
    for match in re.finditer(
        r"\bstruct\s+(?P<name>[A-Za-z_]\w*)\s*\{(?P<body>.*?)\}",
        _strip_comments_preserve_layout(source),
        re.IGNORECASE | re.DOTALL,
    ):
        fields = {
            item.group("name")
            for item in _IR_DECL_RE.finditer(match.group("body"))
            if _TEMPORAL_FIELD_RE.fullmatch(item.group("name"))
        }
        if fields:
            result[match.group("name")] = fields
    return result


def _ir_field_flow_records(
    source: str,
    regions: list[tuple[FunctionRegion, int]],
    functions: list[dict],
) -> list[dict]:
    flows: list[dict] = []
    struct_fields = _ir_struct_fields(source)
    for function, offset in regions:
        header_end = function.text.find("{")
        header = function.text[:header_end] if header_end >= 0 else function.text
        params = _ir_parameter_specs(header)
        body = _strip_comments_preserve_layout(function.text[header_end + 1:]) if header_end >= 0 else ""
        for param in params:
            param_name = param["name"]
            nested_uses = list(re.finditer(rf"\b{re.escape(param_name)}\s*\.\s*([A-Za-z_]\w*)", body))
            field_names = {m.group(1) for m in nested_uses}
            field_names.update(struct_fields.get(param.get("type", ""), set()))
            if _TEMPORAL_FIELD_RE.fullmatch(param_name) and re.search(
                rf"\b{re.escape(param_name)}\b", body
            ):
                field_names.add(param_name)
            for field in sorted(field_names, key=str.lower):
                if not _TEMPORAL_FIELD_RE.fullmatch(field):
                    continue
                field_expr = (
                    f"{param_name}.{field}"
                    if field != param_name
                    else param_name
                )
                guarded = any(
                    re.search(rf"\b{re.escape(field)}\b", guard["condition"], re.IGNORECASE)
                    or re.search(rf"\b{re.escape(field_expr)}\b", guard["condition"], re.IGNORECASE)
                    for guard in next(
                        (item["guards"] for item in functions if item["name"] == function.name), []
                    )
                )
                call_uses_timestamp = bool(re.search(r"\b(?:block\.timestamp|now)\b", body))
                action = "accepted"
                direct_use = bool(
                    re.search(
                        rf"\b{re.escape(param_name)}\s*\.\s*{re.escape(field)}\b",
                        body,
                        re.IGNORECASE,
                    )
                    or (
                        field == param_name
                        and re.search(rf"\b{re.escape(field)}\b", body)
                    )
                )
                if guarded:
                    action = "guarded-use"
                elif call_uses_timestamp and field.lower() in {"deadline", "expiry", "expiration", "expiresat"} and not direct_use:
                    action = "ignore-or-overwrite"
                elif re.search(rf"\b{re.escape(field)}\b\s*=", body, re.IGNORECASE):
                    action = "overwrite"
                flows.append({
                    "field": field,
                    "action": action,
                    "function_name": function.name,
                    "line": function.start_line + body[:(
                        nested_uses[0].start() if nested_uses else 0
                    )].count("\n"),
                    "source": field_expr,
                    "evidence_lines": [function.start_line],
                })

        for field_match in _TEMPORAL_FIELD_RE.finditer(body):
            field = field_match.group(0)
            if any(flow["field"] == field and flow["function_name"] == function.name for flow in flows):
                continue
            if re.search(rf"\b{re.escape(field)}\b\s*=", body, re.IGNORECASE):
                flows.append({
                    "field": field,
                    "action": "state-write",
                    "function_name": function.name,
                    "line": function.start_line + body[:field_match.start()].count("\n"),
                    "source": field,
                    "evidence_lines": [function.start_line + body[:field_match.start()].count("\n")],
                })
    return flows


def build_temporal_ir(source: str) -> dict:
    """Build a conservative source-grounded temporal intermediate representation.

    The IR intentionally records syntax and local data flow only. It never
    invents a vulnerability label; higher-level invariant rules decide whether
    a recorded relation is unsafe.
    """
    regions = _function_regions_with_offsets(source)
    symbols = _ir_symbol_declarations(source, regions)
    function_facts: list[dict] = []
    all_aliases: list[dict] = []
    all_guards: list[dict] = []
    all_state_writes: list[dict] = []
    all_asset_actions: list[dict] = []
    all_external_calls: list[dict] = []
    all_calls: list[dict] = []
    timestamp_reads: list[dict] = []

    for function, offset in regions:
        header_end = function.text.find("{")
        header = function.text[:header_end] if header_end >= 0 else function.text
        body = _strip_comments_preserve_layout(function.text[header_end + 1:]) if header_end >= 0 else ""
        guards = _ir_guard_records(function)
        assignments = []
        for match in _IR_ASSIGNMENT_RE.finditer(body):
            lhs = re.sub(r"\s+", "", match.group("lhs"))
            rhs = match.group("rhs").strip()
            line = function.start_line + body[:match.start()].count("\n")
            assignments.append({
                "name": lhs,
                "source": rhs,
                "function_name": function.name,
                "line": line,
                "kind": "timestamp-alias" if _TEMPORAL_CALL_RE.search(rhs) else "assignment",
            })
            if _TEMPORAL_CALL_RE.search(rhs):
                all_aliases.append(assignments[-1])

        reads = []
        for match in _TEMPORAL_CALL_RE.finditer(body):
            record = {
                "symbol": match.group(0),
                "function_name": function.name,
                "line": function.start_line + body[:match.start()].count("\n"),
                "expression": match.group(0),
            }
            reads.append(record)
            timestamp_reads.append(record)

        state_writes = []
        for match in _IR_ASSIGNMENT_RE.finditer(body):
            lhs = re.sub(r"\s+", "", match.group("lhs"))
            line = function.start_line + body[:match.start()].count("\n")
            if re.search(r"\.|\[|\b(?:state|storage|mapping|balances|config|vault|order|position)\b", lhs, re.IGNORECASE):
                state_writes.append({
                    "field": lhs,
                    "action": match.group("op"),
                    "function_name": function.name,
                    "line": line,
                    "expression": match.group(0).strip(),
                    "guarded_by_time": any(
                        guard["line"] <= line and guard["function_name"] == function.name
                        for guard in guards
                    ),
                })
        all_state_writes.extend(state_writes)

        asset_actions = []
        for match in _IR_ASSET_CALL_RE.finditer(body):
            action = match.group(0).split("(", 1)[0].strip()
            asset_actions.append({
                "action": action,
                "function_name": function.name,
                "line": function.start_line + body[:match.start()].count("\n"),
                "expression": body[match.start():body.find(")", match.start()) + 1] if ")" in body[match.start():] else action,
            })
        all_asset_actions.extend(asset_actions)

        external_calls = []
        for match in _IR_EXTERNAL_CALL_RE.finditer(body):
            method = match.group("method")
            if method in _IR_CONTROL_WORDS:
                continue
            external_calls.append({
                "receiver": re.sub(r"\s+", "", match.group("receiver")),
                "method": method,
                "function_name": function.name,
                "line": function.start_line + body[:match.start()].count("\n"),
            })
        all_external_calls.extend(external_calls)

        calls = _ir_function_calls(function)
        all_calls.extend(calls)
        function_facts.append({
            "name": function.name,
            "start_line": function.start_line,
            "end_line": function.end_line,
            "visibility": next(
                (token for token in ("external", "public", "internal", "private")
                 if re.search(rf"\b{token}\b", header, re.IGNORECASE)),
                "",
            ),
            "parameters": _ir_parameter_specs(header),
            "timestamp_reads": reads,
            "aliases": [item for item in all_aliases if item["function_name"] == function.name],
            "guards": guards,
            "state_writes": state_writes,
            "asset_actions": asset_actions,
            "external_calls": external_calls,
            "calls": calls,
            "missing_invariants": [],
            "evidence_lines": sorted({item["line"] for item in [*reads, *guards, *state_writes, *asset_actions]}),
        })

    field_flows = _ir_field_flow_records(source, regions, function_facts)
    for function in function_facts:
        names = {param["name"] for param in function["parameters"]}
        text = next((item.text for item, _ in regions if item.name == function["name"]), "")
        if any(name.lower() in {"signature", "sig", "proof"} for name in names):
            if not re.search(r"\b(?:deadline|expiry|expiration|validUntil|expiresAt)\b", text, re.IGNORECASE):
                function["missing_invariants"].append("signed_lifecycle_bound")
        if re.search(r"\b(?:recoverSigner|ecrecover|SignatureChecker|ECDSA)\b", text, re.IGNORECASE):
            if not re.search(r"\b(?:deadline|expiry|expiration|validUntil|expiresAt)\b", text, re.IGNORECASE):
                function["missing_invariants"].append("signed_lifecycle_bound")

    relations: list[dict] = []
    for function in function_facts:
        for guard in function["guards"]:
            targets = [
                write for write in function["state_writes"]
                if write["line"] >= guard["line"]
            ] + [
                action for action in function["asset_actions"]
                if action["line"] >= guard["line"]
            ]
            for target in targets:
                relations.append({
                    "kind": "timestamp-alias-to-guard-to-action",
                    "function_name": function["name"],
                    "source": guard["timestamp_terms"] or guard["fields"],
                    "guard_line": guard["line"],
                    "target": target.get("field") or target.get("action"),
                    "target_line": target["line"],
                    "relation": "controls",
                    "evidence_lines": [guard["line"], target["line"]],
                })
        for alias in function["aliases"]:
            if _TEMPORAL_CALL_RE.search(alias["source"]):
                for guard in function["guards"]:
                    if re.search(rf"\b{re.escape(alias['name'])}\b", guard["condition"]):
                        relations.append({
                            "kind": "alias-to-guard",
                            "function_name": function["name"],
                            "source": alias["name"],
                            "guard_line": guard["line"],
                            "target": guard["condition"],
                            "target_line": guard["line"],
                            "relation": "propagates",
                            "evidence_lines": [alias["line"], guard["line"]],
                        })

    for function in function_facts:
        for call in function["calls"]:
            if any(target["name"] == call["name"] for target in function_facts):
                relations.append({
                    "kind": "helper-call",
                    "function_name": function["name"],
                    "source": function["name"],
                    "guard_line": call["line"],
                    "target": call["name"],
                    "target_line": call["line"],
                    "relation": "calls",
                    "evidence_lines": [call["line"]],
                })

    return {
        "schema_version": "temporal-ir-v2",
        "timestamp_reads": timestamp_reads,
        "aliases": all_aliases,
        "calls": all_calls,
        "guards": all_guards or [guard for function in function_facts for guard in function["guards"]],
        "symbols": symbols,
        "state_writes": all_state_writes,
        "asset_actions": all_asset_actions,
        "external_calls": all_external_calls,
        "relations": relations,
        "field_flows": field_flows,
        "functions": function_facts,
    }


def _slice_function_names(source: str, evidence: Any, ir: dict) -> set[str]:
    names: set[str] = set()
    if isinstance(evidence, str) and evidence:
        names.add(evidence)
    elif isinstance(evidence, dict):
        evidence = [evidence]
    if isinstance(evidence, (list, tuple, set)):
        for item in evidence:
            if isinstance(item, str):
                names.add(item)
                continue
            if not isinstance(item, dict):
                continue
            for key in ("function_name", "function", "functions", "related_functions"):
                value = item.get(key)
                if isinstance(value, str):
                    names.add(value)
                elif isinstance(value, (list, tuple, set)):
                    names.update(str(name) for name in value if name)
    if not names:
        for function in ir.get("functions", []):
            if function.get("timestamp_reads") or function.get("guards") or function.get("missing_invariants"):
                names.add(function["name"])
            elif function.get("parameters") and any(
                _TEMPORAL_FIELD_RE.search(param.get("name", ""))
                for param in function["parameters"]
            ):
                names.add(function["name"])

    function_map = {function["name"]: function for function in ir.get("functions", [])}
    # Include one-hop helper closure so a caller and the temporal helper remain
    # visible in the same compact prompt context.
    for _ in range(2):
        for name in list(names):
            function = function_map.get(name)
            if not function:
                continue
            names.update(call["name"] for call in function.get("calls", []) if call["name"] in function_map)
    return names


def build_temporal_slice(source: str, function_name: Any = "") -> str:
    """Return a bounded source slice retaining temporal root-cause functions."""
    ir = build_temporal_ir(source)
    selected = _slice_function_names(source, function_name, ir)
    lines = source.splitlines()
    rendered: list[str] = ["[TEMPORAL SLICE]"]
    if ir.get("field_flows"):
        rendered.append("[TEMPORAL FIELD DEF-USE]")
        for flow in ir["field_flows"]:
            if not selected or flow.get("function_name") in selected:
                rendered.append(
                    f"{flow['function_name']} L{flow['line']}: "
                    f"{flow['field']} -> {flow['action']}"
                )
    for function, _offset in _function_regions_with_offsets(source):
        if selected and function.name not in selected:
            continue
        start = max(function.start_line - 1, 0)
        end = min(function.end_line, len(lines))
        rendered.append(
            f"[FUNCTION {function.name} L{function.start_line}-{function.end_line}]"
        )
        rendered.extend(
            f"L{index + 1}: {lines[index]}"
            for index in range(start, end)
        )

    if len(rendered) == 1:
        return ""
    return "\n".join(rendered)


def summarize_temporal_ir(ir: dict) -> str:
    """Render a compact deterministic summary suitable for the model prompt."""
    lines = ["[TEMPORAL IR v2]"]
    lines.append(
        "timestamp_reads=" + str(len(ir.get("timestamp_reads", [])))
        + " aliases=" + str(len(ir.get("aliases", [])))
        + " guards=" + str(len(ir.get("guards", [])))
        + " state_writes=" + str(len(ir.get("state_writes", [])))
        + " asset_actions=" + str(len(ir.get("asset_actions", [])))
    )
    for relation in ir.get("relations", [])[:80]:
        source = ",".join(map(str, relation.get("source", [])))
        lines.append(
            f"{relation.get('kind', 'relation')}: {source} -> "
            f"{relation.get('target', '')} "
            f"(L{relation.get('guard_line', 0)} -> L{relation.get('target_line', 0)})"
        )
    for flow in ir.get("field_flows", [])[:80]:
        lines.append(
            f"field_flow: {flow.get('field')} {flow.get('action')} "
            f"in {flow.get('function_name')} L{flow.get('line')}"
        )
    return "\n".join(lines)

