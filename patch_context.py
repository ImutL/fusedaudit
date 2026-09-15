"""Bound paired historical code around actual edits, without target labels."""

from __future__ import annotations

import difflib
import re


def bounded_pair_views(
    vulnerable: str, patched: str, document_limit: int, patch_limit: int,
) -> tuple[str, str, dict] | None:
    # Keep literals intact; formatting-only differences should not consume
    # the patch budget ahead of a changed guard or call.
    token_pattern = r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\w+|[^\w\s]'''
    left = list(re.finditer(token_pattern, vulnerable))
    right = list(re.finditer(token_pattern, patched))
    if not left or not right or min(document_limit, patch_limit) <= 0:
        return None
    matcher = difflib.SequenceMatcher(
        None, [t.group() for t in left], [t.group() for t in right], autojunk=False)
    groups = list(matcher.get_grouped_opcodes(n=8))
    if not groups:
        return None
    receipt = {"paired_change_visible": True, "total_change_groups": len(groups)}
    if len(vulnerable) <= document_limit and len(patched) <= patch_limit:
        return vulnerable, patched, {
            **receipt, "prompt_pair_view": "full_pair", "visible_change_groups": len(groups)}

    def window(document, tokens, start, stop):
        start = min(start, len(tokens) - 1)
        stop = min(max(stop, start + 1), len(tokens))
        begin = document.rfind("\n", 0, tokens[start].start()) + 1
        end = document.find("\n", tokens[stop - 1].end())
        if end < 0:
            end = len(document)
        first = document.count("\n", 0, begin) + 1
        last = document.count("\n", 0, end) + 1
        return f"[Historical document lines {first}-{last}]\n" + document[begin:end]

    before, after = [], []
    visible = 0
    for group in groups:
        old = window(vulnerable, left, group[0][1], group[-1][2])
        new = window(patched, right, group[0][3], group[-1][4])
        old_text = "\n...\n".join([*before, old])
        new_text = "\n...\n".join([*after, new])
        if len(old_text) <= document_limit and len(new_text) <= patch_limit:
            before.append(old)
            after.append(new)
            visible += 1
    if not visible:
        return None
    return "\n...\n".join(before), "\n...\n".join(after), {
        **receipt, "prompt_pair_view": "changed_line_windows", "visible_change_groups": visible}
