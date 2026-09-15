"""Source-local proposal, locator, and semantic-admission helpers.

This module keeps a broad candidate proposal pool and applies protection
semantics only at admission.  It is deliberately source-local: it never
consults a dataset/source id and never calls a model or network service.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from functools import lru_cache
from typing import Any, Iterable


LABELS = frozenset(
    {
        "access_control",
        "arithmetic",
        "front_running",
        "reentrancy",
        "time_manipulation",
        "unchecked_low_level_calls",
    }
)

_FUNCTION_HEADER = re.compile(
    r"\b(?:function\s+(?P<function>[A-Za-z_]\w*)|(?P<special>constructor|fallback|receive))\s*\(",
    re.I,
)
_CONTRACT_HEADER = re.compile(
    r"^[ \t]*(?:abstract\s+)?(?:contract|interface|library)\s+"
    r"(?P<name>[A-Za-z_]\w*)[^\{;]*\{",
    re.I | re.M,
)
_EXTERNAL_CALL = re.compile(
    r"\.(?:call|send|delegatecall|staticcall|transfer|transferFrom|approve|burn|mint|"
    r"safeTransfer|safeTransferFrom|safeApprove|safeIncreaseAllowance|safeDecreaseAllowance|"
    r"sendValue|functionCall|functionStaticCall|functionDelegateCall|uniTransfer|uniTransferFrom|"
    r"deposit|withdraw|stake|unstake|stakingRate|swap|execute|settle|liquidate|flashLoan|"
    r"sendETH|sendEther|withdrawETH|withdrawEther)\s*(?:\.|\{|\x28)",
    re.I,
)
_ASSET_OPERATION_CALL = re.compile(
    r"\.(?:transfer|transferFrom|safeTransfer|safeTransferFrom|sendValue|"
    r"deposit|withdraw|stake|unstake|swap|execute|settle|liquidate|flashLoan|"
    r"sendETH|sendEther|withdrawETH|withdrawEther)\s*\(",
    re.I,
)
_SAFE_ERC20_CALLBACK_CALL = re.compile(
    r"\b[A-Za-z_]\w*\s*\.\s*(?:safeTransferFrom|safeTransfer)\s*\(",
    re.I,
)
_GLOBAL_RESOURCE_WRITE = re.compile(
    r"\b(?:_?totalSupply|owner|admin(?:s)?|governance|guardian|operator|"
    r"minter|pauser|roles?|permissions?|allow(?:ed|list)|currentModel|"
    r"stakingRateModel|router|controller|treasury|vault|implementation|"
    r"initialized|emissionRate|endBlock|startBlock)\b"
    r"(?:\s*\[[^\]]+\])?\s*(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)",
    re.I,
)
_CALLER_SCOPED_RESOURCE_WRITE = re.compile(
    r"\b(?:balances?|staked\w*|votingPower|vaults?|troves?|stake\w*|"
    r"reward\w*|referral\w*|deposit\w*|claim\w*)\s*\[[^\]]*"
    r"(?:msg\s*\.\s*sender|_msgSender\s*\(\)|caller)\b",
    re.I,
)
_WORKFLOW_STATE_ROOT = re.compile(
    r"\b(?:commitments?|nonces?|requests?|orders?|pending\w*|"
    r"reveals?|reimbursements?|refunds?)\b",
    re.I,
)
_CALLER_REFUND_TRANSFER = re.compile(
    r"\bmsg\s*\.\s*sender\s*\.\s*transfer\s*\(\s*"
    r"msg\s*\.\s*value(?:\s*[-+]\s*[^)\n;]+)?\s*\)",
    re.I,
)
_SUBJECT_SCOPED_ASSET_PATH = re.compile(
    r"\b(?:getTrove\w*|getPosition\w*|getCollateral\w*|getDebt\w*|"
    r"closeTrove|removeStake|applyPendingRewards|sendETH|sendEther|"
    r"_repay\w*|repay\w*|withdraw\w*|redeem\w*|unstake\w*|"
    r"safeTransfer(?:From)?|transfer(?:From)?)\s*\([^;{}\n]*"
    r"\b(?:msg\s*\.\s*sender|_msgSender\s*\(\))\b",
    re.I,
)
_LOW_LEVEL_CALL = re.compile(r"\.(?:call|send|delegatecall|staticcall)\s*(?:\.|\{|\x28)", re.I)
_STATE_WRITE = re.compile(
    r"(?:\b[A-Za-z_]\w*(?:\s*\[[^\]\n]+\])?(?:\.[A-Za-z_]\w+)?\s*"
    r"(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)|\bdelete\s+[A-Za-z_]\w*)"
)
_MEMBER_CALL = re.compile(
    r"(?P<receiver>[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w+)*"
    r"(?:\s*\([^;{}]*\))?(?:\s*\[[^\]]*\])?)\s*\.\s*"
    r"(?P<method>[A-Za-z_]\w*)\s*(?:\{[^{}]*\})?\(",
    re.I | re.S,
)
_READ_ONLY_METHOD = re.compile(
    r"^(?:balanceof|totalsupply|allowance|decimals|symbol|name|"
    r"supportsinterface|invariant|latest[a-z_]*|get[a-z_]*|is[a-z_]*|"
    r"quote[a-z_]*|price[a-z_]*|view[a-z_]*)$",
    re.I,
)
_CALLBACK_METHOD = re.compile(
    r"(?:callback|tokensreceived|on[a-z_]*|executeoperation|"
    r"flashloan|fallback|receive)",
    re.I,
)
_LIBRARY_STATE_METHOD = frozenset(
    {
        "allocate",
        "deposit",
        "withdraw",
        "remove",
        "swap",
        "mint",
        "burn",
        "settle",
        "update",
        "remove",
    }
)
_LOCAL_LIBRARY_METHOD = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "decadd",
        "decsub",
        "decmul",
        "decdiv",
        "mod",
        "pow",
        "min",
        "max",
        "abs",
        "toint",
        "touint",
        "touint112",
        "touint256",
        "toint256",
        "encode",
        "decode",
        "concat",
        "push",
        "pop",
        "at",
        "contains",
        "remove",
    }
)
_ASSET_SINK_METHOD = frozenset(
    {
        "transfer",
        "transferfrom",
        "safetransfer",
        "safetransferfrom",
        "sendvalue",
        "send",
        "deposit",
        "withdraw",
        "stake",
        "unstake",
        "swap",
        "execute",
        "settle",
        "liquidate",
        "flashloan",
        "sendeth",
        "sendether",
        "withdraweth",
        "withdrawether",
        "callexchange",
        "executeorder",
        "executeorders",
        "fillorder",
        "fillorders",
        "settleorder",
        "settleorders",
        "processorder",
        "processorders",
    }
)
_ECONOMIC_DISPATCH_METHOD = frozenset(
    {
        "callexchange",
        "executeorder",
        "executeorders",
        "fillorder",
        "fillorders",
        "settleorder",
        "settleorders",
        "processorder",
        "processorders",
    }
)
_INBOUND_ASSET_METHOD = frozenset(
    {"transferfrom", "safetransferfrom", "deposit", "stake", "flashloan"}
)
_TIME_PROVIDER_NAME = re.compile(
    r"(?:get|read|current|fetch|load)?(?:block)?(?:timestamp|time|now)$",
    re.I,
)
_ARITHMETIC_HELPER_CALL = re.compile(
    r"(?<![.\w])(?:[A-Za-z_]\w*\s*\.\s*)?"
    r"(?P<method>safePower|bpow|pow|power|exp)\s*\(",
    re.I,
)
_PRIVILEGED_ASSET_FUNCTION = re.compile(
    r"^(?:rescue|sweep|recover|salvage|seize|drain|emergency\w*|"
    r"withdrawall|withdrawtoken|upgrade\w*|set\w*|init(?:ialize)?\w*)$",
    re.I,
)
_QUALIFIED_CALL = re.compile(
    r"(?P<receiver>(?:[A-Za-z_]\w*|[A-Za-z_]\w*\s*\(\s*[A-Za-z_]\w*\s*\)))"
    r"\s*\.\s*(?P<method>[A-Za-z_]\w*)\s*"
    r"(?:(?:\.\s*[A-Za-z_]\w*)|(?:\.\s*[A-Za-z_]\w*\s*\([^)]*\))|(?:\{[^{}]*\}))*\s*\(",
    re.I | re.S,
)
_NATIVE_VALUE_TRANSFER = re.compile(
    r"(?P<recipient>"
    r"msg\s*\.\s*sender"
    r"|payable\s*\(\s*msg\s*\.\s*sender\s*\)"
    r"|address\s*\(\s*msg\s*\.\s*sender\s*\)"
    r")\s*\.\s*transfer\s*\(",
    re.I,
)
_NATIVE_TRANSFER_CALL = re.compile(
    r"(?P<receiver>msg\s*\.\s*sender|_msgSender\s*\(\)|"
    r"payable\s*\([^)]*\)|address\s*\([^)]*\)|[A-Za-z_]\w*)\s*\.\s*"
    r"(?:transfer|send)\s*\(",
    re.I,
)
_BARE_EXTERNAL_EFFECT_METHOD = re.compile(
    r"(?<![.\w])(?P<method>safeTransferFrom|safeTransfer|sendValue|"
    r"functionCallWithValue|functionCall|functionStaticCall|functionDelegateCall)\s*\(",
    re.I,
)
_KNOWN_EXTERNAL_RETURN_METHODS = frozenset(
    {
        "approve",
        "transfer",
        "transferfrom",
        "redeem",
        "withdraw",
        "withdrawall",
        "deposit",
        "harvest",
        "execute",
        "swap",
        "swapexacttokensfortokens",
        "swapexacttokensforeth",
        "swapexactethfortokens",
        "swaptokensforexacttokens",
        "swaptokensforexacteth",
        "swapethforexacttokens",
    }
)
_BARE_CALL = re.compile(r"(?<![.\w])(?P<name>[A-Za-z_]\w*)\s*\(")
_AUTH_MODIFIER = re.compile(
    r"(?:^|\b)(?:only(?:owner|role|minter|pauser|guardian|governance|operator|manager|"
    r"controller|delegate|dao|protocol)|authorized|auth(?:orized)?|governance|admin)\b",
    re.I,
)
_AUTH_BODY = re.compile(
    r"\b(?:hasRole|_checkRole|isAuthorized|isWhitelisted|isAllowed)\s*\(|"
    r"\b(?:require|assert|if)\s*\([^;{}]*"
    r"(?:msg\s*\.\s*sender|_msgSender\s*\(|tx\s*\.\s*origin)"
    r"[^;{}]*(?:==|!=|owner|admin|guardian|operator|governance|manager|role|"
    r"permission|authorized|whitelist|allowlist|roles)\b[^;{}]*\)",
    re.I | re.S,
)
_MUTEX = re.compile(r"\b(?:nonreentrant|reentrancyguard|mutex|locked|_status|_entered)\b", re.I)
_MUTEX_NAME = re.compile(r"\b(?:nonreentrant|reentrancyguard|mutex)\b", re.I)
_MUTEX_STATE = re.compile(r"\b(?:_status|_entered|entered|locked|mutex|reentrancy(?:guard)?state)\b", re.I)
_SEMANTIC_SINK_NAMES = {
    "persistent critical state",
    "asset operation",
    "asset sink",
    "rw_conflict_only",
    "critical state",
}
_REENTRANCY_SINK_HELPER_NAME = re.compile(
    r"^(?:safe|transfer|update|do|process|send|mint|burn)\w*$",
    re.I,
)
_REENTRANCY_READ_HELPER_NAME = re.compile(
    r"^(?:get|read|fetch|load|query|calculate|compute|quote|rate|price|oracle)\w*$",
    re.I,
)
_FRONT_RUNNING_EXPLICIT_SINK_NAME = re.compile(
    r"^_(?:safe(?:swap|execute)|execute|reinvest|swap|trade|quote|liquidate)\w*$",
    re.I,
)
_FINANCIAL_FLOW_TOKEN = re.compile(
    r"\b(?:reserve\w*|liquidity\w*|invariant\w*|quote\w*|price\w*|amountout\w*|"
    r"amountin\w*|amount\w*|value\w*|reward\w*|collateral\w*|share\w*|"
    r"supply\w*|fee\w*|validator\w*|commitment\w*|calibrat\w*|exchange\w*|"
    r"pool\w*|balance\w*)\b",
    re.I,
)
_FINANCIAL_ORDER_OPERATION = re.compile(
    r"\b(?:swap\w*|allocate\w*|quote\w*|invariant\w*|amountout\w*|"
    r"calibrat\w*|callback\w*|burn\w*|redeem\w*|deposit\w*|withdraw\w*|"
    r"liquidat\w*|reinvest\w*|harvest\w*|commit\w*|provide\w*|execute\w*|"
    r"exchange\w*)\b",
    re.I,
)
_FINANCIAL_ORDER_SENSITIVE = re.compile(
    r"\b(?:quote\w*|price\w*|amountout\w*|amountin\w*|invariant\w*|"
    r"exchange\w*|order\w*|slippage\w*|minimum\w*|deadline\w*|nonce\w*|"
    r"commit\w*|reveal\w*|swap\w*|settle\w*|execute\w*|fill\w*|"
    r"liquidat\w*|reinvest\w*|harvest\w*|compound\w*)\b",
    re.I,
)
_ORDER_PROTECTION_TOKEN = re.compile(
    r"\b(?:minout\w*|slippage\w*|deadline\w*|commit\w*|reveal\w*|nonce\w*|"
    r"amountoutmin\w*|minimum\w*output\w*)\b",
    re.I,
)
_INHERITED_MUTATING_METHOD = re.compile(
    r"^_?(?:burn|mint|transfer|withdraw|deposit|claim|redeem|stake|unstake|approve|"
    r"set|upgrade|pause|unpause)\w*$",
    re.I,
)
_TIME_LOCATOR = re.compile(r"\b(?:block\s*\.\s*(?:timestamp|number)|now)\b", re.I)
_TIME_CONDITION_LOCATOR = re.compile(
    r"\b(?:if|require|assert)\s*\([^\n{}]*\b"
    r"(?:block\s*\.\s*timestamp|now|time\w*|timestamp\w*|elapsed\w*|deadline\w*|duration\w*|epoch\w*|start|end)\b",
    re.I,
)
_ARITHMETIC_LOCATOR = re.compile(
    r"(?:\+\+|--|\+=|-=|\*=|/=|\+|-|\*|/|%|\.\s*(?:add|sub|mul|div|mod)\s*\()",
    re.I,
)
_DECLARATION_MODIFIERS = frozenset(
    {
        "public",
        "private",
        "internal",
        "external",
        "constant",
        "immutable",
        "memory",
        "storage",
        "calldata",
        "override",
        "virtual",
        "payable",
        "transient",
    }
)
_DECLARATION_KEYWORDS = frozenset(
    {
        "assert",
        "contract",
        "delete",
        "emit",
        "for",
        "if",
        "import",
        "interface",
        "library",
        "mapping",
        "new",
        "pragma",
        "require",
        "return",
        "returns",
        "revert",
        "struct",
        "throw",
        "using",
        "while",
    }
)
# Imported interfaces are not present in the blinded source.  These are
# receiver-type signatures, rather than a global method-name allowlist, and
# are used only when the source explicitly declares the receiver type.
_KNOWN_INTERFACE_METHODS: dict[str, dict[str, tuple[str, ...]]] = {
    "ierc20": {
        "totalSupply": ("uint256",),
        "balanceOf": ("uint256",),
        "transfer": ("bool",),
        "allowance": ("uint256",),
        "approve": ("bool",),
        "transferFrom": ("bool",),
    },
    # OpenZeppelin and older DApp sources commonly use the concrete/imported
    # ``ERC20`` type without including its definition in the blinded file.
    # Keep the signature receiver-scoped; this is not a global method-name
    # heuristic.
    "erc20": {
        "totalSupply": ("uint256",),
        "balanceOf": ("uint256",),
        "transfer": ("bool",),
        "allowance": ("uint256",),
        "approve": ("bool",),
        "transferFrom": ("bool",),
    },
    "ibep20": {
        "totalSupply": ("uint256",),
        "balanceOf": ("uint256",),
        "transfer": ("bool",),
        "allowance": ("uint256",),
        "approve": ("bool",),
        "transferFrom": ("bool",),
    },
    # Imported interfaces are absent from blinded source files.  Keep these
    # signatures keyed by their declared receiver type, never by method name.
    "itimelock": {
        "queueTransaction": ("bytes32",),
        "executeTransaction": ("bytes",),
    },
    "projectwallet": {
        "transfer": ("bool",),
    },
    # RocketPool 2.5's imported interface declares these mutating calls as
    # ``returns (bool)`` even though an older RocketPool interface omitted the
    # return clause.  The receiver type is explicit in the source, so this
    # does not become a global transfer-name heuristic.
    "rocketvaultinterface": {
        "depositToken": ("bool",),
        "withdrawToken": ("bool",),
        "transferToken": ("bool",),
    },
    "uniswaprouterv2": {
        "swapExactTokensForTokens": ("uint256[]",),
        "swapExactTokensForETH": ("uint256[]",),
        "swapExactETHForTokens": ("uint256[]",),
        "swapTokensForExactTokens": ("uint256[]",),
        "swapTokensForExactETH": ("uint256[]",),
        "swapETHForExactTokens": ("uint256[]",),
        "getAmountsOut": ("uint256[]",),
        "getAmountsIn": ("uint256[]",),
    },
    "iuniswapv2router": {
        "swapExactTokensForTokens": ("uint256[]",),
        "swapExactTokensForETH": ("uint256[]",),
        "swapExactETHForTokens": ("uint256[]",),
        "swapTokensForExactTokens": ("uint256[]",),
        "swapTokensForExactETH": ("uint256[]",),
        "swapETHForExactTokens": ("uint256[]",),
        "getAmountsOut": ("uint256[]",),
        "getAmountsIn": ("uint256[]",),
    },
    "iuniswapv2router02": {
        "swapExactTokensForTokens": ("uint256[]",),
        "swapExactTokensForETH": ("uint256[]",),
        "swapExactETHForTokens": ("uint256[]",),
        "swapTokensForExactTokens": ("uint256[]",),
        "swapTokensForExactETH": ("uint256[]",),
        "swapETHForExactTokens": ("uint256[]",),
        "getAmountsOut": ("uint256[]",),
        "getAmountsIn": ("uint256[]",),
    },
    "ipancakerouter02": {
        "swapExactTokensForTokens": ("uint256[]",),
        "swapExactTokensForETH": ("uint256[]",),
        "swapExactETHForTokens": ("uint256[]",),
        "swapTokensForExactTokens": ("uint256[]",),
        "swapTokensForExactETH": ("uint256[]",),
        "swapETHForExactTokens": ("uint256[]",),
        "getAmountsOut": ("uint256[]",),
        "getAmountsIn": ("uint256[]",),
    },
}


@dataclass(frozen=True)
class FunctionSpan:
    name: str
    start_line: int
    end_line: int
    visibility: str = ""
    modifiers: tuple[str, ...] = ()
    is_constructor: bool = False
    is_fallback: bool = False
    is_receive: bool = False
    state_mutability: str = ""
    contract_name: str = ""


@dataclass(frozen=True)
class StatementSpan:
    start_offset: int
    end_offset: int
    start_line: int
    end_line: int
    text: str


def _line_at(source: str, offset: int) -> int:
    return source.count("\n", 0, max(offset, 0)) + 1


def _balanced_end(source: str, brace_offset: int) -> int:
    depth = 0
    for index in range(brace_offset, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(source) - 1


def _mask_comments_preserve_lines(text: str) -> str:
    """Blank comments without changing offsets or source line numbers."""

    chars = list(text or "")
    state = "code"
    quote = ""
    escaped = False
    index = 0
    while index < len(chars):
        char = chars[index]
        next_char = chars[index + 1] if index + 1 < len(chars) else ""
        if state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                chars[index] = " "
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                chars[index] = " "
                chars[index + 1] = " "
                index += 2
                state = "code"
                continue
            if char != "\n":
                chars[index] = " "
            index += 1
            continue
        if state == "string":
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                state = "code"
            index += 1
            continue
        if char == "/" and next_char == "/":
            chars[index] = " "
            chars[index + 1] = " "
            index += 2
            state = "line_comment"
            continue
        if char == "/" and next_char == "*":
            chars[index] = " "
            chars[index + 1] = " "
            index += 2
            state = "block_comment"
            continue
        if char in {"'", '"'}:
            quote = char
            state = "string"
        index += 1
    return "".join(chars)


def _contract_name_at_line_masked(
    masked: str,
    line_number: int,
    line_starts: list[int] | None = None,
) -> str:
    """Resolve a contract name from an already comment-masked source view."""

    if line_number <= 0:
        return ""
    line_starts = line_starts or _line_start_offsets(masked)
    if line_number > len(line_starts):
        return ""
    target = line_starts[line_number - 1]
    selected: tuple[int, str] | None = None
    for match in _CONTRACT_HEADER.finditer(masked):
        opening = masked.find("{", match.start(), match.end())
        if opening < 0 or opening > target:
            continue
        closing = _balanced_end(masked, opening)
        if target <= closing and (selected is None or opening > selected[0]):
            selected = (opening, match.group("name"))
    return selected[1] if selected else ""


def _contract_name_at_line(source: str, line_number: int) -> str:
    """Return the innermost contract/interface/library containing a line."""

    return _contract_name_at_line_masked(
        _mask_comments_preserve_lines(source),
        line_number,
    )


def _header_attributes(header: str) -> tuple[str, tuple[str, ...], str]:
    visibility = ""
    for value in ("external", "public", "internal", "private"):
        if re.search(rf"\b{value}\b", header, re.I):
            visibility = value
            break
    mutability = ""
    for value in ("view", "pure", "payable", "nonpayable"):
        if re.search(rf"\b{value}\b", header, re.I):
            mutability = value
            break
    modifiers: list[str] = []
    tail = re.sub(r"\b(?:function|constructor|fallback|receive)\b[^\{;]*", "", header, flags=re.I)
    for token in re.findall(r"\b[A-Za-z_]\w*\s*(?:\([^)]*\))?", tail):
        name = token.split("(", 1)[0].strip()
        if name and name.casefold() not in {"returns", "view", "pure", "payable", "external", "public", "internal", "private"}:
            modifiers.append(name)
    return visibility, tuple(dict.fromkeys(modifiers)), mutability


def _regex_function_spans(source: str) -> list[FunctionSpan]:
    spans: list[FunctionSpan] = []
    source = source or ""
    # Mask comments once before scanning.  The previous implementation
    # re-masked and re-scanned the full source for every function header,
    # which made comment-heavy contracts effectively quadratic.
    masked = _mask_comments_preserve_lines(source)
    line_starts = _line_start_offsets(masked)
    contract_name_cache: dict[int, str] = {}
    for match in _FUNCTION_HEADER.finditer(masked):
        start = match.start()
        header_end = masked.find("{", match.end())
        declaration_end = masked.find(";", match.end())
        if header_end < 0 or (declaration_end >= 0 and declaration_end < header_end):
            header_end = declaration_end
        if header_end < 0:
            header_end = match.end()
        header = source[start:header_end]
        end_offset = (
            _balanced_end(masked, header_end)
            if header_end < len(masked) and masked[header_end] == "{"
            else header_end
        )
        special = (match.group("special") or "").casefold()
        name = match.group("function") or special
        visibility, modifiers, mutability = _header_attributes(header)
        start_line = _line_at(source, start)
        if start_line not in contract_name_cache:
            contract_name_cache[start_line] = _contract_name_at_line_masked(
                masked,
                start_line,
                line_starts,
            )
        spans.append(
            FunctionSpan(
                name=name,
                start_line=start_line,
                end_line=_line_at(source, end_offset),
                visibility=visibility,
                modifiers=modifiers,
                is_constructor=special == "constructor",
                is_fallback=special == "fallback",
                is_receive=special == "receive",
                state_mutability=mutability,
                contract_name=contract_name_cache[start_line],
            )
        )
    return spans


def _parse_function_spans_uncached(source: str) -> list[FunctionSpan]:
    """Read parser-owned function ranges, with a balanced-brace fallback."""

    source = source or ""
    # Parser-owned spans are common, so resolve contract names from one shared
    # masked view instead of re-scanning comments for every function.
    masked = _mask_comments_preserve_lines(source)
    line_starts = _line_start_offsets(masked)
    contract_name_cache: dict[int, str] = {}
    spans: list[FunctionSpan] = []
    try:
        from feature_fusion import extract_contract_features

        for contract in extract_contract_features(source):
            for function in getattr(contract, "functions", []) or []:
                name = str(getattr(function, "name", "") or "").strip()
                start = int(getattr(function, "line_number", 0) or 0)
                end = int(getattr(function, "end_line_number", 0) or 0)
                if not name or start <= 0 or end < start:
                    continue
                if start not in contract_name_cache:
                    contract_name_cache[start] = _contract_name_at_line_masked(
                        masked,
                        start,
                        line_starts,
                    )
                spans.append(
                    FunctionSpan(
                        name=name,
                        start_line=start,
                        end_line=end,
                        visibility=str(getattr(function, "visibility", "") or ""),
                        modifiers=tuple(str(v) for v in (getattr(function, "modifiers", []) or [])),
                        is_constructor=bool(getattr(function, "is_constructor", False)),
                        is_fallback=bool(getattr(function, "is_fallback", False)),
                        is_receive=bool(getattr(function, "is_receive", False)),
                        state_mutability=str(getattr(function, "state_mutability", "") or ""),
                        contract_name=contract_name_cache[start],
                    )
                )
    except Exception:
        spans = []
    # Tree-sitter versions in the existing runtime do not expose every
    # special-function node (notably bare fallback/receive declarations).  The
    # balanced lexical fallback is merged only for missing spans, never used to
    # replace parser-owned ranges.
    fallback_spans = _regex_function_spans(source)
    known = {(span.name, span.start_line, span.end_line) for span in spans}
    for span in fallback_spans:
        key = (span.name, span.start_line, span.end_line)
        if key not in known:
            spans.append(span)
            known.add(key)
    if not spans:
        spans = fallback_spans
    return sorted(spans, key=lambda item: (item.start_line, item.end_line, item.name.casefold()))


@lru_cache(maxsize=64)
def _cached_function_spans(source: str) -> tuple[FunctionSpan, ...]:
    return tuple(_parse_function_spans_uncached(source or ""))


def parse_function_spans(source: str) -> list[FunctionSpan]:
    """Return a fresh list of cached source-local function spans."""

    return list(_cached_function_spans(source or ""))


def build_line_to_function_map(source: str, spans: Iterable[FunctionSpan] | None = None) -> dict[int, tuple[FunctionSpan, ...]]:
    spans = list(spans or parse_function_spans(source))
    mapping: dict[int, list[FunctionSpan]] = {}
    for span in spans:
        for line in range(span.start_line, span.end_line + 1):
            mapping.setdefault(line, []).append(span)
    return {line: tuple(sorted(items, key=lambda item: (item.end_line - item.start_line, item.start_line))) for line, items in mapping.items()}


def function_at_line(source: str, line: int, spans: Iterable[FunctionSpan] | dict[int, tuple[FunctionSpan, ...]] | None = None) -> FunctionSpan | None:
    if line <= 0:
        return None
    mapping = spans if isinstance(spans, dict) else build_line_to_function_map(source, spans)
    values = mapping.get(line)
    if values:
        return values[0]
    return None


def _span_by_name(spans: Iterable[FunctionSpan], name: str) -> FunctionSpan | None:
    return _span_by_identity(spans, name)


def _span_by_identity(
    spans: Iterable[FunctionSpan],
    name: str,
    *,
    contract_name: str = "",
    start_line: int | None = None,
    end_line: int | None = None,
) -> FunctionSpan | None:
    """Resolve a function by source identity before falling back to its name."""

    wanted = str(name or "").strip().casefold()
    if not wanted:
        return None
    matches = [span for span in spans if span.name.casefold() == wanted]
    if contract_name:
        scoped = [
            span
            for span in matches
            if span.contract_name.casefold() == str(contract_name).strip().casefold()
        ]
        if scoped:
            matches = scoped
    if start_line is not None:
        exact = [span for span in matches if span.start_line == int(start_line)]
        if exact:
            matches = exact
        else:
            containing = [
                span
                for span in matches
                if span.start_line <= int(start_line) <= span.end_line
            ]
            if containing:
                matches = containing
    if end_line is not None:
        exact_end = [span for span in matches if span.end_line == int(end_line)]
        if exact_end:
            matches = exact_end
    return sorted(matches, key=lambda span: (span.start_line, span.end_line))[0] if matches else None


def _span_from_row(
    source: str,
    spans: Iterable[FunctionSpan],
    row: dict[str, Any],
    role: str,
    *,
    name: str = "",
    line: int | None = None,
    same_as: FunctionSpan | None = None,
) -> FunctionSpan | None:
    """Resolve a proposal role without collapsing same-named functions.

    Explicit source identity wins.  When identity is absent, the locator line
    remains authoritative for the call-site/entrypoint/sink role; name-only
    lookup is the final compatibility fallback for legacy rows.
    """

    prefix = str(role or "function").strip().casefold() or "function"
    contract_name = str(
        row.get(f"{prefix}_contract_name")
        or (row.get("contract_name") if prefix == "function" else "")
        or ""
    ).strip()
    start_value = row.get(f"{prefix}_start_line")
    end_value = row.get(f"{prefix}_end_line")
    try:
        start_line = int(start_value) if start_value is not None else None
    except (TypeError, ValueError):
        start_line = None
    try:
        end_line = int(end_value) if end_value is not None else None
    except (TypeError, ValueError):
        end_line = None
    wanted = str(name or row.get(f"{prefix}_function_name") or "").strip()
    has_explicit_identity = bool(
        contract_name or start_line is not None or end_line is not None
    )
    exact = _span_by_identity(
        spans,
        wanted,
        contract_name=contract_name,
        start_line=start_line,
        end_line=end_line,
    ) if wanted and has_explicit_identity else None
    if exact is not None:
        return exact

    line_value = line
    if line_value is None:
        line_value = row.get("line") if prefix == "function" else row.get(f"{prefix}_line")
    try:
        line_value = int(line_value) if line_value is not None else 0
    except (TypeError, ValueError):
        line_value = 0
    if line_value > 0:
        line_span = function_at_line(source, line_value, spans)
        if line_span is not None:
            return line_span

    if same_as is not None and wanted and same_as.name.casefold() == wanted.casefold():
        return same_as
    return _span_by_name(spans, wanted) if wanted else None


def _span_identity_payload(span: FunctionSpan | None, prefix: str) -> dict[str, Any]:
    """Serialize parser-owned identity for one proposal role."""

    if span is None:
        return {}
    key = str(prefix or "function").strip().casefold() or "function"
    return {
        f"{key}_contract_name": span.contract_name,
        f"{key}_start_line": span.start_line,
        f"{key}_end_line": span.end_line,
    }


def _line_span(source: str, line: int, spans: Iterable[FunctionSpan]) -> FunctionSpan | None:
    """Resolve a line only through the parser-owned line map."""

    return function_at_line(source, line, spans)


def _first_matching_lines(source: str, span: FunctionSpan, pattern: re.Pattern[str]) -> list[int]:
    lines = source.splitlines()
    start = max(span.start_line - 1, 0)
    end = min(span.end_line, len(lines))
    return [line_number for line_number, text in enumerate(lines[start:end], start=span.start_line) if pattern.search(text)]


def _time_locator_lines(source: str, span: FunctionSpan) -> list[int]:
    """Return temporal operation lines, preferring explicit time guards."""

    timestamp_lines = _first_matching_lines(source, span, _TIME_LOCATOR)
    condition_lines = _first_matching_lines(source, span, _TIME_CONDITION_LOCATOR)
    return sorted(set(timestamp_lines) | set(condition_lines))


def _function_effect_lines(source: str, span: FunctionSpan) -> tuple[list[int], list[int]]:
    """Return source-local external-call and state-write event lines."""

    direct = _direct_effects(source, span)
    call_lines = sorted({int(item["line"]) for item in direct if item.get("kind") == "external_call"})
    write_lines = sorted({int(item["line"]) for item in direct if item.get("kind") == "state_write"})
    return call_lines, write_lines


def _function_body(source: str, span: FunctionSpan) -> str:
    lines = source.splitlines()
    return "\n".join(lines[max(span.start_line - 1, 0) : min(span.end_line, len(lines))])


def _line_start_offsets(source: str) -> list[int]:
    offsets = [0]
    offsets.extend(match.end() for match in re.finditer("\n", source or ""))
    return offsets


def _function_body_offsets(source: str, span: FunctionSpan) -> tuple[int, int] | None:
    line_starts = _line_start_offsets(source)
    if span.start_line <= 0 or span.start_line > len(line_starts):
        return None
    start_offset = line_starts[span.start_line - 1]
    end_offset = (
        line_starts[span.end_line]
        if span.end_line < len(line_starts)
        else len(source)
    )
    opening = source.find("{", start_offset, max(start_offset, end_offset))
    if opening < 0:
        return None
    closing = _balanced_end(source, opening)
    return opening + 1, min(closing, len(source))


def _statement_spans(source: str, span: FunctionSpan) -> list[StatementSpan]:
    """Split one function body into semicolon-terminated source statements.

    The scanner keeps parentheses/brackets balanced, so a multiline callback
    or assignment remains one locator unit.  It is intentionally lexical at
    this boundary; semantic classification happens after the span is built.
    """

    bounds = _function_body_offsets(source, span)
    if bounds is None:
        return []
    body_start, body_end = bounds
    statements: list[StatementSpan] = []
    start = body_start
    paren_depth = 0
    bracket_depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False

    def append(end_offset: int) -> None:
        nonlocal start
        if end_offset <= start:
            start = end_offset
            return
        text = source[start:end_offset]
        if text.strip():
            statements.append(
                StatementSpan(
                    start_offset=start,
                    end_offset=end_offset,
                    start_line=_line_at(source, start),
                    end_line=_line_at(source, max(start, end_offset - 1)),
                    text=text,
                )
            )
        start = end_offset

    index = body_start
    while index < body_end:
        char = source[index]
        next_char = source[index + 1] if index + 1 < body_end else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 2
                continue
            index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "/" and next_char == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            index += 2
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == ";" and paren_depth == 0 and bracket_depth == 0:
            append(index + 1)
        elif char == "}" and paren_depth == 0 and bracket_depth == 0:
            append(index)
        index += 1
    append(body_end)
    return statements


def _statement_line_for_offset(statement: StatementSpan, offset: int, source: str) -> int:
    return _line_at(source, statement.start_offset + max(0, offset))


def _strip_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text or "")
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


@lru_cache(maxsize=128)
def _source_storage_identifiers(source: str) -> frozenset[str]:
    """Collect contract-scope storage names, excluding locals and structs."""

    spans = parse_function_spans(source)
    line_starts = _line_start_offsets(source)
    function_ranges = []
    for span in spans:
        if span.start_line <= 0 or span.start_line > len(line_starts):
            continue
        header_match = next(
            (
                match
                for match in _FUNCTION_HEADER.finditer(source or "")
                if _line_at(source, match.start()) == span.start_line
                and (match.group("function") or match.group("special") or "").casefold()
                == span.name.casefold()
            ),
            None,
        )
        start = header_match.start() if header_match is not None else line_starts[span.start_line - 1]
        end = line_starts[span.end_line] if span.end_line < len(line_starts) else len(source)
        function_ranges.append((start, end))
    declaration = re.compile(
        r"\b(?:mapping\s*\([^;{}]*\)|"
        r"(?:u?int\d*|address(?:\s+payable)?|bool|string|bytes\d*|"
        r"[A-Z_][A-Za-z0-9_]*)"
        r"(?:\s*\[[^\]]*\])?)"
        r"(?:\s+(?:public|private|internal|constant|immutable|override|virtual))*"
        r"\s+(?P<name>[A-Za-z_]\w*)\s*(?:=|;)",
        re.I,
    )
    struct_ranges = []
    for match in re.finditer(r"\bstruct\s+[A-Za-z_]\w*\s*\{", source or "", re.I):
        opening = source.find("{", match.start(), match.end())
        if opening >= 0:
            struct_ranges.append((opening, _balanced_end(source, opening) + 1))

    def inside(offset: int, ranges: list[tuple[int, int]]) -> bool:
        return any(start <= offset < end for start, end in ranges)

    def brace_depth(offset: int) -> int:
        depth = 0
        quote: str | None = None
        escaped = False
        line_comment = False
        block_comment = False
        index = 0
        while index < offset:
            char = source[index]
            next_char = source[index + 1] if index + 1 < offset else ""
            if line_comment:
                if char == "\n":
                    line_comment = False
            elif block_comment:
                if char == "*" and next_char == "/":
                    block_comment = False
                    index += 1
            elif quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char == "/" and next_char == "/":
                line_comment = True
                index += 1
            elif char == "/" and next_char == "*":
                block_comment = True
                index += 1
            elif char in {'"', "'"}:
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth = max(0, depth - 1)
            index += 1
        return depth

    names: set[str] = set()
    for match in declaration.finditer(source or ""):
        absolute_match = match.start()
        if inside(absolute_match, function_ranges) or inside(absolute_match, struct_ranges):
            continue
        line_start = source.rfind("\n", 0, absolute_match) + 1
        line_end = source.find("\n", absolute_match)
        if line_end < 0:
            line_end = len(source)
        if re.search(r"\busing\b", source[line_start:line_end], re.I):
            continue
        if brace_depth(absolute_match) != 1:
            continue
        names.add(match.group("name").casefold())
    return frozenset(names)


def _storage_reference_identifiers(source: str, span: FunctionSpan) -> frozenset[str]:
    """Find local ``storage`` aliases that point at contract state."""

    storage_names = _source_storage_identifiers(source)
    body = _function_body(source, span)
    names: set[str] = set()
    for match in re.finditer(
        r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w+)*\s+storage\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<rhs>[A-Za-z_]\w*)"
        r"(?:\s*\[[^\]]*\])?",
        body,
        re.I,
    ):
        if match.group("rhs").casefold() in storage_names or "[" in match.group(0):
            names.add(match.group("name").casefold())
    return frozenset(names)


def _mapping_storage_identifiers(source: str) -> frozenset[str]:
    names: set[str] = set()
    for line in source.splitlines():
        match = re.search(
            r"\bmapping\s*\([^;{}]*\)(?:\s+\w+)*\s+(?P<name>[A-Za-z_]\w*)\s*(?:=|;)",
            line,
            re.I,
        )
        if match:
            names.add(match.group("name").casefold())
    return frozenset(names)


def _persistent_state_write_lines(source: str, span: FunctionSpan) -> list[int]:
    """Return statement-start lines that mutate persistent source state."""

    return sorted(
        {
            int(item["line"])
            for item in _direct_effects(source, span)
            if item.get("kind") == "state_write"
            and isinstance(item.get("line"), int)
        }
    )


def _is_permissionless_workflow_state(
    source: str, write_lines: Iterable[int]
) -> bool:
    """Recognize protocol workflow bookkeeping that is intentionally public.

    Commitment/nonce/request state coordinates a caller's protocol flow; it is
    not administrative authority over a shared asset or configuration.  Keep
    this resource classification source-grounded so generic function names do
    not decide access-control admission.
    """

    lines = (source or "").splitlines()
    valid_lines = [line for line in write_lines if 1 <= line <= len(lines)]
    if not valid_lines:
        return False
    return all(_WORKFLOW_STATE_ROOT.search(lines[line - 1]) for line in valid_lines)


def _delegated_actor_resource_shape(
    source: str, span: FunctionSpan, root: FunctionSpan
) -> dict[str, Any] | None:
    """Find an explicit actor/subject liquidation path at a callable root.

    This is intentionally narrower than generic access-control heuristics.  A
    caller-controlled actor must be passed into a liquidation operation for a
    caller-controlled subject, and that subject must drive a position/safety
    resource path.  Wrapper calls that pass ``msg.sender`` do not match.
    """

    if root.name.casefold() != span.name.casefold():
        return None
    if root.is_constructor or root.state_mutability.casefold() in {"view", "pure", "constant"}:
        return None
    if root.visibility.casefold() not in {"", "public", "external"}:
        return None

    lines = source.splitlines()
    header = "\n".join(lines[max(span.start_line - 1, 0) : min(span.end_line, len(lines))])
    header = header.split("{", 1)[0]
    address_params = [
        match.group("name")
        for match in re.finditer(
            r"\baddress(?:\s+payable)?\s+(?P<name>[A-Za-z_]\w*)\b",
            header,
            re.I,
        )
    ]
    if len(address_params) < 2:
        return None
    body = _function_body(source, span)
    actor_names = {value.casefold() for value in address_params}
    liquidation = re.compile(
        r"\bliquidate\s*\(\s*(?P<actor>[A-Za-z_]\w*)\s*,\s*"
        r"(?P<subject>[A-Za-z_]\w*)\s*,",
        re.I,
    )
    call = next(
        (
            match
            for match in liquidation.finditer(body)
            if match.group("actor").casefold() in actor_names
            and match.group("subject").casefold() in actor_names
            and match.group("actor").casefold() != match.group("subject").casefold()
        ),
        None,
    )
    if call is None:
        return None
    actor = call.group("actor")
    subject = call.group("subject")
    subject_pattern = re.escape(subject)
    resource_patterns = (
        rf"\bpositions\s*\[\s*{subject_pattern}\s*\]",
        rf"\bisSafe(?:WithPrice)?\s*\(\s*{subject_pattern}\b",
        rf"\b(?:calculateLiquidateAmount|marginBalanceWithPrice|maintenanceMarginWithPrice)\s*\(\s*{subject_pattern}\b",
    )
    resource_lines = [
        span.start_line + offset
        for offset, line in enumerate(body.splitlines())
        if any(re.search(pattern, line, re.I) for pattern in resource_patterns)
    ]
    if not resource_lines:
        return None
    call_line = span.start_line + body[: call.start()].count("\n")
    return {
        "actor_name": actor,
        "subject_name": subject,
        "call_line": call_line,
        "resource_lines": sorted(set(resource_lines)),
    }


def _actor_identity_bound(body: str, actor: str) -> bool:
    """Recognize a direct caller-to-actor equality guard."""

    sender = r"(?:msg\s*\.\s*sender|_msgSender\s*\(\))"
    actor_pattern = re.escape(actor)
    equality = rf"(?:{sender}\s*==\s*{actor_pattern}|{actor_pattern}\s*==\s*{sender})"
    return bool(
        re.search(
            rf"\b(?:require|assert)\s*\([^;{{}}]*{equality}[^;{{}}]*\)",
            body,
            re.I | re.S,
        )
        or re.search(
            rf"\bif\s*\([^;{{}}]*{equality}[^;{{}}]*\)",
            body,
            re.I | re.S,
        )
    )


@lru_cache(maxsize=128)
def _address_identifiers(source: str) -> frozenset[str]:
    """Collect source-declared address identifiers for native transfer typing."""

    names = {
        match.group(1)
        for match in re.finditer(
            r"\baddress(?:\s+payable)?\s+([A-Za-z_]\w*)\b", source or "", re.I
        )
    }
    return frozenset(names)


def _is_native_value_transfer(source: str, text: str) -> bool:
    match = _NATIVE_TRANSFER_CALL.search(text or "")
    if match is None:
        return False
    receiver = re.sub(r"\s+", "", match.group("receiver")).casefold()
    if receiver in {"msg.sender", "_msgsender()"} or "(" in receiver:
        return True
    return receiver in {name.casefold() for name in _address_identifiers(source)}


def _call_has_dynamic_amount(text: str, method: str) -> bool:
    """Return whether an asset call transfers a non-constant amount."""

    match = re.search(
        rf"\b{re.escape(method)}\s*\((?P<args>[^;{{}}]*)\)",
        text or "",
        re.I | re.S,
    )
    if match is None:
        return False
    parts = [part.strip() for part in match.group("args").split(",")]
    if not parts:
        return False
    amount = parts[-1]
    if re.fullmatch(
        r"(?:0x[0-9a-f]+|\d+)(?:\s+(?:wei|ether|gwei|finney))?",
        amount,
        re.I,
    ):
        return False
    if re.search(r"\bmsg\s*\.\s*value\b", amount, re.I) and not re.search(
        r"\b(?:balance|reserve|amount|value|reward|share|collateral|supply)\w*\b",
        amount,
        re.I,
    ):
        return False
    return True


@lru_cache(maxsize=128)
def _modifier_definitions(source: str) -> dict[str, dict[str, Any]]:
    """Return source-local modifier bodies for semantic guard checks."""

    definitions: dict[str, dict[str, Any]] = {}
    header = re.compile(r"\bmodifier\s+(?P<name>[A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*\{", re.I)
    scan_source = _mask_comments_preserve_lines(source)
    for match in header.finditer(scan_source or ""):
        opening = scan_source.find("{", match.end() - 1)
        if opening < 0:
            continue
        closing = _balanced_end(scan_source, opening)
        definitions[match.group("name").casefold()] = {
            "body": scan_source[opening + 1 : closing],
            "start_line": _line_at(scan_source, match.start()),
            "end_line": _line_at(scan_source, closing),
        }
    return definitions


def _has_identity_condition(text: str) -> bool:
    """Recognize caller-identity checks without treating state lookups as auth."""

    if re.search(r"\b(?:hasRole|_checkRole|isAuthorized|isWhitelisted|isAllowed)\s*\(", text, re.I):
        return True
    sender = r"(?:msg\s*\.\s*sender|_msgSender\s*\(\)|tx\s*\.\s*origin)"
    subject = (
        r"(?:owner|admin|guardian|operator|governance|manager|role|permission|authorized|"
        r"whitelist|allowlist|members?|admins?|operators?|roles?|"
        r"(?:new|pending|proposed|next|candidate|queued)[A-Za-z_]\w*)"
        r"(?:\s*\[[^\]]+\])?"
    )
    for match in re.finditer(r"\b(?:require|assert|if)\s*\((?P<expr>[^;{}]*)\)", text, re.I | re.S):
        expr = match.group("expr")
        if not re.search(sender, expr, re.I):
            continue
        if re.search(rf"{sender}\s*(?:==|!=)\s*{subject}", expr, re.I) or re.search(
            rf"{subject}\s*(?:==|!=)\s*{sender}", expr, re.I
        ):
            return True
        if re.search(
            rf"\b(?:roles|whitelist|allowlist|authorized|permissions|members|admins|operators)(?:\s*\[[^\]]*\])*\s*\[\s*{sender}",
            expr,
            re.I,
        ) or re.search(
            rf"{sender}[^;{{}}]*(?:is\s*whitelisted|is\s*allowed|authorized|has\s*role|member)",
            expr,
            re.I,
        ):
            return True
    return False


def _modifier_is_identity(name: str, source: str) -> tuple[bool, bool, list[int]]:
    """Return (effective, classified_only, evidence) for one modifier."""

    key = str(name or "").split("(", 1)[0].strip().casefold()
    definitions = _modifier_definitions(source)
    definition = definitions.get(key)
    if definition is not None:
        body = str(definition.get("body") or "")
        if _has_identity_condition(body):
            return True, True, [int(definition["start_line"])]
        return False, bool(re.match(r"only", key, re.I)), [int(definition["start_line"])]
    if _AUTH_MODIFIER.search(key):
        return True, True, []
    return False, bool(re.match(r"only", key, re.I)), []


def _modifier_is_mutex(name: str, source: str) -> tuple[bool, list[int]]:
    """Resolve whether a modifier implements an actual lock, not just a name."""

    key = str(name or "").split("(", 1)[0].strip().casefold()
    definitions = _modifier_definitions(source)
    definition = definitions.get(key)
    if definition is None:
        if not _MUTEX_NAME.search(key):
            return False, []
        # A standard guard may be inherited or imported outside the blinded
        # source.  The modifier name plus that type-level signal is enough to
        # treat the callable as mutex-protected; a local modifier definition
        # still takes precedence and is checked structurally below.
        if re.search(
            r"\b(?:contract|abstract\s+contract)\s+[A-Za-z_]\w*\s+is[^\{;]*"
            r"\b(?:ReentrancyGuard|ReentrancyGuardUpgradeable)\b"
            r"|\bimport\s+[^;\n]*reentrancy[^;\n]*;",
            _mask_comments_preserve_lines(source),
            re.I,
        ):
            return True, []
        return True, []
    body = str(definition.get("body") or "")
    placeholder_match = re.search(r"(?<![A-Za-z0-9_])_(?![A-Za-z0-9_])", body)
    placeholder = placeholder_match.start() if placeholder_match is not None else -1
    guard = re.search(
        r"\b(?:require|assert|if)\s*\([^;{}]*" + _MUTEX_STATE.pattern + r"\b",
        body,
        re.I | re.S,
    )
    writes = list(
        re.finditer(
            r"\b(?:_status|_entered|entered|locked|mutex|reentrancy(?:guard)?state)\b\s*(?:=|\+=|-=)",
            body,
            re.I,
        )
    )
    effective = placeholder >= 0 and guard is not None and len(writes) >= 2 and any(
        match.start() < placeholder for match in writes
    ) and any(match.start() > placeholder for match in writes)
    return (effective, [int(definition["start_line"])] if effective else [])


def _valid_lines(values: Iterable[Any], source: str) -> list[int]:
    limit = len(source.splitlines())
    return sorted({int(value) for value in values if isinstance(value, int) and 1 <= value <= limit})


def authorization_summary(source: str, span: FunctionSpan) -> dict[str, Any]:
    body = _function_body(source, span)
    evidence: list[int] = []
    status = "none"
    classified_only = False
    for modifier in span.modifiers:
        effective, classified, modifier_evidence = _modifier_is_identity(modifier, source)
        classified_only = classified_only or classified
        if effective:
            status = "effective_identity"
            evidence.extend(modifier_evidence or [span.start_line])
            break
    if status == "none" and re.search(r"\btx\s*\.\s*origin\b", body, re.I):
        status = "tx_origin_identity"
        evidence.extend(span.start_line + offset for offset, line in enumerate(body.splitlines()) if re.search(r"tx\s*\.\s*origin", line, re.I))
    elif status == "none" and _has_identity_condition(body):
        status = "effective_identity"
        evidence.extend(span.start_line + offset for offset, line in enumerate(body.splitlines()) if _has_identity_condition(line))
    elif status == "none" and classified_only:
        status = "unclassified_only_modifier"
        evidence.extend(span.start_line for _ in span.modifiers if re.match(r"only", str(_), re.I))
    return {
        "status": status,
        "effective": status in {"effective_identity", "tx_origin_identity"},
        "evidence_lines": sorted(set(evidence)),
        "modifier_names": list(span.modifiers),
    }


SpanKey = tuple[str, str, int, int]


def _span_key(span: FunctionSpan) -> SpanKey:
    return (
        span.contract_name.casefold(),
        span.name.casefold(),
        span.start_line,
        span.end_line,
    )


def _contract_bases(source: str) -> dict[str, tuple[str, ...]]:
    masked = _mask_comments_preserve_lines(source)
    result: dict[str, tuple[str, ...]] = {}
    pattern = re.compile(
        r"^[ \t]*(?:abstract\s+)?(?:contract|interface|library)\s+"
        r"(?P<name>[A-Za-z_]\w*)(?P<header>[^\{;]*)\{",
        re.I | re.M,
    )
    for match in pattern.finditer(masked):
        inheritance = re.search(r"\bis\s+([^\{]+)$", match.group("header") or "", re.I)
        bases = ()
        if inheritance:
            bases = tuple(
                value.casefold()
                for value in re.findall(r"\b[A-Za-z_]\w*\b", inheritance.group(1))
                if value.casefold() not in {"is", "virtual", "override"}
            )
        result[match.group("name").casefold()] = bases
    return result


def _call_graph(
    source: str, spans: list[FunctionSpan]
) -> dict[SpanKey, list[tuple[int, SpanKey]]]:
    """Build a source-local call graph without crossing contract boundaries."""

    by_key = {_span_key(span): span for span in spans}
    by_scope_and_name: dict[tuple[str, str], list[FunctionSpan]] = {}
    by_name: dict[str, list[FunctionSpan]] = {}
    for span in spans:
        scope = span.contract_name.casefold()
        by_scope_and_name.setdefault((scope, span.name.casefold()), []).append(span)
        by_name.setdefault(span.name.casefold(), []).append(span)
    bases = _contract_bases(source)
    graph: dict[SpanKey, list[tuple[int, SpanKey]]] = {
        _span_key(span): [] for span in spans
    }
    super_call = re.compile(r"\bsuper\s*\.\s*(?P<name>[A-Za-z_]\w*)\s*\(", re.I)
    scan_source = _mask_comments_preserve_lines(source)
    for span in spans:
        current_key = _span_key(span)
        body = _function_body(scan_source, span)
        local_scope = span.contract_name.casefold()
        for match in _BARE_CALL.finditer(body):
            callee = match.group("name").casefold()
            # Event emission is syntactically call-shaped, but it is not an
            # internal control-flow edge.  Treating ``emit Transfer(...)`` as
            # a call to a same-named function can make a constructor reach
            # every transfer helper in the contract and poison reentrancy
            # localization with unrelated state writes.
            prefix = body[max(0, match.start() - 24) : match.start()]
            if re.search(
                r"\b(?:emit|require|assert|revert|new|super)\s*$",
                prefix,
                re.I,
            ):
                continue
            if callee == span.name.casefold():
                continue
            targets = list(by_scope_and_name.get((local_scope, callee), ()))
            if not targets and not local_scope:
                targets = by_name.get(callee, [])
            if not targets:
                continue
            line = span.start_line + body[: match.start()].count("\n")
            for target in targets:
                target_key = _span_key(target)
                if target_key != current_key:
                    graph[current_key].append((line, target_key))

        for match in super_call.finditer(body):
            callee = match.group("name").casefold()
            targets: list[FunctionSpan] = []
            for base in bases.get(local_scope, ()):
                targets.extend(by_scope_and_name.get((base, callee), ()))
            if not targets:
                continue
            line = span.start_line + body[: match.start()].count("\n")
            for target in targets:
                graph[current_key].append((line, _span_key(target)))
    return graph


def _assignment_roots(text: str) -> list[str]:
    roots: list[str] = []
    assignment = re.compile(
        r"(?P<lhs>\([^;{}]*\)|[A-Za-z_]\w*(?:\s*\[[^\]]*\])?"
        r"(?:\s*\.\s*[A-Za-z_]\w+)?)\s*"
        r"(?:\+=|-=|\*=|/=|=(?!=)|\+\+|--)",
        re.I | re.S,
    )
    for match in assignment.finditer(text or ""):
        lhs = match.group("lhs")
        if lhs.startswith("("):
            roots.extend(re.findall(r"\b[A-Za-z_]\w*\b", lhs))
        else:
            root = re.match(r"\s*([A-Za-z_]\w*)", lhs)
            if root:
                roots.append(root.group(1))
    for match in re.finditer(r"\bdelete\s+([A-Za-z_]\w*)", text or "", re.I):
        roots.append(match.group(1))
    return roots


def _receiver_root(receiver: str) -> str:
    match = re.match(r"\s*([A-Za-z_]\w*)", receiver or "")
    return match.group(1).casefold() if match else ""


@lru_cache(maxsize=128)
def _source_library_methods(source: str) -> frozenset[tuple[str, str]]:
    """Return source-declared library receiver/function pairs.

    A qualified call such as ``UniswapV2Library.pairFor(...)`` is a call to a
    fixed source-local library, not an arbitrary contract receiver.  Keep the
    lookup source-scoped so imported or user-controlled receivers retain the
    existing external-call rules.
    """

    masked = _mask_comments_preserve_lines(source or "")
    library_names = {
        match.group("name").casefold()
        for match in re.finditer(
            r"^[ \t]*(?:abstract\s+)?library\s+(?P<name>[A-Za-z_]\w*)[^\{;]*\{",
            masked,
            re.I | re.M,
        )
    }
    if not library_names:
        return frozenset()
    return frozenset(
        (span.contract_name.casefold(), span.name.casefold())
        for span in _cached_function_spans(source or "")
        if span.contract_name.casefold() in library_names
    )


def _is_source_library_call(source: str, receiver: str, method: str) -> bool:
    root = _receiver_root(receiver)
    return bool(
        root
        and (root, str(method or "").casefold()) in _source_library_methods(source)
    )


def _is_library_receiver(source: str, span: FunctionSpan, receiver: str, method: str) -> bool:
    method_key = method.casefold()
    if method_key not in _LIBRARY_STATE_METHOD:
        return False
    if "(" in receiver:
        return False
    root = _receiver_root(receiver)
    if not root:
        return False
    if "[" in receiver:
        return root in _source_storage_identifiers(source)
    if root in _storage_reference_identifiers(source, span):
        return True
    return root in _mapping_storage_identifiers(source)


def _is_probable_external_receiver(
    source: str,
    span: FunctionSpan,
    receiver: str,
    method: str,
    storage_names: frozenset[str],
    storage_refs: frozenset[str],
) -> bool:
    """Recognize external receiver shapes omitted by the method allowlist."""

    method_key = method.casefold()
    if method_key in _LOCAL_LIBRARY_METHOD:
        return False
    if "(" in receiver:
        return True
    root = _receiver_root(receiver)
    if not root:
        return False
    if root in {value.casefold() for value in storage_names | storage_refs}:
        return True
    if re.match(
        r"(?:router|pool|pair|token|vault|strategy|oracle|factory|manager|"
        r"controller|registry|market|bridge|l1|l2|weth|uni|bsc)",
        root,
        re.I,
    ):
        return True
    header = _function_body(source, span)
    # Interface-like parameters are external receivers even when the method
    # name is project-specific (for example `callFunction` or `executeOrder`).
    return bool(
        re.search(
            rf"\b(?:I[A-Z][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*(?:Interface|Router|Pool|Manager|Token|Vault))\s+{re.escape(root)}\b",
            header,
        )
    )


def _is_callback_call(receiver: str, method: str) -> bool:
    if _CALLBACK_METHOD.search(method or ""):
        return True
    receiver_key = re.sub(r"\s+", "", receiver or "").casefold()
    return "(" in receiver_key and bool(
        re.search(r"(?:hook|callback|receiver|recipient|handler|plugin)", receiver_key, re.I)
    )


@lru_cache(maxsize=256)
def _source_local_low_level_wrapper(source: str, method: str) -> bool:
    """Prove that a source-defined token wrapper reaches a low-level call."""

    method_key = str(method or "").casefold()
    if method_key not in {
        "safetransfer",
        "safetransferfrom",
        "safeapprove",
        "safeincreaseallowance",
        "safedecreaseallowance",
    }:
        return False
    masked = _mask_comments_preserve_lines(source)
    for match in re.finditer(r"\blibrary\s+SafeERC20\b[^\{]*\{", masked, re.I):
        opening = masked.find("{", match.start(), match.end())
        if opening < 0:
            continue
        body = masked[opening + 1 : _balanced_end(masked, opening)]
        if not re.search(
            rf"\bfunction\s+{re.escape(method_key)}\b[\s\S]*?\b(?:callOptionalReturn|_callOptionalReturn)\s*\(",
            body,
            re.I,
        ):
            continue
        if re.search(
            r"\bfunction\s+(?:callOptionalReturn|_callOptionalReturn)\b[\s\S]*?\.\s*(?:call|delegatecall)\s*(?:\{|\(|;)",
            body,
            re.I,
        ):
            return True
    return False


def _direct_effects(source: str, span: FunctionSpan) -> list[dict[str, Any]]:
    storage_names = _source_storage_identifiers(source)
    storage_refs = _storage_reference_identifiers(source, span)
    local_function_names = {
        item.name.casefold() for item in _cached_function_spans(source)
    }
    effects: list[dict[str, Any]] = []
    for statement in _statement_spans(source, span):
        text = statement.text
        scan_text = _mask_comments_preserve_lines(text)
        if not re.search(r"[;()]", scan_text):
            continue
        for match in _MEMBER_CALL.finditer(scan_text):
            receiver = match.group("receiver")
            method = match.group("method")
            method_key = method.casefold()
            line = _statement_line_for_offset(statement, match.start(), source)
            if _READ_ONLY_METHOD.match(method_key) or method_key == "staticcall":
                continue
            if _is_source_library_call(source, receiver, method):
                continue
            if _is_library_receiver(source, span, receiver, method):
                effects.append(
                    {
                        "kind": "state_write",
                        "line": line,
                        "end_line": statement.end_line,
                        "text": text.strip(),
                        "method": method,
                        "receiver": receiver.strip(),
                        "library_call": True,
                    }
                )
                continue
            low_level = method_key in {"call", "send", "delegatecall"}
            asset_call = method_key in _ASSET_SINK_METHOD
            economic_dispatch = method_key in _ECONOMIC_DISPATCH_METHOD
            probable_external = _is_probable_external_receiver(
                source,
                span,
                receiver,
                method,
                storage_names,
                storage_refs,
            )
            safe_erc20_wrapper = method_key in {
                "safetransfer",
                "safetransferfrom",
                "safeapprove",
                "safeincreaseallowance",
                "safedecreaseallowance",
            }
            dynamic_asset_amount = _call_has_dynamic_amount(text, method)
            source_wrapper_callback = (
                safe_erc20_wrapper
                and _source_local_low_level_wrapper(source, method)
            )
            callback = (
                (_is_callback_call(receiver, method) and not economic_dispatch)
                or low_level
                or (asset_call and not safe_erc20_wrapper and not economic_dispatch)
                or (safe_erc20_wrapper and dynamic_asset_amount)
                or source_wrapper_callback
                or (
                    probable_external
                    and method_key not in {"approve", "safeapprove"}
                    and not safe_erc20_wrapper
                    and not economic_dispatch
                )
            )
            if not (
                low_level
                or asset_call
                or callback
                or probable_external
                or _EXTERNAL_CALL.search(scan_text)
            ):
                continue
            effects.append(
                {
                    "kind": "external_call",
                    "line": line,
                    "end_line": statement.end_line,
                    "text": text.strip(),
                    "method": method,
                    "receiver": receiver.strip(),
                "asset_call": asset_call or economic_dispatch,
                "economic_dispatch": economic_dispatch,
                    "callback_capable": callback,
                    "source_wrapper_callback": source_wrapper_callback,
                    "low_level": low_level,
                    "dynamic_asset_amount": dynamic_asset_amount,
                }
            )

        existing_bare_effects = {
            (int(item.get("line") or 0), str(item.get("method") or "").casefold())
            for item in effects
        }
        for bare in _BARE_EXTERNAL_EFFECT_METHOD.finditer(scan_text):
            method = bare.group("method")
            method_key = method.casefold()
            line = _statement_line_for_offset(statement, bare.start(), source)
            key = (line, method_key)
            if key in existing_bare_effects:
                continue
            dynamic_asset_amount = _call_has_dynamic_amount(text, method)
            source_wrapper_callback = _source_local_low_level_wrapper(source, method)
            effects.append(
                {
                    "kind": "external_call",
                    "line": line,
                    "end_line": statement.end_line,
                    "text": text.strip(),
                    "method": method,
                    "receiver": "",
                    "asset_call": True,
                    "callback_capable": dynamic_asset_amount or source_wrapper_callback,
                    "source_wrapper_callback": source_wrapper_callback,
                    "low_level": False,
                    "dynamic_asset_amount": dynamic_asset_amount,
                }
            )

        # Calls into inherited state-mutating functions are not visible in a
        # blinded source file.  Preserve their state transition as an event so
        # a callback followed by an inherited write still forms a CEI proof.
        for bare in re.finditer(
            r"(?<![.\w])(?P<method>[A-Za-z_]\w*)\s*\(",
            scan_text,
            re.I,
        ):
            method = bare.group("method")
            method_key = method.casefold()
            if method_key in local_function_names:
                continue
            if re.search(r"\bemit\b", scan_text[: bare.start()], re.I):
                continue
            if not (
                _INHERITED_MUTATING_METHOD.match(method)
                or re.match(
                    r"(?:sync|update|set|record|write|adjust|refresh|checkpoint|apply)[A-Za-z_]\w*",
                    method,
                    re.I,
                )
            ):
                continue
            line = _statement_line_for_offset(statement, bare.start(), source)
            effects.append(
                {
                    "kind": "state_write",
                    "line": line,
                    "end_line": statement.end_line,
                    "text": text.strip(),
                    "cross_function_state_write": True,
                    "inherited_effect": True,
                    "method": method,
                }
            )

        for inherited in re.finditer(
            r"\bsuper\s*\.\s*(?P<method>[A-Za-z_]\w*)\s*\(",
            scan_text,
            re.I,
        ):
            method = inherited.group("method")
            if not _INHERITED_MUTATING_METHOD.match(method):
                continue
            line = _statement_line_for_offset(statement, inherited.start(), source)
            effects.append(
                {
                    "kind": "state_write",
                    "line": line,
                    "end_line": statement.end_line,
                    "text": text.strip(),
                    "cross_function_state_write": True,
                    "inherited_effect": True,
                    "method": f"super.{method}",
                }
            )

        persistent_roots = {value.casefold() for value in _assignment_roots(scan_text)}
        persistent_roots &= storage_names | storage_refs
        if persistent_roots and not re.search(r"\b(?:require|assert|return|emit)\b", scan_text, re.I):
            first_root = next(
                (
                    match
                    for match in re.finditer(r"\b[A-Za-z_]\w*\b", scan_text)
                    if match.group(0).casefold() in persistent_roots
                ),
                None,
            )
            line = _statement_line_for_offset(
                statement,
                first_root.start() if first_root is not None else 0,
                source,
            )
            effects.append(
                {
                    "kind": "state_write",
                    "line": line,
                    "end_line": statement.end_line,
                    "text": text.strip(),
                    "persistent_roots": sorted(persistent_roots),
                }
            )
    return effects


_ARITHMETIC_STATEMENT = re.compile(
    r"(?<![=!<>+\-*/])(?:\*\*|[+\-*/%])(?![=+\-*/])|"
    r"(?:\+\+|--|\+=|-=|\*=|/=)|"
    r"\.\s*(?:add|sub|mul|div|mod)\s*\(",
    re.I,
)
_ARITHMETIC_ASSIGNMENT = re.compile(
    r"(?:^|[;{}])\s*(?:[A-Za-z_]\w*(?:\s+storage)?\s+)?"
    r"(?P<lhs>[A-Za-z_]\w*(?:\s*\[[^\]\n]+\])?(?:\s*\.\s*[A-Za-z_]\w+)?)\s*"
    r"(?P<op>=|\+=|-=|\*=|/=|\+\+|--)",
    re.I,
)


def _solidity_pragma_version(source: str) -> tuple[int, int] | None:
    match = re.search(r"\bpragma\s+solidity\s+[^0-9]*(\d+)\.(\d+)", source or "", re.I)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _statement_for_line(
    statements: list[StatementSpan], line: int
) -> StatementSpan | None:
    candidates = [
        statement
        for statement in statements
        if statement.start_line <= int(line) <= statement.end_line
    ]
    return min(
        candidates,
        key=lambda statement: (
            statement.end_line - statement.start_line,
            statement.start_line,
        ),
    ) if candidates else None


def _statement_has_arithmetic(statement: StatementSpan) -> bool:
    return bool(_ARITHMETIC_STATEMENT.search(_strip_comments(statement.text)))


def _statement_content_start_line(source: str, statement: StatementSpan) -> int:
    masked = _mask_comments_preserve_lines(statement.text)
    first = re.search(r"\S", masked)
    if first is None:
        return statement.start_line
    return _line_at(source, statement.start_offset + first.start())


def _statement_base_name(statement: StatementSpan) -> str:
    text = _strip_comments(statement.text)
    match = _ARITHMETIC_ASSIGNMENT.search(text)
    if not match:
        return ""
    return re.sub(r"\s+", "", match.group("lhs")).split("[", 1)[0].split(".", 1)[0].casefold()


def _statement_is_protected_arithmetic(
    source: str,
    statements: list[StatementSpan],
    index: int,
) -> bool:
    """Reject source arithmetic whose local statement path already proves safety."""

    statement = statements[index]
    text = _strip_comments(statement.text)
    if re.search(r"\b(?:SafeMath|CheckedMath)\b", text, re.I):
        return True
    if re.search(r"\.\s*(?:add|sub|mul|div|mod)\s*\(", text, re.I) and re.search(
        r"\b(?:overflow|underflow|division\s+by\s+zero|zero)\b",
        text,
        re.I,
    ):
        return True

    nearby = "\n".join(
        _strip_comments(item.text)
        for item in statements[max(0, index - 2) : min(len(statements), index + 3)]
    )
    if re.search(r"\b(?:require|assert)\s*\(", nearby, re.I) and re.search(
        r"\b(?:overflow|underflow|division\s+by\s+zero)\b|"
        r"(?:>=|<=|==|!=)",
        nearby,
        re.I,
    ):
        lhs = _statement_base_name(statement)
        if lhs and re.search(rf"\b{re.escape(lhs)}\b", nearby):
            return True
        if re.search(r"\b(?:overflow|underflow|division\s+by\s+zero)\b", nearby, re.I):
            return True
    return False


def _arithmetic_context_is_unchecked(
    source: str, span: FunctionSpan, statement: StatementSpan
) -> bool:
    version = _solidity_pragma_version(source)
    if version is None or version < (0, 8):
        return True
    body = _function_body(source, span)
    if re.search(r"\bunchecked\s*\{", body, re.I):
        return True
    # Yul arithmetic is not protected by Solidity 0.8 checked arithmetic.
    return bool(
        re.search(r"\bassembly\s*\{", body, re.I)
        and re.search(r"\b(?:add|sub|mul|div|sdiv|mod|smod|addmod|mulmod|shl|shr|sar)\s*\(", statement.text, re.I)
    )


def _span_line_set(statements: list[StatementSpan], lines: Iterable[Any]) -> set[int]:
    anchors: set[int] = set()
    for value in lines:
        if not isinstance(value, int) or value <= 0:
            continue
        statement = _statement_for_line(statements, value)
        if statement is None:
            anchors.add(value)
            continue
        anchors.update(range(statement.start_line, statement.end_line + 1))
    return anchors


def _arithmetic_provenance_proposals(
    source: str,
    spans: list[FunctionSpan],
    provenance_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn feature-fusion arithmetic provenance into statement-grounded candidate rows.

    Provenance is only a seed.  The operation and impact anchors are rebuilt
    from this module's balanced statement spans so a declaration line or a
    nearby comment cannot become the primary locator.
    """

    rows: list[dict[str, Any]] = []
    for provenance in provenance_rows:
        if not isinstance(provenance, dict):
            continue
        if provenance.get("source_grounded_arithmetic") is not True:
            continue
        name = str(provenance.get("function_name") or "").strip()
        if not name:
            continue
        span = _span_by_identity(
            spans,
            name,
            contract_name=str((provenance.get("function_span") or {}).get("contract_name") or "") or None,
            start_line=(provenance.get("function_span") or {}).get("start_line"),
            end_line=(provenance.get("function_span") or {}).get("end_line"),
        )
        if span is None:
            span = _span_by_name(spans, name)
        if span is None:
            continue

        statements = _statement_spans(source, span)
        if not statements:
            continue
        seed_lines = [
            int(value)
            for value in provenance.get("arithmetic_operation_lines", []) or []
            if isinstance(value, int) and value > 0
        ]
        if not seed_lines:
            continue
        raw_state_lines = [
            int(value)
            for value in provenance.get("state_write_lines", []) or []
            if isinstance(value, int) and value > 0
        ]
        raw_value_lines = [
            int(value)
            for value in provenance.get("value_relevance_lines", []) or []
            if isinstance(value, int) and value > 0
        ]
        direct = _direct_effects(source, span)
        direct_state_lines = [
            int(item["line"])
            for item in direct
            if item.get("kind") == "state_write" and isinstance(item.get("line"), int)
        ]
        direct_asset_lines = [
            int(item["line"])
            for item in direct
            if item.get("kind") == "external_call"
            and (item.get("asset_call") or item.get("callback_capable"))
            and isinstance(item.get("line"), int)
        ]
        impact_seed_lines = [*raw_state_lines, *raw_value_lines, *direct_state_lines, *direct_asset_lines]
        impact_statements = [
            statement
            for statement in statements
            if statement.start_line in _span_line_set(statements, impact_seed_lines)
            or re.search(r"\breturn\b", _strip_comments(statement.text), re.I)
            or re.search(r"\b(?:transfer|send|deposit|withdraw|swap|mint|burn|settle|execute)\s*\(", _strip_comments(statement.text), re.I)
        ]
        impact_text = [_strip_comments(statement.text) for statement in impact_statements]
        selected: list[tuple[int, StatementSpan, list[int]]] = []
        for index, statement in enumerate(statements):
            if not _statement_has_arithmetic(statement):
                continue
            if not _arithmetic_context_is_unchecked(source, span, statement):
                continue
            if _statement_is_protected_arithmetic(source, statements, index):
                continue
            base = _statement_base_name(statement)
            seeded = any(
                statement.start_line <= line <= statement.end_line
                for line in seed_lines
            )
            downstream: list[StatementSpan] = []
            if base:
                downstream = [
                    target
                    for target in impact_statements
                    if target.start_line >= statement.start_line
                    and re.search(rf"\b{re.escape(base)}\b", _strip_comments(target.text), re.I)
                ]
            direct_impact = any(
                target.start_line == statement.start_line
                for target in impact_statements
            )
            if not (seeded or downstream or direct_impact):
                continue
            impact_targets = downstream + (impact_statements if direct_impact else [])
            impact_lines = sorted(
                {
                    line
                    for target in impact_targets
                    for line in range(target.start_line, target.end_line + 1)
                }
            )
            selected.append(
                (_statement_content_start_line(source, statement), statement, impact_lines)
            )

        if not selected:
            continue
        for operation_line, statement, impact_lines in selected:
            operation_span_lines = list(range(statement.start_line, statement.end_line + 1))
            evidence_lines = sorted(
                set(operation_span_lines)
                | set(impact_lines)
                | _span_line_set(statements, provenance.get("evidence_lines", []) or [])
            )
            proof = str(provenance.get("source_arithmetic_proof") or "").strip()
            rows.append(
                {
                    "risk_type": "arithmetic",
                    "source_evidence_kind": "arithmetic_ast_statement_provenance",
                    "source_grounded": True,
                    "source_grounded_arithmetic": True,
                    "function_name": span.name,
                    "line": operation_line,
                    "arithmetic_operation_lines": [operation_line],
                    "state_write_lines": sorted(set(raw_state_lines + direct_state_lines)),
                    "value_relevance_lines": sorted(set(raw_value_lines)),
                    "evidence_lines": evidence_lines,
                    "source_arithmetic_proof": proof,
                    "reason": (
                        f"{span.name}() contains unchecked arithmetic at statement span "
                        f"L{statement.start_line}-L{statement.end_line} whose result reaches "
                        f"a source-local state, return, or asset-impact span"
                    ),
                }
            )
    return rows


def _helper_arithmetic_proposals(
    source: str, spans: list[FunctionSpan]
) -> list[dict[str, Any]]:
    """Recover arithmetic-helper loci whose value controls an order flow.

    Helper names such as ``safePower`` do not contain a raw operator, so the
    ordinary arithmetic provenance pass cannot see them.  This rule requires
    an assigned helper result, a non-literal operand, a later source-local use
    in a guard/update/return, and an economic or asset operation in the same
    function.  Checked ``.add/.sub/...`` calls remain outside this channel.
    """

    rows: list[dict[str, Any]] = []
    for span in spans:
        statements = _statement_spans(source, span)
        if not statements or not _arithmetic_context_is_unchecked(source, span, statements[0]):
            continue
        body = _function_body(source, span)
        if not (
            _FINANCIAL_FLOW_TOKEN.search(body)
            or _FINANCIAL_ORDER_OPERATION.search(body)
        ):
            continue
        direct = _direct_effects(source, span)
        asset_lines = sorted(
            int(item["line"])
            for item in direct
            if item.get("kind") == "external_call"
            and (item.get("asset_call") or item.get("economic_dispatch"))
            and isinstance(item.get("line"), int)
        )
        if not asset_lines:
            continue
        for index, statement in enumerate(statements):
            text = _strip_comments(statement.text)
            helper = _ARITHMETIC_HELPER_CALL.search(text)
            if helper is None:
                continue
            if _statement_is_protected_arithmetic(source, statements, index):
                continue
            lhs = _statement_base_name(statement)
            if not lhs:
                continue
            argument_text = text[helper.end() :]
            if not re.search(r"\b[A-Za-z_]\w*\b", argument_text):
                continue
            downstream = [
                target
                for target in statements[index + 1 :]
                if re.search(rf"\b{re.escape(lhs)}\b", _strip_comments(target.text), re.I)
                and re.search(
                    r"\b(?:require|assert|if|while|return|delete)\b|"
                    r"(?:=|\+=|-=|\*=|/=)|"
                    r"\b(?:exchange|order|swap|settle|execute|fill)\w*\s*\(",
                    _strip_comments(target.text),
                    re.I,
                )
            ]
            if not downstream:
                continue
            operation_line = _statement_content_start_line(source, statement)
            impact_lines = sorted(
                {
                    line
                    for target in downstream
                    for line in range(target.start_line, target.end_line + 1)
                }
            )
            rows.append(
                {
                    "risk_type": "arithmetic",
                    "source_evidence_kind": "helper_arithmetic_provenance",
                    "source_grounded": True,
                    "source_grounded_arithmetic": True,
                    "function_name": span.name,
                    "line": operation_line,
                    "arithmetic_operation_lines": [operation_line],
                    "value_relevance_lines": impact_lines,
                    "asset_sink_lines": asset_lines,
                    "evidence_lines": sorted(
                        set(
                            range(statement.start_line, statement.end_line + 1)
                        )
                        | set(impact_lines)
                        | set(asset_lines)
                    ),
                    "source_arithmetic_proof": "helper_arithmetic_to_order_control_or_asset_operation",
                    "reason": (
                        f"{span.name}() uses {helper.group('method')}() in a statement span "
                        "whose result reaches an order guard/update and an asset operation"
                    ),
                }
            )
    return rows


def _span_mutex_summary(source: str, span: FunctionSpan) -> tuple[bool, list[int]]:
    """Return mutex evidence for one function, including source-defined locks."""

    evidence: list[int] = []
    for modifier in span.modifiers:
        effective, modifier_evidence = _modifier_is_mutex(modifier, source)
        if effective:
            evidence.extend(modifier_evidence or [span.start_line])
            return True, evidence
    body = _function_body(source, span)
    inline_guard = re.search(
        r"\b(?:require|assert|if)\s*\([^;{}]*" + _MUTEX_STATE.pattern + r"\b",
        body,
        re.I | re.S,
    )
    inline_writes = list(
        re.finditer(
            r"\b(?:_status|_entered|entered|locked|mutex|reentrancy(?:guard)?state)\b\s*(?:=|\+=|-=)",
            body,
            re.I,
        )
    )
    if inline_guard is not None and len(inline_writes) >= 2:
        return True, [span.start_line]
    return False, []


def _state_write_closure_lines(
    source: str,
    start: FunctionSpan,
    spans: list[FunctionSpan],
    graph: dict[SpanKey, list[tuple[int, SpanKey]]],
    direct_effects_cache: dict[SpanKey, list[dict[str, Any]]],
) -> list[int]:
    """Collect state-transition lines reachable from one source-local function."""

    by_key = {_span_key(span): span for span in spans}
    memo: dict[SpanKey, list[int]] = {}
    visiting: set[SpanKey] = set()

    def visit(span: FunctionSpan) -> list[int]:
        key = _span_key(span)
        if key in memo:
            return memo[key]
        if key in visiting:
            return []
        visiting.add(key)
        effects = direct_effects_cache.setdefault(key, _direct_effects(source, span))
        lines = [
            int(item["line"])
            for item in effects
            if item.get("kind") == "state_write" and isinstance(item.get("line"), int)
        ]
        for _, target_key in graph.get(key, []):
            target = by_key.get(target_key)
            if target is not None:
                lines.extend(visit(target))
        visiting.remove(key)
        memo[key] = sorted(set(lines))
        return memo[key]

    return visit(start)


def reentrancy_effect_summary(
    source: str,
    root: FunctionSpan,
    spans: Iterable[FunctionSpan] | None = None,
    *,
    _graph: dict[SpanKey, list[tuple[int, SpanKey]]] | None = None,
    _direct_effects_cache: dict[SpanKey, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    all_spans = list(spans or parse_function_spans(source))
    by_key = {_span_key(span): span for span in all_spans}
    graph = _graph if _graph is not None else _call_graph(source, all_spans)
    direct_effects_cache = _direct_effects_cache if _direct_effects_cache is not None else {}
    visiting: set[str] = set()

    def expand(span: FunctionSpan) -> list[dict[str, Any]]:
        key = _span_key(span)
        if key in visiting:
            return []
        visiting.add(key)
        result: list[dict[str, Any]] = []
        if key not in direct_effects_cache:
            direct_effects_cache[key] = _direct_effects(source, span)
        direct = direct_effects_cache[key]
        span_mutex, _ = _span_mutex_summary(source, span)
        effects_by_line: dict[int, list[dict[str, Any]]] = {}
        for effect in direct:
            effects_by_line.setdefault(int(effect["line"]), []).append(
                {
                    **effect,
                    "function": span.name,
                    "contract_name": span.contract_name,
                    "function_start_line": span.start_line,
                    "mutex": span_mutex,
                }
            )
        calls_by_line: dict[int, list[str]] = {}
        for line, callee in graph.get(key, []):
            calls_by_line.setdefault(line, []).append(callee)
        for line in range(span.start_line, span.end_line + 1):
            result.extend(effects_by_line.get(line, []))
            for callee in calls_by_line.get(line, []):
                target = by_key.get(callee)
                if target is not None:
                    result.extend(expand(target))
        visiting.remove(key)
        return result

    events = [
        item
        for item in expand(root)
        if not (
            item["kind"] == "external_call"
            and _is_native_value_transfer(source, str(item.get("text") or ""))
            and not _call_has_dynamic_amount(str(item.get("text") or ""), str(item.get("method") or "transfer"))
        )
    ]
    first_call_index = next((index for index, item in enumerate(events) if item["kind"] == "external_call"), None)
    first_call = events[first_call_index] if first_call_index is not None else None
    later_write = next(
        (item for index, item in enumerate(events) if item["kind"] == "state_write" and first_call_index is not None and index > first_call_index),
        None,
    )
    mutex, mutex_evidence = _span_mutex_summary(source, root)
    # ``events`` already contains the root's reachable helper closure.  The
    # previous implementation scanned every public function in the contract,
    # which attached unrelated state writes to every callback path (and was
    # especially damaging for constructors and shared token helpers).
    cross_function_state_lines = sorted(
        {
            int(item["line"])
            for item in events
            if item.get("kind") == "state_write"
            and item.get("function", "").casefold() != root.name.casefold()
            and isinstance(item.get("line"), int)
        }
    )
    external_events = [item for item in events if item["kind"] == "external_call"]
    nested_mutex_evidence: list[int] = []
    if external_events and all(bool(item.get("mutex")) for item in external_events):
        mutex = True
        for item in external_events:
            event_span = next(
                (
                    candidate
                    for candidate in all_spans
                    if candidate.contract_name.casefold()
                    == str(item.get("contract_name") or root.contract_name).casefold()
                    and candidate.name.casefold()
                    == str(item.get("function") or "").casefold()
                    and candidate.start_line == item.get("function_start_line")
                ),
                None,
            )
            if event_span is None or event_span.name == root.name:
                continue
            _, evidence = _span_mutex_summary(source, event_span)
            nested_mutex_evidence.extend(evidence or [event_span.start_line])
    mutex_evidence.extend(nested_mutex_evidence)
    cei_complete = first_call is None or later_write is None
    callback_capable = any(
        item.get("kind") == "external_call" and item.get("callback_capable")
        for item in events
    )
    source_wrapper_callback = any(
        item.get("kind") == "external_call" and item.get("source_wrapper_callback")
        for item in events
    )
    has_state_write = any(item.get("kind") == "state_write" for item in events)
    return {
        "root_function": root.name,
        "events": events,
        "mutex": mutex,
        "cei_complete": cei_complete,
        "callback_capable": callback_capable,
        "source_wrapper_callback": source_wrapper_callback,
        "cross_function_state_write": bool(cross_function_state_lines),
        "cross_function_state_write_lines": cross_function_state_lines,
        "reentrant_order": bool(
            first_call
            and (has_state_write or cross_function_state_lines)
            and callback_capable
            and (later_write or source_wrapper_callback)
            and not mutex
        ),
        "protection_evidence": _valid_lines(mutex_evidence, source),
    }


def _extract_return_types(text: str) -> tuple[str, ...]:
    match = re.search(r"\breturns\s*\((?P<body>.*?)\)", text, re.I | re.S)
    if not match:
        return ()
    result: list[str] = []
    for item in match.group("body").split(","):
        tokens = item.strip().split()
        if tokens and tokens[0].casefold() not in {"void", ""}:
            result.append(tokens[0].casefold())
    return tuple(result)


def typed_return_signatures(source: str) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, str]]:
    """Resolve visible method returns by receiver/interface type."""

    source = _mask_comments_preserve_lines(source)
    type_methods: dict[str, dict[str, tuple[str, ...]]] = {}
    type_bases: dict[str, tuple[str, ...]] = {}
    # Solidity comments and string literals frequently contain prose such as
    # ``contract address`` or ``contract ...``.  Require declarations to start
    # on a source line so those prose fragments cannot become fake receiver
    # types and swallow the real contract body.
    for block in re.finditer(
        r"^[ \t]*(?:interface|contract|library)\s+"
        r"(?P<type>[A-Za-z_]\w*)(?P<header>[^\{]*)\{",
        source,
        re.I | re.M,
    ):
        end = _balanced_end(source, source.find("{", block.end() - 1))
        body = source[block.end() : end]
        methods: dict[str, tuple[str, ...]] = {}
        for method in re.finditer(
            r"\bfunction\s+(?P<name>[A-Za-z_]\w*)\s*\(.*?\)(?P<tail>.*?)(?:;|\{)",
            body,
            re.I | re.S,
        ):
            methods[method.group("name").casefold()] = _extract_return_types(method.group(0))
        type_name = block.group("type").casefold()
        known = {
            method.casefold(): returns
            for method, returns in _KNOWN_INTERFACE_METHODS.get(type_name, {}).items()
        }
        type_methods[type_name] = {**known, **methods}
        bases = re.search(r"\bis\s+([^\{]+)$", block.group("header") or "", re.I)
        type_bases[type_name] = (
            tuple(
                item.casefold()
                for item in re.findall(r"\b[A-Za-z_]\w*\b", bases.group(1))
                if item.casefold() not in {"is", "virtual", "override"}
            )
            if bases
            else ()
        )

    # Inherited interface methods remain part of a child receiver contract even
    # when the child source does not redeclare them. Resolve this transitively,
    # while keeping lookup keyed by the declared receiver type.
    def inherited_methods(type_name: str, seen: set[str] | None = None) -> dict[str, tuple[str, ...]]:
        seen = set(seen or ())
        if type_name in seen:
            return {}
        seen.add(type_name)
        merged: dict[str, tuple[str, ...]] = {}
        for base in type_bases.get(type_name, ()):
            merged.update(inherited_methods(base, seen))
            merged.update(type_methods.get(base, {}))
        merged.update(type_methods.get(type_name, {}))
        return merged

    for type_name in list(type_methods):
        type_methods[type_name] = inherited_methods(type_name)
    variable_types: dict[str, str] = {}
    # Parse declarations line-by-line.  The previous whitespace-spanning
    # expression treated comments, ``return true`` and assignment expressions
    # as declarations and could overwrite ``IERC20 stakedToken`` with
    # ``override`` or ``address``.  Anchoring at a line start keeps receiver
    # types tied to an actual declaration while still covering state and local
    # variables with multi-token modifiers such as ``public override``.
    declaration = re.compile(
        r"^[ \t]*(?P<type>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s+"
        r"(?P<modifiers>(?:(?:[A-Za-z_]\w*)\s+)*)"
        r"(?P<name>[A-Za-z_]\w*)\s*(?:=[^;]*)?;\s*(?://.*)?$",
        re.I,
    )
    for line in source.splitlines():
        match = declaration.match(line)
        if not match:
            continue
        type_name = match.group("type").casefold()
        if type_name in _DECLARATION_KEYWORDS:
            continue
        modifiers = [value.casefold() for value in match.group("modifiers").split()]
        if any(value not in _DECLARATION_MODIFIERS for value in modifiers):
            continue
        variable_types[match.group("name").casefold()] = type_name

    # Function parameters are visible receiver declarations too.  Their
    # parenthesized lists can span lines, so parse each function header with a
    # balanced scanner rather than applying the state-variable regex above.
    for header in _FUNCTION_HEADER.finditer(source):
        opening = source.find("(", header.end() - 1)
        if opening < 0:
            continue
        closing = _call_end_offset(source, opening)
        params = source[opening + 1 : closing]
        for item in params.split(","):
            tokens = re.findall(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?", item)
            if len(tokens) < 2:
                continue
            type_name, variable_name = tokens[0].casefold(), tokens[-1].casefold()
            if type_name in _DECLARATION_KEYWORDS or variable_name in _DECLARATION_MODIFIERS:
                continue
            variable_types[variable_name] = type_name

    for type_name, methods in _KNOWN_INTERFACE_METHODS.items():
        type_methods.setdefault(
            type_name,
            {method.casefold(): returns for method, returns in methods.items()},
        )
    return type_methods, variable_types


def _statement_context(lines: list[str], line_number: int) -> str:
    if line_number <= 0 or line_number > len(lines):
        return ""
    start = line_number - 1
    while start > 0:
        previous = lines[start - 1]
        if ";" in previous or "{" in previous or "}" in previous:
            break
        start -= 1
    end = line_number - 1
    depth = 0
    while end < len(lines):
        text = lines[end]
        depth += text.count("(") - text.count(")")
        if ";" in text and depth <= 0:
            break
        if "}" in text and depth <= 0:
            break
        end += 1
    return "\n".join(lines[start : min(end + 1, len(lines))])


def _call_end_offset(source: str, opening_offset: int) -> int:
    depth = 0
    for index in range(opening_offset, len(source)):
        char = source[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth <= 0:
                return index
    return len(source) - 1


def _receiver_type_from_prefix(prefix: str, receiver: str, variable_types: dict[str, str]) -> str | None:
    if receiver.casefold() == "super":
        return "__super__"
    receiver_type = variable_types.get(receiver.casefold())
    if receiver_type:
        return receiver_type
    cast = re.search(
        rf"\b(?P<type>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*\(\s*{re.escape(receiver)}\s*\)\s*$",
        prefix,
        re.I | re.S,
    )
    return cast.group("type").casefold() if cast else None


def _enclosing_contract_type(source: str, offset: int) -> str | None:
    """Resolve the source-local contract/library containing one call site."""

    selected: tuple[int, str] | None = None
    for match in re.finditer(
        r"\b(?:interface|contract|library)\s+(?P<type>[A-Za-z_]\w*)[^\{]*\{",
        source or "",
        re.I,
    ):
        opening = source.find("{", match.end() - 1)
        if opening < 0 or opening > offset:
            continue
        closing = _balanced_end(source, opening)
        if offset <= closing and (selected is None or opening > selected[0]):
            selected = (opening, match.group("type").casefold())
    return selected[1] if selected else None


def _return_is_consumed(context: str, receiver_offset: int) -> bool:
    prefix = context[:receiver_offset] if receiver_offset >= 0 else context
    if re.search(r"\btry\b", prefix, re.I):
        # A bare try/catch only handles revert control flow.  It consumes a
        # non-void return only when the try clause binds it with returns (...).
        return bool(re.search(r"\breturns\s*\(", context[receiver_offset:] if receiver_offset >= 0 else context, re.I))
    # A preceding, unrelated require/assert is not a consumer of this call's
    # return. Only a condition whose opening keyword belongs to the same
    # statement counts; multiline require/if/assert remains covered.
    if re.search(r"\b(?:require|assert|if|while)\s*\([^;{}]*$", prefix, re.I | re.S):
        return True
    if re.search(r"(?:^|[^=!<>])=\s*$|\breturn\s*$|\bemit\s*$", prefix, re.I):
        return True
    return False


def _native_value_transfer_candidates(
    source: str, spans: list[FunctionSpan]
) -> list[dict[str, Any]]:
    """Recover native refund sinks whose result/control outcome is discarded.

    Solidity's native ``transfer`` has no source-level return value.  In the
    E2 development rule it is nevertheless retained as an unchecked native
    value sink when the caller receives ``msg.value`` (or an unscoped amount),
    while claimant-scoped balance debits remain clean controls.
    """

    scan_source = _mask_comments_preserve_lines(source)
    lines = scan_source.splitlines()
    results: list[dict[str, Any]] = []
    for match in _NATIVE_VALUE_TRANSFER.finditer(scan_source or ""):
        line_number = _line_at(scan_source, match.start())
        span = function_at_line(source, line_number, spans)
        if span is None:
            continue
        opening = source.find("(", match.end() - 1)
        call_end = _call_end_offset(scan_source, opening if opening >= 0 else match.end())
        argument = source[opening + 1 : call_end].strip() if opening >= 0 else ""
        body_before = "\n".join(
            lines[max(span.start_line - 1, 0) : min(line_number, len(lines))]
        )
        # A claimant balance/reward debit establishes the intended ownership
        # path for the outgoing native transfer; it is not a generic unchecked
        # sink.  The same guard covers trove/stake/reward mappings.
        if _CALLER_SCOPED_RESOURCE_WRITE.search(body_before):
            continue
        if not argument and not re.search(r"\bmsg\s*\.\s*value\b", body_before, re.I):
            continue
        evidence_lines = list(range(line_number, _line_at(source, call_end) + 1))
        results.append(
            {
                "risk_type": "unchecked_low_level_calls",
                "source_evidence_kind": "native_value_transfer_discard",
                "source_grounded": True,
                "native_value_transfer": True,
                "receiver": match.group("recipient"),
                "receiver_type": "native",
                "method": "transfer",
                "return_types": [],
                "function_name": span.name,
                "line": line_number,
                "evidence_lines": evidence_lines,
                "reason": f"{span.name}() sends native value to msg.sender at @L{line_number} without a scoped claimant debit",
            }
        )
    return results


# Scan the complete source so multiline calls, local receiver declarations,
# try/catch, assignments, and typed non-bool returns are resolved without a
# global method-name fallback.
def typed_return_candidates(source: str, spans: Iterable[FunctionSpan] | None = None) -> list[dict[str, Any]]:
    scan_source = _mask_comments_preserve_lines(source)
    type_methods, variable_types = typed_return_signatures(scan_source)
    lines = scan_source.splitlines()
    resolved_spans = list(spans or parse_function_spans(source))
    storage_names = _source_storage_identifiers(source)
    results: list[dict[str, Any]] = []
    for match in _QUALIFIED_CALL.finditer(scan_source):
        receiver_expression = match.group("receiver")
        cast = re.fullmatch(
            r"(?P<type>[A-Za-z_]\w*)\s*\(\s*(?P<value>[A-Za-z_]\w*)\s*\)",
            receiver_expression,
            re.I,
        )
        receiver = cast.group("value") if cast else receiver_expression
        method = match.group("method")
        line_number = _line_at(source, match.start())
        statement = _statement_context(lines, line_number)
        # ``statement.find(receiver)`` can resolve to an earlier occurrence in
        # the same multiline statement (for example a preceding
        # ``require(token.transferFrom(...))``).  Anchor consumption checks to
        # the receiver/method pair for this exact qualified call instead.
        receiver_pattern = re.escape(receiver_expression)
        receiver_pattern = receiver_pattern.replace(r"\ ", r"\s*")
        call_anchor = re.compile(
            rf"(?<![.\w]){receiver_pattern}\s*\.\s*{re.escape(method)}\b",
            re.I | re.S,
        )
        call_matches = list(call_anchor.finditer(statement))
        receiver_offset = call_matches[-1].start() if call_matches else statement.rfind(receiver)
        header_prefix = statement[:receiver_offset] if receiver_offset >= 0 else statement
        if re.search(r"\bfunction\s+[A-Za-z_]\w*\s*\(", header_prefix, re.I) and "{" not in header_prefix:
            continue
        receiver_type = (
            cast.group("type").casefold()
            if cast
            else _receiver_type_from_prefix(scan_source[: match.start()], receiver, variable_types)
        )
        if receiver.casefold() == "super":
            receiver_type = _enclosing_contract_type(scan_source, match.start())
        returns = (type_methods.get(receiver_type, {}) if receiver_type else {}).get(method.casefold(), ())
        low_level_method = method.casefold() in {
            "call",
            "send",
            "delegatecall",
            "staticcall",
        }
        probable_external = _is_probable_external_receiver(
            source,
            span or FunctionSpan("", 0, 0),
            receiver,
            method,
            storage_names,
            frozenset(),
        ) if (span := function_at_line(source, line_number, resolved_spans)) else False
        known_external_return = method.casefold() in _KNOWN_EXTERNAL_RETURN_METHODS
        if not (
            returns
            or low_level_method
            or (known_external_return and (cast or receiver_type or probable_external))
        ) or re.search(
            r"\b(?:safeerc20|transferhelper)\b|\.safe(?:transfer|approve|transferfrom|increaseallowance|decreaseallowance)\s*\(",
            statement,
            re.I,
        ):
            continue
        if _return_is_consumed(statement, receiver_offset):
            continue
        opening = scan_source.find("(", match.end() - 1)
        call_end = _call_end_offset(scan_source, opening if opening >= 0 else match.end())
        span = function_at_line(source, line_number, resolved_spans)
        if span is None:
            continue
        evidence_lines = list(range(line_number, _line_at(source, call_end) + 1))
        results.append(
            {
                "risk_type": "unchecked_low_level_calls",
                "source_evidence_kind": "typed_nonvoid_return_discard",
                "source_grounded": True,
                "typed_return_discard": True,
                "receiver": receiver,
                "receiver_type": receiver_type,
                "method": method,
                "return_types": list(returns) or ["unknown"] if low_level_method or known_external_return else list(returns),
                "function_name": span.name,
                "line": line_number,
                "evidence_lines": evidence_lines,
                "typed_return_discard": bool(returns),
                "low_level_return_discard": low_level_method,
                "source_evidence_kind": (
                    "typed_nonvoid_return_discard"
                    if returns
                    else "unchecked_low_level_call_return_discard"
                    if low_level_method
                    else "external_return_discard"
                ),
                "reason": f"{span.name}() discards the external {method}() return at @L{line_number}",
            }
        )
    results.extend(_native_value_transfer_candidates(source, resolved_spans))
    return results


def stable_candidate_id(row: dict[str, Any], *, source_id: str = "", prefix: str = "SRC") -> str:
    payload = {
        "source_id": source_id,
        "risk_type": str(row.get("risk_type") or row.get("primary_category") or "").casefold(),
        "function_name": str(row.get("function_name") or ""),
        "line": row.get("line"),
        "evidence_lines": sorted({int(value) for value in row.get("evidence_lines", []) if isinstance(value, int) and value > 0}),
        "source_evidence_kind": str(row.get("source_evidence_kind") or ""),
        "entrypoint": str(row.get("entrypoint_function_name") or row.get("entrypoint") or ""),
        "sink": str(row.get("sink_function_name") or row.get("sink") or ""),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _clean_sentinel(row: dict[str, Any], source: str, spans: list[FunctionSpan]) -> tuple[bool, str, list[int]]:
    category = str(row.get("risk_type") or row.get("primary_category") or "").casefold()
    line = row.get("line") if isinstance(row.get("line"), int) else 0
    span = _span_from_row(
        source,
        spans,
        row,
        "function",
        name=str(row.get("function_name") or ""),
        line=line,
    )
    if span is None:
        return False, "no_resolved_function", []
    entrypoint_name = str(
        row.get("entrypoint_function_name") or row.get("entrypoint") or ""
    ).strip()
    entrypoint_lines = row.get("entrypoint_lines") or []
    entrypoint_line = entrypoint_lines[0] if entrypoint_lines else None
    entrypoint_span = _span_from_row(
        source,
        spans,
        row,
        "entrypoint",
        name=entrypoint_name,
        line=entrypoint_line,
        same_as=span,
    ) if entrypoint_name else None
    if category == "access_control" and row.get("source_evidence_kind") in {
        "source_access_surface",
        "inherited_effect_propagation",
    }:
        access_proof = row.get("access_semantic_proof")
        if not isinstance(access_proof, dict) or not str(access_proof.get("resource") or "").strip():
            return True, "access_resource_proof_missing", []
        if entrypoint_span is None or entrypoint_span.is_constructor:
            return True, "non_callable_or_constructor_entrypoint", []
        entrypoint_auth = authorization_summary(source, entrypoint_span)
        if entrypoint_auth["status"] != "none":
            return True, "authority_unknown_or_present", entrypoint_auth.get("evidence_lines", [])
        if access_proof.get("resource_kind") == "generic_state":
            return True, "non_privileged_shared_state", []
    # Authority is evaluated at the callable boundary. A protected public
    # entrypoint must also protect an internal helper reached from that entry;
    # checking only the helper would create a false access-control finding.
    auth_evidence: list[int] = []
    if category == "access_control":
        for auth_span in (entrypoint_span, span):
            if auth_span is None:
                continue
            auth = authorization_summary(source, auth_span)
            if auth["effective"]:
                auth_evidence.extend(auth["evidence_lines"] or [auth_span.start_line])
        if auth_evidence:
            return True, "effective_authorization_guard", sorted(set(auth_evidence))
    if category == "reentrancy":
        if span.is_constructor or (entrypoint_span is not None and entrypoint_span.is_constructor):
            return True, "constructor_not_callable_reentrancy", []
        # Reentrancy protection is likewise an entrypoint property when the
        # external effect occurs in a reachable helper.
        effects = reentrancy_effect_summary(source, entrypoint_span or span, spans)
        if effects["mutex"]:
            return True, "reentrancy_mutex", effects["protection_evidence"]
        if not effects.get("reentrant_order"):
            return True, "no_callback_state_order", []
        if effects["cei_complete"] and not effects.get("callback_capable"):
            return True, "complete_cei_or_no_callback_state_path", []
    if category == "unchecked_low_level_calls":
        context = _statement_context(source.splitlines(), line) if line > 0 else ""
        match = _QUALIFIED_CALL.search(context)
        receiver_offset = match.start("receiver") if match else -1
        if _return_is_consumed(context, receiver_offset):
            return True, "checked_or_captured_return", [line]
    return False, "no_clean_sentinel", []


def _callable_span(span: FunctionSpan) -> bool:
    return span.visibility.casefold() in {"", "public", "external"} or span.is_constructor or span.is_fallback or span.is_receive


def _reachable_entrypoints(
    spans: list[FunctionSpan],
    graph: dict[SpanKey, list[tuple[int, SpanKey]]],
) -> dict[SpanKey, tuple[FunctionSpan, ...]]:
    """Return every callable root that can reach each function.

    A helper may be shared by several public entrypoints.  Keeping only the
    first BFS root loses the caller/resource proof for later entrypoints and
    makes interprocedural localization depend on source declaration order.
    """

    by_key = {_span_key(span): span for span in spans}
    roots = [span for span in spans if _callable_span(span)]
    reachable: dict[SpanKey, list[FunctionSpan]] = {
        _span_key(span): [span] for span in roots
    }
    queue: list[tuple[FunctionSpan, FunctionSpan]] = [(span, span) for span in roots]
    visited: set[tuple[SpanKey, SpanKey]] = set()
    while queue:
        current, origin = queue.pop(0)
        visit_key = (_span_key(current), _span_key(origin))
        if visit_key in visited:
            continue
        visited.add(visit_key)
        for _, callee in graph.get(_span_key(current), []):
            target = by_key.get(callee)
            if target is None:
                continue
            target_roots = reachable.setdefault(_span_key(target), [])
            if not any(_span_key(item) == _span_key(origin) for item in target_roots):
                target_roots.append(origin)
            queue.append((target, origin))
    return {key: tuple(items) for key, items in reachable.items()}


def _temporal_risk_lines(source: str, span: FunctionSpan) -> list[int]:
    """Keep temporal locators only when time reaches an outcome-relevant path."""

    time_lines = _time_locator_lines(source, span)
    if not time_lines:
        return []
    lines = source.splitlines()
    body_lines = lines[max(span.start_line, 0) : min(span.end_line, len(lines))]
    body = "\n".join(body_lines)
    condition_lines = set(_first_matching_lines(source, span, _TIME_CONDITION_LOCATOR))
    producer_lines: set[int] = set()
    for line_number in time_lines:
        text = lines[line_number - 1].strip()
        if re.search(r"\bemit\b", text, re.I) and not _STATE_WRITE.search(text):
            continue
        if (
            _STATE_WRITE.search(text)
            and span.state_mutability.casefold() not in {"view", "pure", "constant"}
            and re.search(
            r"\b(?:reward|checkpoint|accumul|stake|share|rate|epoch|period|s(?:at)?|last|timestamp|"
            r"contribution|balance|amount|global|total)\w*\b",
            text,
            re.I,
            )
        ):
            producer_lines.add(line_number)

    # Eligibility checks use a temporal condition plus a cumulative amount or
    # cap; they are consumers even when the function is view-only.
    eligibility = bool(
        condition_lines
        and re.search(r"\b(?:contribution|cap|limit|amount|msg\s*\.\s*value|balance|quota)\w*\b", body, re.I)
        and re.search(r"\b(?:return|require|assert|if)\b", body, re.I)
    )
    semantic_effect = _has_semantic_state_or_asset_effect(source, span)
    condition_relevant = bool(
        condition_lines
        and (
            eligibility
            or semantic_effect
            or bool(re.search(r"\b(?:checkpoint|reward|stake|order|expiration|deadline)\w*\b", body, re.I))
        )
    )
    selected = set(producer_lines)
    if condition_relevant:
        selected.update(condition_lines)
    elif not condition_lines:
        # Capturing the current timestamp into an unlock/expiry field is
        # lifecycle bookkeeping, not a temporal exploit by itself.  Keep
        # producer-only rows when the timestamp feeds a rate/reward/state
        # calculation, but exclude plain lock-time capture assignments.
        selected = {
            line_number
            for line_number in selected
            if not re.search(
                r"\b(?:unlock|expiry|expiration|release|maturity|start|end)\w*\b\s*=",
                lines[line_number - 1],
                re.I,
            )
        }
    if not selected:
        return []

    # Keep an event timestamp only when the same function already contains a
    # producer/consumer locator and a mutable or asset effect.  Plain event
    # timestamps and cooldown bookkeeping therefore remain negative examples.
    has_effect = semantic_effect
    if has_effect and (condition_lines or producer_lines):
        for line_number in time_lines:
            if re.search(r"\bemit\b", lines[line_number - 1], re.I):
                selected.add(line_number)
                continue
            start = max(span.start_line - 1, line_number - 12)
            context = "\n".join(lines[start:line_number])
            if re.search(r"\bemit\b", context, re.I):
                selected.add(line_number)
    return sorted(selected)


def _has_semantic_state_or_asset_effect(source: str, span: FunctionSpan) -> bool:
    """Ignore local temporaries when deciding whether time reaches an outcome."""

    body = _function_body(source, span)
    if _EXTERNAL_CALL.search(body):
        return True
    lines = source.splitlines()
    state_name = re.compile(
        r"\b(?:state|reward|stake|share|balance|total|s(?:at)?|last|checkpoint|"
        r"accumul|amount|liquidity|reserve|supply|status|locked|active)\w*\b",
        re.I,
    )
    for line_number in range(span.start_line, min(span.end_line, len(lines)) + 1):
        text = lines[line_number - 1]
        if not _STATE_WRITE.search(text):
            continue
        lhs = re.split(r"(?:=|\+=|-=|\*=|/=|\+\+|--)", text, maxsplit=1)[0]
        if "." in lhs or state_name.search(lhs):
            return True
    return False


def _temporal_semantic_proposals(
    source: str,
    spans: list[FunctionSpan],
    roots: dict[str, tuple[FunctionSpan, ...]],
) -> list[dict[str, Any]]:
    graph = _call_graph(source, spans)
    by_key = {_span_key(span): span for span in spans}
    rows: list[dict[str, Any]] = []
    for span in spans:
        reachable_roots = roots.get(_span_key(span), ())
        locator_lines = _temporal_risk_lines(source, span)
        if not reachable_roots and locator_lines:
            # Old Solidity sources may hide the public fallback entrypoint or
            # route reachability through a modifier body.  A strong
            # producer/consumer path is still source-local evidence; retain
            # the function itself as the conservative entrypoint metadata.
            reachable_roots = (span,)
        if not reachable_roots:
            continue
        if not locator_lines:
            continue

        provider_edges: list[tuple[FunctionSpan, int, list[int]]] = []
        for call_line, target_key in graph.get(_span_key(span), []):
            provider = by_key.get(target_key)
            if provider is None or not _TIME_PROVIDER_NAME.fullmatch(provider.name):
                continue
            provider_lines = _first_matching_lines(source, provider, _TIME_LOCATOR)
            if provider_lines:
                provider_edges.append((provider, int(call_line), provider_lines))

        # A timestamp getter is only a producer.  Do not emit it as a
        # standalone vulnerability; emit it from the consumer path below.
        if _TIME_PROVIDER_NAME.fullmatch(span.name) and not provider_edges:
            continue

        if provider_edges:
            for provider, call_line, provider_lines in provider_edges:
                primary_line = provider_lines[0]
                for root in reachable_roots:
                    rows.append(
                        {
                            "risk_type": "time_manipulation",
                            "source_evidence_kind": "temporal_producer_consumer",
                            "source_grounded": True,
                            "function_name": provider.name,
                            "entrypoint_function_name": root.name,
                            "entrypoint_lines": [root.start_line],
                            "sink_function_name": span.name,
                            "sink_line": locator_lines[0],
                            "line": primary_line,
                            "time_provider_line": primary_line,
                            "time_consumer_call_line": call_line,
                            "time_consumer_lines": locator_lines,
                            "evidence_lines": sorted(
                                {
                                    provider.start_line,
                                    span.start_line,
                                    primary_line,
                                    call_line,
                                    *locator_lines,
                                }
                            ),
                            "reason": (
                                f"{provider.name}() returns a block time value used by "
                                f"{span.name}() in a time-sensitive economic or eligibility branch"
                            ),
                        }
                    )
            continue

        call_lines, write_lines = _function_effect_lines(source, span)
        for root in reachable_roots:
            evidence = sorted({span.start_line, root.start_line, *locator_lines, *call_lines, *write_lines})
            for line in locator_lines:
                rows.append(
                    {
                        "risk_type": "time_manipulation",
                        "source_evidence_kind": "temporal_producer_consumer",
                        "source_grounded": True,
                        "function_name": span.name,
                        "entrypoint_function_name": root.name,
                        "entrypoint_lines": [root.start_line],
                        "line": line,
                        "evidence_lines": evidence,
                        "reason": f"time value at @{line} reaches a state, reward, order, or eligibility outcome",
                    }
                )
    return rows


def _financial_flow_proposals(
    source: str,
    spans: list[FunctionSpan],
    roots: dict[str, tuple[FunctionSpan, ...]],
) -> list[dict[str, Any]]:
    """Find order-sensitive quote/reserve flows with an explicit asset bind."""

    rows: list[dict[str, Any]] = []
    ignored_identifiers = {
        "address",
        "bool",
        "bytes",
        "calldata",
        "data",
        "external",
        "false",
        "if",
        "memory",
        "msg",
        "public",
        "return",
        "sender",
        "storage",
        "this",
        "true",
        "uint",
        "uint256",
        "view",
    }

    def identifiers(text: str) -> set[str]:
        result: set[str] = set()
        for value in re.findall(r"\b[A-Za-z_]\w*\b", text or ""):
            pieces = re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", value
            )
            for piece in pieces or [value]:
                key = piece.casefold()
                if key not in ignored_identifiers:
                    result.add(key)
        return result

    for span in spans:
        reachable_roots = roots.get(_span_key(span), ())
        if not reachable_roots:
            reachable_roots = (span,)
        body = _function_body(source, span)
        order_sensitive = bool(_FINANCIAL_ORDER_SENSITIVE.search(body))
        if (
            not _FINANCIAL_FLOW_TOKEN.search(body)
            or not _FINANCIAL_ORDER_OPERATION.search(body)
            or not _ARITHMETIC_LOCATOR.search(body)
            or not order_sensitive
        ):
            continue
        if _ORDER_PROTECTION_TOKEN.search(body):
            continue
        statements = _statement_spans(source, span)
        economic_statements = [
            statement
            for statement in statements
            if _FINANCIAL_FLOW_TOKEN.search(_mask_comments_preserve_lines(statement.text))
            and (
                _ARITHMETIC_LOCATOR.search(_mask_comments_preserve_lines(statement.text))
                or _ARITHMETIC_HELPER_CALL.search(_mask_comments_preserve_lines(statement.text))
                or _FINANCIAL_ORDER_OPERATION.search(_mask_comments_preserve_lines(statement.text))
            )
        ]
        economic_lines: set[int] = set()
        priority_economic_lines: set[int] = set()
        for statement in economic_statements:
            clean_statement = _mask_comments_preserve_lines(statement.text)
            matches = [
                pattern.search(clean_statement)
                for pattern in (
                    _ARITHMETIC_LOCATOR,
                    _ARITHMETIC_HELPER_CALL,
                    _FINANCIAL_ORDER_OPERATION,
                )
            ]
            statement_lines = {
                _statement_line_for_offset(statement, match.start(), source)
                for match in matches
                if match is not None
            }
            # An order-sensitive statement may contain no arithmetic itself
            # while the enclosing function still has an arithmetic producer.
            # Keep the statement span as a source-grounded locator instead of
            # allowing a valid semantic candidate to reach an empty line list.
            economic_lines.update(statement_lines or {statement.start_line})
            helper_match = _ARITHMETIC_HELPER_CALL.search(clean_statement)
            if helper_match is not None:
                priority_economic_lines.add(
                    _statement_line_for_offset(statement, helper_match.start(), source)
                )
        economic_lines = sorted(economic_lines)
        ordered_economic_lines = sorted(priority_economic_lines) + [
            line for line in economic_lines if line not in priority_economic_lines
        ]
        if not economic_statements:
            continue
        direct = _direct_effects(source, span)
        sink_events = [
            item
            for item in direct
            if item.get("kind") == "external_call"
            and (
                item.get("economic_dispatch")
                or (
                    item.get("asset_call")
                    and str(item.get("method") or "").casefold()
                    not in _INBOUND_ASSET_METHOD
                )
            )
        ]
        if not sink_events:
            continue
        binding: dict[str, Any] | None = None
        ordered_economic_statements = sorted(
            economic_statements,
            key=lambda item: (
                0 if _ARITHMETIC_HELPER_CALL.search(_mask_comments_preserve_lines(item.text)) else 1,
                item.start_line,
            ),
        )
        for economic in ordered_economic_statements:
            economic_vars = identifiers(_strip_comments(economic.text))
            for sink in sink_events:
                sink_vars = identifiers(_strip_comments(str(sink.get("text") or "")))
                shared = sorted(economic_vars & sink_vars)
                if not shared or int(sink.get("line") or 0) < economic.start_line:
                    continue
                binding = {
                    "bound": True,
                    "shared_variables": shared,
                    "economic_operation_lines": ordered_economic_lines,
                    "asset_sink_lines": sorted(
                        {
                            int(item.get("line") or 0)
                            for item in sink_events
                            if int(item.get("line") or 0) > 0
                        }
                    ),
                    "proof": f"{', '.join(shared)} flows from economic operation to asset sink",
                }
                break
            if binding is not None:
                break
        if binding is None:
            continue
        call_lines = sorted(
            {
                int(item.get("line") or 0)
                for item in sink_events
                if int(item.get("line") or 0) > 0
            }
        )
        write_lines = _persistent_state_write_lines(source, span)
        if re.search(r"\b(?:reward|reinvest|harvest|compound)\w*\b", body, re.I):
            evidence_kind = "permissionless_reward_quote_then_reinvestment_no_slippage_bound"
        elif re.search(r"\b(?:commit|commitment|reveal)\w*\b", body, re.I):
            evidence_kind = "permissionless_quote_then_asset_settlement"
        elif re.search(r"\b(?:share|mint|provide)\w*\b", body, re.I):
            evidence_kind = "permissionless_share_mint_without_minimum_shares"
        else:
            evidence_kind = "permissionless_ratio_quote_then_asset_settlement"
        source_local_fallback = not roots.get(_span_key(span))
        for root in reachable_roots:
            if not _callable_span(root) and not source_local_fallback:
                continue
            rows.append(
                {
                    "risk_type": "front_running",
                    "source_evidence_kind": evidence_kind,
                    "source_grounded": True,
                    "function_name": span.name,
                    "entrypoint_function_name": root.name,
                    "sink_function_name": span.name,
                    "sink_line": call_lines[0],
                    "entrypoint_lines": [root.start_line],
                    "line": ordered_economic_lines[0],
                    "economic_operation_lines": ordered_economic_lines,
                    "asset_sink_lines": binding["asset_sink_lines"],
                    "economic_to_sink_binding": binding,
                    "evidence_lines": sorted({span.start_line, root.start_line, *economic_lines, *call_lines, *write_lines}),
                    "reason": "bound economic operation reaches an asset sink without an effective ordering bound",
                }
            )
    return rows


def _inherited_effect_proposals(
    source: str,
    spans: list[FunctionSpan],
    roots: dict[str, tuple[FunctionSpan, ...]],
) -> list[dict[str, Any]]:
    """Propagate source-local effects through inherited ``super.*`` calls."""

    rows: list[dict[str, Any]] = []
    super_call = re.compile(r"\bsuper\s*\.\s*(?P<method>[A-Za-z_]\w*)\s*\(", re.I)
    for span in spans:
        reachable_roots = roots.get(_span_key(span), ())
        if not reachable_roots:
            continue
        body = _function_body(source, span)
        for match in super_call.finditer(body):
            method = match.group("method")
            if not _INHERITED_MUTATING_METHOD.match(method):
                continue
            line = span.start_line + body[: match.start()].count("\n")
            for root in reachable_roots:
                rows.append(
                    {
                        "risk_type": "access_control",
                        "source_evidence_kind": "inherited_effect_propagation",
                        "source_grounded": True,
                        "function_name": span.name,
                        "entrypoint_function_name": root.name,
                        "entrypoint_lines": [root.start_line],
                        "sink_function_name": method,
                        "sink_line": line,
                        "line": line,
                        "evidence_lines": sorted({span.start_line, root.start_line, line}),
                        "access_semantic_proof": {
                            "resource": f"inherited mutating effect super.{method}()",
                            "actor_path": "callable entrypoint reaches inherited resource operation",
                        },
                        "reason": f"{span.name}() forwards a mutating effect through super.{method}()",
                    }
                )
    return rows


def _access_resource_proof(
    source: str, span: FunctionSpan, root: FunctionSpan
) -> dict[str, Any] | None:
    """Return a source-local resource proof for a possible access finding."""

    if root.is_constructor or span.is_constructor:
        return None
    if root.state_mutability.casefold() in {"view", "pure", "constant"}:
        return None
    body = _function_body(source, span)
    root_body = _function_body(source, root)
    delegated_shape = _delegated_actor_resource_shape(source, span, root)
    if delegated_shape is not None:
        authority = authorization_summary(source, root)
        if authority["status"] != "none" or _actor_identity_bound(body, delegated_shape["actor_name"]):
            return None
        resource_lines = delegated_shape["resource_lines"]
        call_line = int(delegated_shape["call_line"])
        return {
            "resource": f"positions[{delegated_shape['subject_name']}] liquidation state",
            "resource_kind": "subject_scoped_state",
            "resource_scope": "shared",
            "resource_operation": f"liquidate({delegated_shape['actor_name']}, {delegated_shape['subject_name']}, ...)",
            "resource_evidence_lines": sorted({call_line, *resource_lines}),
            "actor": "permissionless caller",
            "actor_path": "public entrypoint accepts an unbound liquidation actor",
            "actor_evidence_lines": [root.start_line, call_line],
            "authority": "none",
            "authority_effective": False,
            "authority_evidence_lines": [],
            "authority_requirement": f"{delegated_shape['actor_name']} must be bound to msg.sender or an effective authority guard",
        }
    call_lines, _effect_write_lines = _function_effect_lines(source, span)
    write_lines = _persistent_state_write_lines(source, span)
    if not call_lines and not write_lines:
        return None
    authority = authorization_summary(source, root)
    explicit_delegatecall = bool(re.search(r"\.\s*delegatecall\s*\(", body, re.I))
    # The resource name may be domain-specific (for example ``value`` or
    # ``state``), so a non-caller-scoped persistent write is still a shared
    # resource candidate.  The actor/authority proof below decides whether it
    # is actually permissionless; lexical names alone must not be the gate.
    asset_operation = bool(_ASSET_OPERATION_CALL.search(body))
    allowance_only_write = bool(write_lines) and all(
        re.search(
            r"\ballow(?:ance|ances|ed)\b",
            source.splitlines()[line - 1],
            re.I,
        )
        for line in write_lines
    )
    caller_scoped_write = bool(_CALLER_SCOPED_RESOURCE_WRITE.search(body)) or allowance_only_write
    subject_scoped_asset = bool(_SUBJECT_SCOPED_ASSET_PATH.search(body))
    workflow_state_write = _is_permissionless_workflow_state(source, write_lines)
    caller_refund = bool(_CALLER_REFUND_TRANSFER.search(body))
    shared_state_write = bool(write_lines) and not caller_scoped_write
    persistent_text = "\n".join(
        source.splitlines()[line - 1]
        for line in write_lines
        if 1 <= line <= len(source.splitlines())
    )
    global_write = bool(_GLOBAL_RESOURCE_WRITE.search(persistent_text))
    if caller_refund and not global_write:
        asset_operation = False

    # Standard ERC20 entrypoints and their internal accounting helper are not
    # authority boundaries.  Their shared-looking balance writes are part of
    # the token interface contract, not an access-control failure.
    standard_token_path = (
        root.name.casefold() in {"transfer", "transferfrom", "approve"}
        and bool(re.search(r"\b(?:_balances?|_allowed|allowance)\b", body, re.I))
    )
    if standard_token_path:
        return None

    # Receipt-token mint/burn helpers are often reached from a public stake or
    # unstake entrypoint.  Evaluate the caller's asset flow at the root as
    # well as the helper body; otherwise `_mint`/`_burn` is mistaken for an
    # arbitrary supply-authority operation.
    root_receipt_flow = bool(
        re.search(r"\b_?(?:mint|burn)\s*\(", root_body, re.I)
        and re.search(
            r"\.(?:safeTransferFrom|transferFrom|safeTransfer|transfer)\s*\([^;{}\n]*"
            r"(?:msg\s*\.\s*sender|_msgSender\s*\(\)|address\s*\(\s*this\s*\)|[A-Za-z_]\w*)",
            root_body,
            re.I,
        )
        and re.search(r"\b(?:msg\s*\.\s*sender|_msgSender\s*\(\))\b", root_body, re.I)
    )
    if root_receipt_flow:
        return None

    # A public timestamp/rate accumulator is not an authority-sensitive
    # resource merely because it writes storage.  Keep the access category
    # focused on ownership, permissions, supply, and shared asset control.
    critical_global_resource = bool(
        re.search(
            r"\b(?:owner|admin(?:s)?|governance|guardian|operator|minter|pauser|"
            r"roles?|permissions?|allow(?:ed|ance|list)|_?totalSupply|supply|"
            r"treasury|router|controller|vault|implementation|initialized|"
            r"emissionRate|endBlock|startBlock)\b",
            persistent_text,
            re.I,
        )
    )
    temporal_accumulator = bool(
        re.search(
            r"\b(?:lastUpdateTimestamp|timeElapsed|stakingRate|stakingRateStored|"
            r"ratePerSecond|timestamp|epoch|deadline|unlockTime)\b",
            persistent_text,
            re.I,
        )
    )
    if global_write and temporal_accumulator and not critical_global_resource and not asset_operation:
        return None
    generic_state = False
    if (
        not global_write
        and not asset_operation
        and not explicit_delegatecall
        and authority["status"] == "none"
    ):
        if not shared_state_write or workflow_state_write or temporal_accumulator:
            return None
        generic_state = True
    privileged_sink = bool(_PRIVILEGED_ASSET_FUNCTION.match(root.name))
    # A generic public protocol operation is not an authority boundary.  Keep
    # a candidate for an explicitly classified guard so the clean sentinel can
    # record the protection, but require a privileged resource shape for an
    # otherwise permissionless entrypoint.
    if (
        not global_write
        and not explicit_delegatecall
        and not privileged_sink
        and not generic_state
        and authority["status"] == "none"
    ):
        return None
    # User-owned deposits, stakes, and withdrawals are resource use, not
    # missing authority.  Keep them out unless the same body also mutates a
    # global critical resource such as supply, ownership, or configuration.
    if caller_scoped_write and not global_write:
        return None
    # A caller-bound trove/position/withdrawal path is an operation on the
    # caller's own resource, not an authority bypass.  This covers external
    # protocol calls such as closeTrove(msg.sender) where the local body does
    # not contain a Solidity mapping write for the subject check.
    if subject_scoped_asset and not global_write:
        return None
    # Receipt-token mint/burn around an incoming/outgoing user asset flow is
    # ordinary accounting for the caller's position, not arbitrary supply
    # authority.  Keep unrelated global supply writes (for example burn of a
    # caller-selected balance) on the access-control path.
    receipt_flow = bool(
        re.search(r"\b_?(?:mint|burn)\s*\(", body, re.I)
        and re.search(
            r"\.(?:safeTransferFrom|transferFrom|safeTransfer|transfer)\s*\([^;{}\n]*"
            r"(?:msg\s*\.\s*sender|_msgSender\s*\(\)|address\s*\(\s*this\s*\))",
            body,
            re.I,
        )
    )
    if receipt_flow:
        global_write = False
        if not asset_operation:
            asset_operation = True
    if authority["effective"]:
        actor = "authorized caller"
    elif authority["status"] == "unclassified_only_modifier":
        actor = "caller with unclassified authority guard"
    else:
        actor = "permissionless caller"
    resource_lines = call_lines if asset_operation else write_lines
    resource_text = next(
        (
            line.strip()
            for line_number, line in enumerate(source.splitlines(), start=1)
            if line_number in set(resource_lines)
        ),
        "",
    )
    if global_write:
        resource = "global critical state or supply"
        resource_kind = "global_state"
    elif generic_state:
        resource = "non-privileged shared state"
        resource_kind = "generic_state"
    else:
        resource = "concrete asset operation"
        resource_kind = "asset_operation"
    return {
        "resource": resource,
        "resource_kind": resource_kind,
        "resource_scope": "shared",
        "resource_operation": resource_text,
        "resource_evidence_lines": sorted(set(resource_lines)),
        "actor": actor,
        "actor_path": "callable entrypoint reaches the resource operation",
        "actor_evidence_lines": [root.start_line],
        "authority": authority["status"],
        "authority_effective": bool(authority["effective"]),
        "authority_evidence_lines": list(authority.get("evidence_lines") or []),
        "authority_requirement": "entrypoint authority must be proven separately",
    }


def _reentrancy_locator_line(
    source: str,
    span: FunctionSpan,
    summary: dict[str, Any],
) -> int:
    """Choose the semantic operation line for a reentrancy proposal."""

    events = [
        item
        for item in (summary.get("events") or [])
        if item.get("kind") == "external_call" and isinstance(item.get("line"), int)
    ]
    if not events:
        return span.start_line
    first_call = min(int(item["line"]) for item in events)
    return first_call


def _source_semantic_proposals(source: str, spans: list[FunctionSpan]) -> list[dict[str, Any]]:
    """Recover source-local access/reentrancy proposals lost before admission.

    This is deliberately proposal-only.  Authorization, mutex, and CEI are
    attached as metadata by ``normalize_proposal`` and still reject the row at
    admission; they never remove it from this high-recall pool.
    """

    graph = _call_graph(source, spans)
    roots = _reachable_entrypoints(spans, graph)
    direct_effects_cache: dict[SpanKey, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for span in spans:
        reachable_roots = roots.get(_span_key(span), ())
        if not reachable_roots:
            continue
        span_key = _span_key(span)
        direct_effects_cache[span_key] = _direct_effects(source, span)
        call_lines = sorted(
            {int(item["line"]) for item in direct_effects_cache[span_key] if item.get("kind") == "external_call"}
        )
        write_lines = sorted(
            {int(item["line"]) for item in direct_effects_cache[span_key] if item.get("kind") == "state_write"}
        )
        summary = reentrancy_effect_summary(
            source,
            span,
            spans,
            _graph=graph,
            _direct_effects_cache=direct_effects_cache,
        )
        events = summary.get("events") or []
        external_events = [item for item in events if item.get("kind") == "external_call"]
        state_events = [item for item in events if item.get("kind") == "state_write"]
        for root in reachable_roots:
            evidence = sorted({span.start_line, root.start_line, *call_lines, *write_lines})
            if not evidence:
                continue
            access_proof = _access_resource_proof(source, span, root)
            if access_proof is not None:
                rows.append(
                    {
                        "risk_type": "access_control",
                        "source_evidence_kind": "source_access_surface",
                        "source_grounded": True,
                        "function_name": span.name,
                        "entrypoint_function_name": root.name,
                        "entrypoint_lines": [root.start_line],
                        "sink_function_name": span.name,
                        "sink_line": call_lines[-1] if call_lines else write_lines[-1] if write_lines else span.start_line,
                        "line": call_lines[0] if call_lines else write_lines[0] if write_lines else span.start_line,
                        "evidence_lines": evidence,
                        "access_semantic_proof": access_proof,
                        **_span_identity_payload(span, "function"),
                        **_span_identity_payload(root, "entrypoint"),
                    }
                )
            # A callback/asset interaction is proposal-worthy even when the local
            # ordering is complete or a mutex is present; those are admission
            # sentinels, not proposal filters.
            if external_events and summary.get("reentrant_order"):
                event_lines = sorted(
                    {
                        int(item.get("line") or 0)
                        for item in [*external_events, *state_events]
                        if int(item.get("line") or 0) > 0
                    }
                    | {
                        int(line)
                        for line in (summary.get("cross_function_state_write_lines") or [])
                        if isinstance(line, int) and line > 0
                    }
                )
                sink_event = external_events[0]
                sink_function = str(sink_event.get("function") or span.name)
                direct_external_events = [
                    item
                    for item in direct_effects_cache[span_key]
                    if item.get("kind") == "external_call"
                ]
                if (
                    span.visibility.casefold() in {"public", "external"}
                    and not direct_external_events
                    and sink_function.casefold() != span.name.casefold()
                    and span.name.casefold() != root.name.casefold()
                ):
                    # The source-local helper span owns the callback locus;
                    # keep a public wrapper only as entrypoint metadata.
                    continue
                entrypoint_call_line = _entrypoint_call_site_line(
                    source,
                    root.name,
                    root,
                    [sink_function],
                )
                call_site_line = _reentrancy_locator_line(source, span, summary)
                report_function = span.name
                report_line = call_site_line
                if (
                    _REENTRANCY_READ_HELPER_NAME.match(sink_function)
                    and entrypoint_call_line is not None
                ):
                    # A getter/rate/oracle helper is the untrusted read boundary,
                    # but the externally reachable operation owns the finding.
                    report_function = root.name
                    report_line = entrypoint_call_line
                rows.append(
                    {
                        "risk_type": "reentrancy",
                        "source_evidence_kind": "source_event_sequence",
                        "source_grounded": True,
                        "function_name": report_function,
                        "entrypoint_function_name": root.name,
                        "entrypoint_lines": [root.start_line],
                        "sink_function_name": sink_function,
                        "sink_line": int(sink_event.get("line") or span.start_line),
                        "line": report_line,
                        "entrypoint_call_site_line": entrypoint_call_line,
                        "evidence_lines": sorted({span.start_line, root.start_line, *event_lines}),
                        **_span_identity_payload(span, "function"),
                        **_span_identity_payload(root, "entrypoint"),
                    }
                )
    rows.extend(_temporal_semantic_proposals(source, spans, roots))
    rows.extend(_financial_flow_proposals(source, spans, roots))
    rows.extend(_inherited_effect_proposals(source, spans, roots))
    return rows


def _entrypoint_call_site_line(
    source: str,
    entrypoint: str,
    entrypoint_span: FunctionSpan | None,
    targets: Iterable[str],
) -> int | None:
    """Locate the entrypoint's local call into a cross-function sink."""

    if entrypoint_span is None:
        return None
    wanted = {
        str(value).strip().casefold()
        for value in targets
        if str(value).strip() and str(value).strip().casefold() != entrypoint.casefold()
    }
    if not wanted:
        return None
    body = _function_body(_mask_comments_preserve_lines(source), entrypoint_span)
    for target in sorted(wanted):
        match = re.search(rf"(?<![.\w]){re.escape(target)}\s*\(", body, re.I | re.S)
        if match:
            return entrypoint_span.start_line + body[: match.start()].count("\n")
    return None


def _reported_function_name(category: str, call_site: str, sink: str) -> str:
    """Choose the primary report function without discarding call-site metadata.

    Cross-function candidates retain both names.  Access-control findings use
    the concrete sink when one is available; front-running uses a helper sink
    only when it is explicit (the common ``_safeSwap``/``_execute`` shape).
    Generic front-running rows continue to report their call-site function.
    """

    call_site = str(call_site or "").strip()
    sink = str(sink or "").strip()
    if sink and sink.casefold() != call_site.casefold():
        if category == "access_control":
            return sink
        if category == "reentrancy":
            # The proposal's function is the canonical vulnerable operation;
            # the sink is a separate evidence role.  Promoting the sink here
            # collapses ``_transfer -> _swap`` into ``_swap`` and loses the
            # Gold-aligned wrapper locus.
            return call_site
        if category == "front_running" and _FRONT_RUNNING_EXPLICIT_SINK_NAME.match(sink):
            return sink
    return call_site


def normalize_proposal(row: dict[str, Any], source: str, source_id: str, spans: list[FunctionSpan]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    category = str(row.get("risk_type") or row.get("primary_category") or row.get("category") or "").strip().casefold()
    if category not in LABELS:
        return None
    proof = row.get("proof") if isinstance(row.get("proof"), dict) else {}
    call_path = [str(value).strip() for value in (proof.get("call_path") or []) if str(value).strip()]
    raw_function = str(row.get("function_name") or "").strip()
    raw_entrypoint = str(row.get("entrypoint_function_name") or row.get("entrypoint") or proof.get("entrypoint") or "").strip()
    raw_sink = str(row.get("sink_function_name") or row.get("sink") or proof.get("sink") or "").strip()
    raw_entrypoint_lines = [*(row.get("entrypoint_lines") or [])]
    evidence = _valid_lines(
        [row.get("line"), *(row.get("evidence_lines") or []), *raw_entrypoint_lines, *(proof.get("evidence_lines") or [])],
        source,
    )
    raw_line = row.get("line") if isinstance(row.get("line"), int) and row.get("line") > 0 else (evidence[0] if evidence else 0)
    line = raw_line
    call_span = _span_from_row(
        source,
        spans,
        row,
        "function",
        name=raw_function,
        line=raw_line,
    )
    # Function identity is resolved from the parser-owned line map.  The raw
    # detector name is retained separately so cross-function evidence cannot
    # overwrite the call-site identity.
    function_name = call_span.name if call_span else raw_function
    entrypoint_span = _span_from_row(
        source,
        spans,
        row,
        "entrypoint",
        name=raw_entrypoint,
        line=raw_entrypoint_lines[0] if raw_entrypoint_lines else None,
        same_as=call_span,
    ) if (raw_entrypoint or raw_entrypoint_lines) else None
    entrypoint = raw_entrypoint or (entrypoint_span.name if entrypoint_span else "")
    if not entrypoint and call_path:
        entrypoint = call_path[0]
    if not entrypoint:
        entrypoint = raw_function or function_name
    if entrypoint_span is None and entrypoint:
        entrypoint_span = _span_from_row(
            source,
            spans,
            row,
            "entrypoint",
            name=entrypoint,
            same_as=call_span,
        )
    cross_function_call_line = None
    # Reentrancy and temporal producer rows intentionally report the sink or
    # producer span.  Only access-control/front-running rows use the public
    # entrypoint call as their primary locator.
    should_relocate_cross_function = category in {"access_control", "front_running"}
    entrypoint_call_line = None
    if entrypoint and entrypoint_span:
        entrypoint_call_line = _entrypoint_call_site_line(
            source,
            entrypoint,
            entrypoint_span,
            [*call_path[1:], raw_function, raw_sink],
        )
    if category in {"access_control", "front_running"} and entrypoint and should_relocate_cross_function:
        cross_function_call_line = entrypoint_call_line
        if cross_function_call_line is not None:
            line = cross_function_call_line
            call_span = function_at_line(source, line, spans) or call_span
            function_name = call_span.name if call_span else raw_function
    sink = raw_sink
    if sink.casefold() in _SEMANTIC_SINK_NAMES:
        sink = ""
    if not sink and len(call_path) > 1:
        for value in reversed(call_path[1:]):
            if value.casefold() not in _SEMANTIC_SINK_NAMES and value.casefold() != entrypoint.casefold():
                sink = value
                break
    sink_lines = _valid_lines([*(row.get("state_write_lines") or []), *(row.get("sink_lines") or []), row.get("sink_line")], source)
    explicit_sink_line = row.get("sink_line") if isinstance(row.get("sink_line"), int) and row.get("sink_line") > 0 else None
    resolved_sink_line = explicit_sink_line or (sink_lines[-1] if sink_lines else 0)
    sink_span = _span_from_row(
        source,
        spans,
        row,
        "sink",
        name=sink,
        line=resolved_sink_line or None,
        same_as=call_span,
    ) if sink else None
    if cross_function_call_line is not None and sink_span is None and raw_function.casefold() != entrypoint.casefold():
        sink = raw_function
        sink_span = _span_from_row(
            source,
            spans,
            row,
            "sink",
            name=raw_function,
            same_as=call_span,
        )
    if not sink and sink_span:
        sink = sink_span.name
    if not resolved_sink_line and sink_span:
        sink_call_lines, sink_write_lines = _function_effect_lines(source, sink_span)
        resolved_sink_line = (
            (sink_call_lines or sink_write_lines or [sink_span.start_line])[-1]
        )
    # Locator-only refinement: when an upstream row points at a declaration,
    # move the primary line to the category's source operation while keeping
    # the proposal and admission semantics unchanged.
    if call_span and category == "time_manipulation":
        time_lines = _time_locator_lines(source, call_span)
        if time_lines:
            condition_lines = _first_matching_lines(source, call_span, _TIME_CONDITION_LOCATOR)
            # Preserve an explicitly supplied guard/event locator.  A bare
            # legacy row that points at a timestamp assignment still falls
            # back to its enclosing guard; source-semantic rows carry their
            # producer/consumer kind and retain the assignment line.
            raw_text = source.splitlines()[raw_line - 1] if 1 <= raw_line <= len(source.splitlines()) else ""
            preserve_raw = raw_line in time_lines and (
                bool(_TIME_CONDITION_LOCATOR.search(raw_text))
                or bool(re.search(r"\bemit\b", raw_text, re.I))
                or row.get("source_evidence_kind") == "temporal_producer_consumer"
            )
            line = raw_line if preserve_raw else (condition_lines[-1] if condition_lines else time_lines[-1])
            call_span = function_at_line(source, line, spans) or call_span
            evidence = sorted(set(evidence) | set(time_lines))
    elif call_span and category == "arithmetic":
        arithmetic_lines = _first_matching_lines(source, call_span, _ARITHMETIC_LOCATOR)
        if arithmetic_lines:
            span_lines = set(range(call_span.start_line, call_span.end_line + 1))
            explicit_operation_lines = [
                value
                for value in _valid_lines(
                    [
                        *(row.get("arithmetic_operation_lines") or []),
                        *(row.get("value_relevance_lines") or []),
                        *(
                            row.get("state_write_lines") or []
                            if not (
                                row.get("arithmetic_operation_lines")
                                or row.get("value_relevance_lines")
                            )
                            else []
                        ),
                    ],
                    source,
                )
                if value in span_lines
            ]
            if explicit_operation_lines:
                anchors = explicit_operation_lines
            else:
                anchors = [
                    value
                    for value in [raw_line, *evidence]
                    if value in span_lines
                ]
            locator_lines = sorted(set(arithmetic_lines) | set(explicit_operation_lines))
            line = min(
                locator_lines,
                key=lambda candidate: (
                    min(abs(candidate - anchor) for anchor in anchors)
                    if anchors
                    else 0,
                    abs(candidate - raw_line) if raw_line > 0 else 0,
                    candidate,
                ),
            )
            evidence = sorted(set(evidence) | set(arithmetic_lines))
    entrypoint_auth = (
        authorization_summary(source, entrypoint_span)
        if entrypoint_span
        else {"status": "unknown", "effective": False, "evidence_lines": []}
    )
    local_auth = (
        authorization_summary(source, call_span)
        if call_span
        else {"status": "unknown", "effective": False, "evidence_lines": []}
    )
    auth = entrypoint_auth if entrypoint_span else local_auth
    effects = (
        reentrancy_effect_summary(source, entrypoint_span or call_span, spans)
        if (entrypoint_span or call_span)
        else {"mutex": False, "cei_complete": False, "protection_evidence": []}
    )
    sentinel_row = {
        **row,
        "risk_type": category,
        "line": line,
        "function_name": function_name,
    }
    sentinel_row.update(_span_identity_payload(call_span, "function"))
    sentinel_row.update(_span_identity_payload(entrypoint_span, "entrypoint"))
    sentinel_row.update(_span_identity_payload(sink_span, "sink"))
    clean, admission_reason, protection = _clean_sentinel(sentinel_row, source, spans)
    reported_function_name = _reported_function_name(category, call_span.name if call_span else function_name, sink)
    entrypoint_lines = _valid_lines(
        [*raw_entrypoint_lines, entrypoint_span.start_line if entrypoint_span else 0],
        source,
    )
    proof_path = [
        value
        for value in (entrypoint, function_name, sink)
        if value
    ]
    deduped_proof_path: list[str] = []
    for value in proof_path:
        if value.casefold() not in {item.casefold() for item in deduped_proof_path}:
            deduped_proof_path.append(value)
    semantic_proof = {
        "actor": (
            "permissionless caller"
            if entrypoint_auth["status"] == "none"
            else "authorized caller"
            if entrypoint_auth["effective"]
            else "authority unknown"
        ),
        "resource": str(
            (row.get("access_semantic_proof") or {}).get("resource")
            or sink
            or function_name
            or entrypoint
        ),
        "authority": entrypoint_auth["status"],
        "path": deduped_proof_path,
        "evidence_lines": sorted(set([*evidence, *entrypoint_lines, *protection])),
    }
    candidate = dict(row)
    candidate.update(
        {
            "risk_type": category,
            "function_name": function_name,
            "contract_name": call_span.contract_name if call_span else "",
            **_span_identity_payload(call_span, "function"),
            **_span_identity_payload(entrypoint_span, "entrypoint"),
            **_span_identity_payload(sink_span, "sink"),
            "reported_function_name": reported_function_name,
            "source_function_name": raw_function,
            "entrypoint_function_name": entrypoint,
            "call_site_function": call_span.name if call_span else function_name,
            "call_site_line": entrypoint_call_line or line,
            "sink_function_name": sink or (sink_span.name if sink_span else ""),
            "sink_line": resolved_sink_line or (raw_line if cross_function_call_line is not None else (line if sink and call_span and call_span.name.casefold() == sink.casefold() else 0)),
            "line": line,
            "evidence_lines": evidence,
            "entrypoint_lines": entrypoint_lines,
            "proposal_stage": "proposal",
            "guard_status": {
                "authorization": auth["status"],
                "mutex": bool(effects.get("mutex")),
                "cei_complete": bool(effects.get("cei_complete")),
                "callback_capable": bool(effects.get("callback_capable")),
                "typed_return_checked": admission_reason == "checked_or_captured_return",
            },
            "protection_evidence": sorted(set([*auth.get("evidence_lines", []), *effects.get("protection_evidence", []), *protection])),
            "semantic_proof": semantic_proof,
            "access_semantic_proof": row.get("access_semantic_proof"),
            "admission": not clean,
            "admission_reason": "clean_sentinel_rejected" if clean else "proposal_admitted",
            "clean_sentinel": clean,
            "candidate_id": str(row.get("candidate_id") or stable_candidate_id({**row, "risk_type": category, "line": line, "function_name": function_name}, source_id=source_id)),
        }
    )
    return candidate


def merge_proposals(
    lifecycle_rows: Iterable[dict[str, Any]],
    c2_rows: Iterable[dict[str, Any]],
    source: str,
    source_id: str,
    *,
    arithmetic_provenance_rows: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    spans = parse_function_spans(source)
    merged: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in [
        *lifecycle_rows,
        *c2_rows,
        *typed_return_candidates(source, spans),
        *_source_semantic_proposals(source, spans),
        *_arithmetic_provenance_proposals(source, spans, arithmetic_provenance_rows),
        *_helper_arithmetic_proposals(source, spans),
    ]:
        candidate = normalize_proposal(row, source, source_id, spans)
        if candidate is None:
            continue
        key = (
            candidate["risk_type"],
            candidate["function_name"].casefold(),
            int(candidate.get("line") or 0),
            str(candidate.get("entrypoint_function_name") or "").casefold(),
        )
        if key not in merged:
            merged[key] = candidate
            continue
        current = merged[key]
        current["evidence_lines"] = sorted(set(current.get("evidence_lines", [])) | set(candidate.get("evidence_lines", [])))
        current["protection_evidence"] = sorted(set(current.get("protection_evidence", [])) | set(candidate.get("protection_evidence", [])))
        current["source_evidence_kind"] = current.get("source_evidence_kind") or candidate.get("source_evidence_kind")
        current["candidate_id"] = current.get("candidate_id") or candidate.get("candidate_id")
        for field in (
            "arithmetic_operation_lines",
            "state_write_lines",
            "value_relevance_lines",
            "economic_operation_lines",
            "asset_sink_lines",
            "sink_lines",
        ):
            current[field] = sorted(
                set(current.get(field) or []) | set(candidate.get(field) or [])
            )
        current["clean_sentinel"] = bool(current.get("clean_sentinel")) and bool(candidate.get("clean_sentinel"))
        current["admission"] = not current["clean_sentinel"]
    return list(merged.values())


def candidate_to_finding(candidate: dict[str, Any], source: str, *, locator_mode: str = "call_site", spans: Iterable[FunctionSpan] | dict[int, tuple[FunctionSpan, ...]] | None = None) -> dict[str, Any]:
    spans = spans or parse_function_spans(source)
    category = str(candidate.get("risk_type") or candidate.get("vulnerability_type") or "").casefold()
    call_site_line = int(candidate.get("call_site_line") or candidate.get("line") or 0)
    call_span = function_at_line(source, call_site_line, spans)
    entrypoint = str(candidate.get("entrypoint_function_name") or candidate.get("function_name") or "")
    sink = str(candidate.get("sink_function_name") or "")
    if locator_mode == "entrypoint":
        function_name = entrypoint or (call_span.name if call_span else str(candidate.get("function_name") or ""))
    elif locator_mode == "sink" and sink:
        function_name = sink
    else:
        function_name = str(
            candidate.get("reported_function_name")
            or candidate.get("function_name")
            or (call_span.name if call_span else entrypoint)
        )
    sink_line = int(candidate.get("sink_line") or 0)
    arithmetic_operation_lines = _valid_lines(
        candidate.get("arithmetic_operation_lines") or [], source
    )
    economic_operation_lines = _valid_lines(
        candidate.get("economic_operation_lines") or [], source
    )
    state_write_lines = _valid_lines(
        candidate.get("state_write_lines") or [], source
    )
    asset_sink_lines = _valid_lines(
        candidate.get("asset_sink_lines") or [], source
    )
    if category == "arithmetic" and arithmetic_operation_lines:
        primary_line = arithmetic_operation_lines[0]
    elif category == "front_running" and economic_operation_lines:
        primary_line = economic_operation_lines[0]
    else:
        primary_line = (
            sink_line
            if sink_line > 0 and sink and function_name.casefold() == sink.casefold()
            else call_site_line
        )
    evidence = _valid_lines(
        [
            call_site_line,
            primary_line,
            *arithmetic_operation_lines,
            *economic_operation_lines,
            *state_write_lines,
            *asset_sink_lines,
            *(candidate.get("evidence_lines") or []),
            *(candidate.get("entrypoint_lines") or []),
            sink_line,
        ],
        source,
    )
    cid = str(candidate.get("candidate_id") or stable_candidate_id(candidate))
    finding = {
        "alert_id": f"oracle:{cid}",
        "vulnerability_type": category,
        "function_name": function_name,
        "reported_function_name": function_name,
        "entrypoint_function_name": entrypoint,
        "call_site_function": call_span.name if call_span else "",
        "call_site_line": call_site_line,
        "entrypoint_lines": _valid_lines(candidate.get("entrypoint_lines") or [], source),
        "sink_function_name": sink,
        "sink_line": sink_line,
        "line": primary_line,
        "primary_line": primary_line,
        "source_anchor": {"source_line": primary_line, "line": primary_line} if primary_line > 0 else {},
        "source_anchor_lines": evidence,
        "attack_path": str(candidate.get("reason") or ""),
        "mechanism": str(candidate.get("source_evidence_kind") or candidate.get("admission_reason") or "source_proposal"),
        "candidate_id": cid,
    }
    for field, values in (
        ("arithmetic_operation_lines", arithmetic_operation_lines),
        ("economic_operation_lines", economic_operation_lines),
        ("state_write_lines", state_write_lines),
        ("asset_sink_lines", asset_sink_lines),
    ):
        if values:
            finding[field] = values
    return finding


__all__ = [
    "FunctionSpan",
    "LABELS",
    "authorization_summary",
    "build_line_to_function_map",
    "candidate_to_finding",
    "function_at_line",
    "merge_proposals",
    "normalize_proposal",
    "parse_function_spans",
    "reentrancy_effect_summary",
    "stable_candidate_id",
    "typed_return_candidates",
    "typed_return_signatures",
]
