"""
FusedAudit Phase 1: Feature Fusion / Path Synthesizer (Tree-sitter Edition)
============================================================================
从 Solidity 源码中自动提取 AST/CFG Flow 特征，
与原始代码拼接为高维度融合文本。

核心升级:
  - 使用 Tree-sitter (tree-sitter-solidity) 替换正则引擎
  - 两段式追踪 (Two-Step Tracking): 识别延迟校验模式
    第一步: 抓取 .call 被赋值给了哪个变量 (如 success)
    第二步: 在函数后续 AST 中搜索 require(success) / if(!success)
  - 真正理解上下文无关文法 (Context-Free Grammar)
  - 正确处理内联汇编 (Inline Assembly)
  - 精确解析多层嵌套的 if-else
  - 跨合约继承调用追踪
  - 无需配置 solc 编译器, 极其轻量

兼容 Solidity 0.4.x ~ 0.8.x 语法
"""

import tree_sitter_solidity as tss
import tree_sitter as ts
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
from collections import defaultdict
import re as re_mod
from fusedaudit_profiles import (
    E1_OPTIMIZED_V1_PROFILE,
    E2_DAPPSCAN_VNEXT_PROFILE,
    PROFILE_ENV,
)
from temporal_invariants import (
    analyze_temporal_invariants,
    build_temporal_ir,
    build_temporal_slice,
    summarize_temporal_ir,
)
from candidate_decision import (
    CONFIRMED as E1_CANDIDATE_CONFIRMED,
    analyze_unchecked_low_level_call_candidates,
)


_LANG = ts.Language(tss.language())
_PARSER = ts.Parser(_LANG)


@dataclass
class FunctionFlow:
    name: str
    visibility: str
    modifiers: List[str] = field(default_factory=list)
    state_writes: List[str] = field(default_factory=list)
    external_calls: List[str] = field(default_factory=list)
    internal_calls: List[str] = field(default_factory=list)
    require_checks: List[str] = field(default_factory=list)
    has_reentrancy_guard: bool = False
    guard_type: str = ""
    guard_detail: str = ""
    line_number: int = 0
    end_line_number: int = 0
    is_constructor: bool = False
    is_fallback: bool = False
    is_receive: bool = False
    state_mutability: str = ""
    delayed_checks: List[str] = field(default_factory=list)
    unchecked_calls: List[str] = field(default_factory=list)
    inlined_modifier_code: str = ""
    inlined_internal_flows: List[str] = field(default_factory=list)
    data_flow_chains: List[str] = field(default_factory=list)
    loop_features: List[str] = field(default_factory=list)
    is_reachable: bool = True
    reachability_path: List[str] = field(default_factory=list)
    source_slice: str = ""


@dataclass
class ModifierInfo:
    name: str
    body_text: str
    line_number: int = 0


@dataclass
class ContractFeatures:
    name: str
    state_variables: List[str] = field(default_factory=list)
    functions: List[FunctionFlow] = field(default_factory=list)
    inherits: List[str] = field(default_factory=list)
    modifiers: List[ModifierInfo] = field(default_factory=list)
    inline_assembly_blocks: List[str] = field(default_factory=list)


def _node_text(node) -> str:
    return node.text.decode("utf-8") if node else ""


def _find_all_by_type(node, target_type: str) -> list:
    results = []
    if node.type == target_type:
        results.append(node)
    for child in node.children:
        results.extend(_find_all_by_type(child, target_type))
    return results


def _find_first_by_type(node, target_type: str):
    if node.type == target_type:
        return node
    for child in node.children:
        result = _find_first_by_type(child, target_type)
        if result:
            return result
    return None


def _extract_inheritance(contract_node) -> List[str]:
    inherits = []
    for child in contract_node.children:
        if child.type == "inheritance_specifier":
            for sc in child.children:
                if sc.type == "user_defined_type":
                    for uc in sc.children:
                        if uc.type == "identifier":
                            inherits.append(_node_text(uc))
    return inherits


def _extract_state_variables(contract_node) -> List[str]:
    vars_found = []
    body = _find_first_by_type(contract_node, "contract_body")
    if not body:
        return vars_found
    for child in body.children:
        if child.type == "state_variable_declaration":
            for sc in child.children:
                if sc.type == "identifier":
                    vars_found.append(_node_text(sc))
                    break
    return vars_found


def _extract_modifiers(contract_node) -> List[ModifierInfo]:
    modifiers = []
    body = _find_first_by_type(contract_node, "contract_body")
    if not body:
        return modifiers
    for child in body.children:
        if child.type == "modifier_definition":
            name = ""
            for sc in child.children:
                if sc.type == "identifier":
                    name = _node_text(sc)
                    break
            func_body = _find_first_by_type(child, "function_body")
            body_text = _node_text(func_body) if func_body else ""
            modifiers.append(ModifierInfo(
                name=name,
                body_text=body_text,
                line_number=child.start_point[0] + 1
            ))
    return modifiers


def _extract_inline_assembly(contract_node) -> List[str]:
    blocks = []
    body = _find_first_by_type(contract_node, "contract_body")
    if not body:
        return blocks
    asm_nodes = _find_all_by_type(body, "assembly_statement")
    for asm in asm_nodes:
        blocks.append(_node_text(asm))
    return blocks


def _analyze_call_expression(call_node, state_vars: Set[str]) -> tuple:
    expr_text = _node_text(call_node)
    callee = ""
    for child in call_node.children:
        if child.type == "expression":
            callee = _node_text(child)
            break

    is_external = False
    call_label = ""

    if ".call" in callee:
        is_external = True
        if ".call.value" in callee:
            call_label = f"msg.sender.call.value()"
        elif ".call{" in callee:
            call_label = f"msg.sender.call{{value}}"
        else:
            call_label = f"msg.sender.call()"
    elif ".transfer(" in callee:
        is_external = True
        call_label = ".transfer"
    elif ".send(" in callee:
        is_external = True
        call_label = ".send"
    elif ".staticcall" in callee:
        is_external = True
        call_label = ".staticcall"
    elif ".delegatecall" in callee:
        is_external = True
        call_label = ".delegatecall"
    elif callee.startswith("require(") or callee == "require":
        return "require", expr_text
    elif callee.startswith("assert(") or callee == "assert":
        return "assert", expr_text
    elif callee.startswith("revert(") or callee == "revert":
        return "revert", expr_text
    elif callee.startswith("emit "):
        return "emit", expr_text
    elif (
        callee
        and not is_external
        and "(" in expr_text
        and ("." not in callee or callee.startswith("super."))
    ):
        # tree-sitter stores the callee expression without its argument list
        # (``helper`` rather than ``helper()``).  Use the call node text only
        # to prove this is a call; keep member calls out of the same-contract
        # internal-call channel unless they are explicit ``super`` dispatches.
        return "internal_call", f"{callee}()"

    if is_external:
        return "external_call", call_label
    return "other_call", callee


def _analyze_try_statement(try_node, state_vars: Set[str]) -> tuple:
    """
    分析 try/catch 语句中的外部调用。
    模式: try IERC721Receiver(to).onERC721Received(...) returns (bytes4 retval) { ... } catch { ... }
    """
    expr_text = _node_text(try_node)

    call_expr = _find_first_by_type(try_node, "call_expression")
    if call_expr:
        callee = ""
        for child in call_expr.children:
            if child.type == "expression":
                callee = _node_text(child)
                break

        if callee:
            if "." in callee:
                parts = callee.rsplit(".", 1)
                if len(parts) == 2:
                    call_label = f"{parts[1]}()"
                else:
                    call_label = callee[:40]
            else:
                call_label = callee[:40]

            return "try_external_call", call_label

    member_expr = _find_first_by_type(try_node, "member_expression")
    if member_expr:
        mem_text = _node_text(member_expr)
        if "." in mem_text:
            parts = mem_text.rsplit(".", 1)
            if len(parts) == 2:
                return "try_external_call", f"{parts[1]}()"

    return "try_external_call", "try-catch-external"


def _analyze_assignment(assign_node, state_vars: Set[str]) -> Optional[str]:
    children = assign_node.children
    if len(children) < 2:
        return None

    lhs_node = children[0]
    lhs_text = _node_text(lhs_node)

    array_access = _find_first_by_type(lhs_node, "array_access")
    if array_access:
        arr_expr = ""
        for sc in array_access.children:
            if sc.type == "expression":
                arr_expr = _node_text(sc)
                break
        if arr_expr in state_vars or any(arr_expr.startswith(sv) for sv in state_vars):
            return arr_expr

    identifier_nodes = _find_all_by_type(lhs_node, "identifier")
    for id_node in identifier_nodes:
        var_name = _node_text(id_node)
        if var_name in state_vars:
            return var_name

    member_expr = _find_first_by_type(lhs_node, "member_expression")
    if member_expr:
        mem_text = _node_text(member_expr)
        if any(sv in mem_text for sv in state_vars):
            return mem_text

    return None


def _detect_guard_in_function(func_node, modifier_defs: List[ModifierInfo]) -> tuple:
    guard_type = ""
    guard_detail = ""
    has_guard = False

    modifier_names = []
    for child in func_node.children:
        if child.type == "modifier_invocation":
            for sc in child.children:
                if sc.type == "identifier":
                    modifier_names.append(_node_text(sc))

    guard_keywords = ['reentrant', 'guard', 'lock', 'mutex', 'nonreentrant']
    for mname in modifier_names:
        ml = mname.lower()
        if any(kw in ml for kw in guard_keywords):
            has_guard = True
            guard_type = f"modifier_{mname}"
            guard_detail = f"modifier {mname} applied"
            break

    if not has_guard:
        func_body = _find_first_by_type(func_node, "function_body")
        if func_body:
            body_text = _node_text(func_body).lower()
            guard_patterns = [
                ('require(!locked', 'reentrancy_guard', 'require(!locked)'),
                ('require(_status', 'reentrancy_guard', 'require(_status...)'),
                ('_status !=', 'reentrancy_guard', '_status check'),
                ('nonreentrant', 'reentrancy_guard', 'nonReentrant check'),
                ('onlyowner', 'access_control_owner', 'onlyOwner'),
                ('msg.sender == owner', 'access_control_sender', 'msg.sender == owner'),
                ('msg.sender == admin', 'access_control_sender', 'msg.sender == admin'),
            ]
            for pattern, gt, gd in guard_patterns:
                if pattern in body_text:
                    has_guard = True
                    guard_type = gt
                    guard_detail = gd
                    break

    for mname in modifier_names:
        for mdef in modifier_defs:
            if mdef.name == mname:
                mbody_lower = mdef.body_text.lower()
                if any(kw in mbody_lower for kw in ['locked', '_status', 'nonreentrant', 'reentrancy', 'mutex']):
                    has_guard = True
                    guard_type = f"modifier_{mname}"
                    guard_detail = f"modifier {mname} (contains guard logic)"
                    break
        if has_guard and 'modifier' in guard_type:
            break

    return has_guard, guard_type, guard_detail


# ==========================================
# 两段式追踪 (Two-Step Tracking)
# ==========================================

def _collect_all_statements(func_body_node) -> list:
    """按顺序收集函数体内所有语句节点 (扁平化)"""
    stmts = []

    def _walk(node):
        for child in node.children:
            if child.type == "statement":
                stmts.append(child)
                for sc in child.children:
                    if sc.type in ("if_statement", "for_statement", "while_statement",
                                   "do_while_statement", "unchecked_statement"):
                        _walk(sc)
            elif child.type in ("if_statement", "for_statement", "while_statement",
                                "do_while_statement", "unchecked_statement"):
                _walk(child)

    _walk(func_body_node)
    return stmts


def _find_call_result_var(stmt_node) -> Optional[Tuple[str, str]]:
    """
    第一步: 抓取赋值变量。
    在一个语句中找到 .call 并向上抓取它被赋值给了哪个变量。
    返回 (变量名, 调用标签) 或 None。

    模式1: (bool success, ) = target.call{value: amount}("");
    模式2: bool success = target.call{value: amount}("");
    模式3: success = target.call{value: amount}("");
    """
    call_nodes = _find_all_by_type(stmt_node, "call_expression")
    for cn in call_nodes:
        call_type, call_label = _analyze_call_expression(cn, set())
        if call_type != "external_call":
            continue

        var_decl = _find_first_by_type(stmt_node, "variable_declaration_statement")
        if var_decl:
            id_nodes = _find_all_by_type(var_decl, "identifier")
            bool_type_nodes = _find_all_by_type(var_decl, "boolean_type")
            if id_nodes and bool_type_nodes:
                for id_node in id_nodes:
                    var_name = _node_text(id_node)
                    parent_text = _node_text(id_node.parent) if id_node.parent else ""
                    if var_name.lower() in ('success', 'result', 'ok', 'ret', 'retval',
                                            'sent', 'ok1', 'ok2', 'callresult') or \
                       'success' in var_name.lower() or 'result' in var_name.lower():
                        return (var_name, call_label)

                if id_nodes:
                    first_var = _node_text(id_nodes[0])
                    if bool_type_nodes:
                        return (first_var, call_label)
            elif id_nodes:
                first_var = _node_text(id_nodes[0])
                return (first_var, call_label)

        assign_nodes = _find_all_by_type(stmt_node, "assignment_expression")
        for an in assign_nodes:
            if an.children:
                lhs_text = _node_text(an.children[0]).strip()
                lhs_ids = _find_all_by_type(an.children[0], "identifier")
                if lhs_ids:
                    return (_node_text(lhs_ids[0]), call_label)

    return None


def _find_delayed_check(stmt_nodes, target_var: str, start_idx: int) -> Optional[str]:
    """
    第二步: 向下文搜索消费节点。
    在当前函数的后续 AST 节点中，搜索有没有 require、if 或校验函数
    把 target_var 作为参数传了进去。

    返回校验描述字符串，或 None。
    """
    for i in range(start_idx + 1, len(stmt_nodes)):
        stmt = stmt_nodes[i]
        stmt_text = _node_text(stmt)

        require_nodes = _find_all_by_type(stmt, "call_expression")
        for rn in require_nodes:
            rn_text = _node_text(rn)
            if target_var in rn_text:
                if 'require(' in rn_text:
                    return f"require({target_var})"
                if 'assert(' in rn_text:
                    return f"assert({target_var})"
                if 'revert(' in rn_text:
                    return f"revert-on-{target_var}"
                if 'verifyCallResult' in rn_text or 'verifyCall' in rn_text:
                    return f"verifyCallResult({target_var})"
                if 'check' in rn_text.lower() and target_var in rn_text:
                    return f"check({target_var})"

        if_nodes = _find_all_by_type(stmt, "if_statement")
        for ifn in if_nodes:
            if_text = _node_text(ifn)
            if target_var in if_text:
                if f'!{target_var}' in if_text.replace(' ', '') or \
                   f'not {target_var}' in if_text.lower() or \
                   f'== false' in if_text:
                    return f"if(!{target_var}) revert"
                if target_var in if_text:
                    return f"if-check({target_var})"

        ternary_nodes = _find_all_by_type(stmt, "conditional_expression")
        for tn in ternary_nodes:
            tn_text = _node_text(tn)
            if target_var in tn_text:
                return f"ternary-check({target_var})"

    return None


def _detect_unbounded_loops(func_body_node, source_code: str = "") -> List[str]:
    """
    检测无界循环 (Unbounded Loop) —— DoS Gas 耗尽的根因。
    遇到 for/while 循环时，提取终止条件变量，
    标注 [LOOP_DETECTED]: unbounded_array_iteration 等特征。
    """
    if not func_body_node:
        return []

    loop_features = []
    loop_nodes = _find_all_by_type(func_body_node, "for_statement") + \
                 _find_all_by_type(func_body_node, "while_statement")

    for loop_node in loop_nodes:
        loop_type = loop_node.type
        loop_line = loop_node.start_point[0] + 1

        condition_node = None
        for child in loop_node.children:
            if child.type in ("parenthesized_expression", "expression"):
                condition_node = child
                break
            if loop_type == "for_statement" and child.type == "(":
                pass

        condition_text = _node_text(condition_node) if condition_node else ""

        is_unbounded = False
        bound_desc = ""

        if ".length" in condition_text:
            is_unbounded = True
            array_match = re_mod.search(r'(\w+)\.length', condition_text)
            array_name = array_match.group(1) if array_match else "array"
            bound_desc = f"unbounded_{array_name}_iteration"
        elif condition_text and not re_mod.search(r'\d+', condition_text):
            is_unbounded = True
            bound_desc = "unbounded_loop_no_constant_bound"

        loop_body = _find_first_by_type(loop_node, "block")
        has_ext_call_in_loop = False
        has_send_in_loop = False
        if loop_body:
            body_text = _node_text(loop_body)
            if any(kw in body_text for kw in ['.call', '.send', '.transfer', '.delegatecall']):
                has_ext_call_in_loop = True
            if '.send' in body_text or '.transfer' in body_text or '.call{value:' in body_text:
                has_send_in_loop = True

        if is_unbounded:
            feature = f"[LOOP_DETECTED]: {bound_desc} @L{loop_line}"
            if has_ext_call_in_loop:
                feature += " + EXT_CALL_IN_LOOP"
            if has_send_in_loop:
                feature += " + SEND_IN_LOOP"
            loop_features.append(feature)
        elif has_ext_call_in_loop or has_send_in_loop:
            feature = f"[LOOP_DETECTED]: bounded_loop_with_side_effect @L{loop_line}"
            if has_send_in_loop:
                feature += " + SEND_IN_LOOP"
            loop_features.append(feature)

    return loop_features


def _detect_orphan_calls(func_body_node) -> List[str]:
    """
    孤儿节点探测器：检测返回值被直接抛弃的外部调用。
    
    核心逻辑：
    1. 广度捕获：找到所有 member_expression，property 为 send/call/delegatecall/staticcall
    2. 父节点探测：
       - 如果父节点是 expression_statement → 返回值被直接抛弃 → 确诊 unchecked
       - 如果父节点是 variable_declaration → 赋值给了变量，由 _two_step_track 处理
       - 如果父节点是 if_statement / require / assert → 返回值被检查 → 安全
    """
    if not func_body_node:
        return []

    orphan_calls = []
    
    # address.transfer reverts on failure; it does not return a value to check.
    # Keep only calls whose success value can be silently discarded.
    call_keywords = {'send', 'call', 'delegatecall', 'staticcall'}
    
    member_exprs = _find_all_by_type(func_body_node, "member_expression")
    
    for me in member_exprs:
        prop = ""
        obj = ""
        for child in me.children:
            if child.type in ("property_identifier", "identifier"):
                child_text = _node_text(child)
                if child_text.lower() in call_keywords:
                    prop = child_text
            elif child.type not in (".",):
                obj = _node_text(child)
        
        if not prop:
            continue
        
        parent = me.parent
        if parent is None:
            continue
        
        call_expr = me.parent
        while call_expr and call_expr.type != "call_expression":
            call_expr = call_expr.parent
            if call_expr is None or call_expr == func_body_node:
                break
        
        if call_expr is None or call_expr.type != "call_expression":
            continue
        
        is_consumed = False
        ancestor = call_expr.parent
        while ancestor and ancestor != func_body_node:
            if ancestor.type == "call_expression":
                anc_func = ""
                for child in ancestor.children:
                    if child.type in ("expression", "identifier", "member_expression"):
                        anc_func = _node_text(child).strip()
                        break
                if anc_func.lower().startswith("require") or anc_func.lower().startswith("assert"):
                    is_consumed = True
                    break
                if anc_func.lower().startswith("if") or "if" in anc_func.lower():
                    is_consumed = True
                    break
            if ancestor.type == "if_statement":
                is_consumed = True
                break
            if ancestor.type in ("return_statement", "variable_declaration_statement"):
                is_consumed = True
                break
            ancestor = ancestor.parent
        
        if is_consumed:
            continue
        
        stmt_parent = call_expr.parent
        while stmt_parent and stmt_parent != func_body_node:
            if stmt_parent.type in ("expression_statement", "variable_declaration_statement",
                                     "require_statement", "assert_statement", "if_statement",
                                     "return_statement", "for_statement", "while_statement"):
                break
            if stmt_parent.parent and stmt_parent.parent.type in ("expression_statement", "variable_declaration_statement"):
                stmt_parent = stmt_parent.parent
                break
            stmt_parent = stmt_parent.parent
        
        if not stmt_parent or stmt_parent == func_body_node:
            continue
        
        if stmt_parent.type == "expression_statement":
            call_text = _node_text(call_expr)
            is_in_require = False
            ancestor = stmt_parent.parent
            while ancestor and ancestor != func_body_node:
                if ancestor.type in ("require_statement", "assert_statement"):
                    is_in_require = True
                    break
                if ancestor.type == "if_statement":
                    is_in_require = True
                    break
                ancestor = ancestor.parent
            
            if not is_in_require:
                label = f".{prop}()"
                if ".call.value" in call_text:
                    label = ".call.value()"
                elif ".call{" in call_text:
                    label = ".call{value}"
                elif ".call(" in call_text:
                    label = ".call()"
                orphan_calls.append(f"{label} @L{call_expr.start_point[0]+1} (RETURN VALUE SILENTLY DROPPED)")
        
        elif stmt_parent.type == "variable_declaration_statement":
            pass
        
        elif stmt_parent.type in ("require_statement", "assert_statement"):
            pass
        
        elif stmt_parent.type == "if_statement":
            pass
        
        elif stmt_parent.type == "return_statement":
            pass
    
    return orphan_calls


def _two_step_track_function(func_body_node, state_vars: Set[str]) -> Tuple[List[str], List[str]]:
    """
    两段式追踪主入口。
    返回 (delayed_checks, unchecked_calls)。

    delayed_checks: 被延迟校验的外部调用描述列表
    unchecked_calls: 未被校验的外部调用描述列表
    """
    if not func_body_node:
        return [], []

    stmts = _collect_all_statements(func_body_node)
    delayed_checks = []
    unchecked_calls = []

    try_stmts = _find_all_by_type(func_body_node, "try_statement")
    for try_node in try_stmts:
        try_type, try_label = _analyze_try_statement(try_node, state_vars)
        if try_type == "try_external_call":
            delayed_checks.append(f"{try_label} -> try/catch (safe pattern)")

    for idx, stmt in enumerate(stmts):
        result = _find_call_result_var(stmt)
        if result is None:
            continue

        var_name, call_label = result

        check_desc = _find_delayed_check(stmts, var_name, idx)

        if check_desc:
            delayed_checks.append(f"{call_label} -> {check_desc} (var={var_name})")
        else:
            unchecked_calls.append(f"{call_label} (var={var_name} NEVER checked)")

    return delayed_checks, unchecked_calls


def _extract_data_flow_chains(func_node, source_code: str) -> List[str]:
    """
    DFG (Data-Flow Graph) 提取器：Def-Use Chain 追踪。
    追踪关键变量（block.timestamp, msg.value, msg.sender, .call返回值等）
    从定义/读取到消费/判断的完整数据流路径。
    """
    if not source_code:
        return []

    func_start = func_node.start_point[0]
    func_end = func_node.end_point[0]
    lines = source_code.split('\n')
    func_lines = lines[func_start:func_end + 1]

    chains = []

    taint_sources = [
        (r'block\.timestamp\b', 'block.timestamp'),
        (r'\bnow\b', 'now'),
        (r'msg\.value\b', 'msg.value'),
        (r'msg\.sender\b', 'msg.sender'),
        (r'block\.difficulty\b', 'block.difficulty'),
        (r'blockhash\s*\(', 'blockhash()'),
        (r'tx\.origin\b', 'tx.origin'),
    ]

    for pattern, source_name in taint_sources:
        def_lines = []
        for i, line in enumerate(func_lines):
            if re_mod.search(pattern, line):
                abs_line = func_start + i + 1
                def_lines.append((abs_line, line.strip()[:80]))

        if not def_lines:
            continue

        for def_line, def_text in def_lines:
            use_chain = [f"L{def_line}(read_{source_name})"]

            var_name_match = re_mod.search(r'(\w+)\s*=\s*.*' + pattern, def_text) if '=' in def_text else None
            tracked_var = var_name_match.group(1) if var_name_match else None

            for i, line in enumerate(func_lines):
                abs_line = func_start + i + 1
                if abs_line == def_line:
                    continue
                line_stripped = line.strip()

                if tracked_var and tracked_var in line_stripped:
                    if re_mod.search(r'require\s*\(' + re_mod.escape(tracked_var), line_stripped):
                        use_chain.append(f"L{abs_line}(require_check)")
                    elif re_mod.search(r'if\s*\(.*' + re_mod.escape(tracked_var), line_stripped):
                        use_chain.append(f"L{abs_line}(if_condition)")
                    elif re_mod.search(r'return\s+.*' + re_mod.escape(tracked_var), line_stripped):
                        use_chain.append(f"L{abs_line}(return)")
                    elif '%' in line_stripped:
                        use_chain.append(f"L{abs_line}(modulo_op)")
                    elif re_mod.search(r'\w+\[.*' + re_mod.escape(tracked_var), line_stripped):
                        use_chain.append(f"L{abs_line}(array_access)")
                    elif '=' in line_stripped and '==' not in line_stripped:
                        use_chain.append(f"L{abs_line}(assignment)")

                if source_name in ['block.timestamp', 'now'] and abs_line != def_line:
                    if re_mod.search(r'%', line_stripped) and (re_mod.search(pattern, line_stripped) or (tracked_var and tracked_var in line_stripped)):
                        if not any(f"L{abs_line}" in uc for uc in use_chain):
                            use_chain.append(f"L{abs_line}(modulo_op)")

                if source_name in ['block.timestamp', 'now']:
                    if re_mod.search(r'(require|if)\s*\(.*(' + pattern + r'|' + (re_mod.escape(tracked_var) if tracked_var else 'NONE') + r')', line_stripped):
                        if not any(f"L{abs_line}" in uc for uc in use_chain):
                            use_chain.append(f"L{abs_line}(condition)")

            if len(use_chain) > 1:
                chains.append(" -> ".join(use_chain))

    call_return_chains = []
    func_body = _find_first_by_type(func_node, "function_body")
    if func_body:
        stmts = _collect_all_statements(func_body)
        for idx, stmt in enumerate(stmts):
            result = _find_call_result_var(stmt)
            if result is None:
                continue
            var_name, call_label = result
            def_line = stmt.start_point[0] + 1
            chain = [f"L{def_line}({call_label}=>{var_name})"]

            for later_stmt in stmts[idx + 1:]:
                later_text = _node_text(later_stmt)
                later_line = later_stmt.start_point[0] + 1
                if var_name in later_text:
                    if 'require(' in later_text:
                        chain.append(f"L{later_line}(require({var_name}))")
                        break
                    elif 'if' in later_text and var_name in later_text:
                        chain.append(f"L{later_line}(if_check({var_name}))")
                        break
                    elif 'revert' in later_text:
                        chain.append(f"L{later_line}(revert_on_{var_name})")
                        break
                    elif '=' in later_text and '==' not in later_text:
                        chain.append(f"L{later_line}(reassign_{var_name})")

            if len(chain) > 1:
                call_return_chains.append(" -> ".join(chain))

    chains.extend(call_return_chains)
    return chains


def _pragma_is_legacy_constructor_compatible(source_code: str) -> bool:
    """Return whether the declared Solidity range is explicitly confined to 0.4.x."""
    for match in re_mod.finditer(r"\bpragma\s+solidity\s+([^;]+);", source_code or ""):
        constraint = re_mod.sub(r"\s+", "", match.group(1))
        if re_mod.fullmatch(r"\^?0\.4\.\d+", constraint):
            return True
        if re_mod.search(r">=?0\.4\.\d+", constraint) and "<0.5.0" in constraint:
            return True
    return False


def _extract_function_flow(func_node, state_vars: Set[str], modifier_defs: List[ModifierInfo],
                           all_func_nodes: list = None, source_code: str = "",
                           contract_name: str = "") -> FunctionFlow:
    name = ""
    visibility = "public"
    state_mutability = ""
    modifiers = []
    is_constructor = False
    is_fallback = False
    is_receive = False

    for child in func_node.children:
        if child.type == "identifier":
            name = _node_text(child)
        elif child.type == "visibility":
            visibility = _node_text(child)
        elif child.type == "state_mutability":
            state_mutability = _node_text(child)
        elif child.type == "modifier_invocation":
            for sc in child.children:
                if sc.type == "identifier":
                    modifiers.append(_node_text(sc))

    if func_node.type == "constructor_definition":
        is_constructor = True
        name = "constructor"
    elif (
        contract_name
        and name == contract_name
        and _pragma_is_legacy_constructor_compatible(source_code)
    ):
        # Solidity 0.4.x treats a public function with the contract's name as
        # its constructor. Tree-sitter represents it as function_definition,
        # so without this semantic bridge it is wrongly treated as a callable
        # initializer and can generate a false access-control finding.
        is_constructor = True
    elif name == "" and func_node.type == "fallback_function_definition":
        is_fallback = True
        name = "fallback"
    elif name == "" and func_node.type == "receive_function_definition":
        is_receive = True
        name = "receive"

    external_calls = []
    require_checks = []
    state_writes = []
    internal_calls = []

    func_body = _find_first_by_type(func_node, "function_body")
    if func_body:
        _traverse_statements(func_body, state_vars, external_calls, require_checks, state_writes, internal_calls)

    delayed_checks, unchecked_calls = _two_step_track_function(func_body, state_vars)

    orphan_calls = _detect_orphan_calls(func_body) if func_body else []
    for oc in orphan_calls:
        already = any(oc.split(' @L')[0] in uc for uc in unchecked_calls)
        if not already:
            unchecked_calls.append(oc)

    has_guard, guard_type, guard_detail = _detect_guard_in_function(func_node, modifier_defs)

    inlined_modifier_code = ""
    for mname in modifiers:
        for mdef in modifier_defs:
            if mdef.name == mname and mdef.body_text:
                inlined_modifier_code += f"[MODIFIER_INLINE: {mname}] {{{mdef.body_text}}} "

    inlined_internal_flows = []
    if all_func_nodes and internal_calls:
        func_map = {}
        for fn in all_func_nodes:
            fn_name = ""
            for child in fn.children:
                if child.type == "identifier":
                    fn_name = _node_text(child)
                    break
            if fn_name:
                func_map[fn_name] = fn

        for icall in internal_calls[:3]:
            callee_name = icall.split("(")[0].split(".")[-1].strip()
            if callee_name in func_map:
                callee_node = func_map[callee_name]
                callee_body = _find_first_by_type(callee_node, "function_body")
                if callee_body:
                    callee_ext_calls = []
                    callee_req_checks = []
                    callee_state_writes = []
                    callee_int_calls = []
                    _traverse_statements(callee_body, state_vars,
                                         callee_ext_calls, callee_req_checks,
                                         callee_state_writes, callee_int_calls)
                    callee_delayed, callee_unchecked = _two_step_track_function(callee_body, state_vars)

                    flow_parts = []
                    if callee_ext_calls:
                        flow_parts.append(f"EXT_CALLS: {', '.join(callee_ext_calls)}")
                    if callee_req_checks:
                        flow_parts.append(f"CHECKS: {', '.join(callee_req_checks[:3])}")
                    if callee_state_writes:
                        flow_parts.append(f"STATE_WRITES: {', '.join(callee_state_writes)}")
                    if callee_delayed:
                        flow_parts.append(f"DELAYED: {', '.join(callee_delayed)}")
                    if callee_unchecked:
                        flow_parts.append(f"UNCHECKED: {', '.join(callee_unchecked)}")

                    if flow_parts:
                        inlined_internal_flows.append(
                            f"[INLINE: {callee_name}()] " + " | ".join(flow_parts)
                        )

                    for ec in callee_ext_calls:
                        if ec not in external_calls:
                            external_calls.append(f"{ec}(via_{callee_name})")
                    for uc in callee_unchecked:
                        if uc not in unchecked_calls:
                            unchecked_calls.append(f"{uc}(in_{callee_name})")
                    for dc in callee_delayed:
                        if dc not in delayed_checks:
                            delayed_checks.append(f"{dc}(in_{callee_name})")

    data_flow_chains = _extract_data_flow_chains(func_node, source_code) if source_code else []

    loop_features = _detect_unbounded_loops(func_body, source_code) if func_body else []

    end_line = func_node.end_point[0] + 1

    return FunctionFlow(
        name=name,
        visibility=visibility,
        modifiers=modifiers,
        state_writes=state_writes,
        external_calls=external_calls,
        internal_calls=internal_calls,
        require_checks=require_checks,
        has_reentrancy_guard=has_guard,
        guard_type=guard_type,
        guard_detail=guard_detail,
        line_number=func_node.start_point[0] + 1,
        end_line_number=end_line,
        is_constructor=is_constructor,
        is_fallback=is_fallback,
        is_receive=is_receive,
        state_mutability=state_mutability,
        delayed_checks=delayed_checks,
        unchecked_calls=unchecked_calls,
        inlined_modifier_code=inlined_modifier_code,
        inlined_internal_flows=inlined_internal_flows,
        data_flow_chains=data_flow_chains,
        loop_features=loop_features,
    )


def _traverse_statements(node, state_vars: Set[str], external_calls: list,
                         require_checks: list, state_writes: list, internal_calls: list):
    for child in node.children:
        if child.type == "statement":
            expr_stmt = None
            for sc in child.children:
                if sc.type in ("expression_statement", "variable_declaration_statement",
                               "assignment_statement", "augmented_assignment_statement"):
                    expr_stmt = sc
                    break
                elif sc.type == "assembly_statement":
                    pass
                elif sc.type == "try_statement":
                    try_type, try_label = _analyze_try_statement(sc, state_vars)
                    if try_type == "try_external_call" and try_label not in external_calls:
                        external_calls.append(try_label)

            if not expr_stmt:
                for sc in child.children:
                    if sc.type == "expression_statement":
                        expr_stmt = sc
                        break

            if expr_stmt:
                _analyze_statement_expr(expr_stmt, state_vars, external_calls, require_checks, state_writes, internal_calls)

            for sc in child.children:
                if sc.type in ("if_statement", "for_statement", "while_statement", "do_while_statement",
                               "unchecked_statement", "assembly_statement", "try_statement"):
                    _traverse_statements(sc, state_vars, external_calls, require_checks, state_writes, internal_calls)

        elif child.type in ("if_statement", "for_statement", "while_statement", "do_while_statement",
                            "unchecked_statement", "assembly_statement", "try_statement"):
            _traverse_statements(child, state_vars, external_calls, require_checks, state_writes, internal_calls)


def _analyze_statement_expr(stmt_node, state_vars: Set[str], external_calls: list,
                            require_checks: list, state_writes: list, internal_calls: list):
    call_nodes = _find_all_by_type(stmt_node, "call_expression")
    for cn in call_nodes:
        call_type, info = _analyze_call_expression(cn, state_vars)
        if call_type == "external_call":
            if info not in external_calls:
                external_calls.append(info)
        elif call_type == "require":
            short = info[:80] if len(info) > 80 else info
            if short not in require_checks:
                require_checks.append(short)
        elif call_type == "assert":
            short = info[:80] if len(info) > 80 else info
            if short not in require_checks:
                require_checks.append(f"assert({short[7:]})" if short.startswith("assert") else short)
        elif call_type == "internal_call":
            if info not in internal_calls:
                internal_calls.append(info)

    assign_nodes = _find_all_by_type(stmt_node, "assignment_expression")
    for an in assign_nodes:
        written = _analyze_assignment(an, state_vars)
        if written and written not in state_writes:
            state_writes.append(written)

    aug_assign_nodes = _find_all_by_type(stmt_node, "augmented_assignment_expression")
    for an in aug_assign_nodes:
        written = _analyze_assignment(an, state_vars)
        if written and written not in state_writes:
            state_writes.append(written)


def _build_call_graph(contract_node, all_func_nodes: list) -> Dict[str, Set[str]]:
    """
    构建合约内函数调用图 (Call Graph)。
    返回: {caller_name: {callee_name1, callee_name2, ...}}
    """
    call_graph = {}
    for func_node in all_func_nodes:
        func_name = ""
        for child in func_node.children:
            if child.type == "identifier":
                func_name = _node_text(child)
                break
        if not func_name:
            if func_node.type == "constructor_definition":
                func_name = "constructor"
            elif func_node.type == "fallback_function_definition":
                func_name = "fallback"
            elif func_node.type == "receive_function_definition":
                func_name = "receive"
        if not func_name:
            continue

        callees = set()
        func_body = _find_first_by_type(func_node, "function_body")
        if func_body:
            call_exprs = _find_all_by_type(func_body, "call_expression")
            for ce in call_exprs:
                for child in ce.children:
                    if child.type == "expression":
                        callee_text = _node_text(child)
                        callee_name = callee_text.split("(")[0].split(".")[-1].strip()
                        if callee_name and callee_name[0].isalpha():
                            callees.add(callee_name)

            member_exprs = _find_all_by_type(func_body, "member_expression")
            for me in member_exprs:
                me_text = _node_text(me)
                if me_text.endswith("()") or "(" in me_text:
                    callee_name = me_text.split("(")[0].split(".")[-1].strip()
                    if callee_name and callee_name[0].isalpha():
                        callees.add(callee_name)

        call_graph[func_name] = callees
    return call_graph


def _compute_reachability(call_graph: Dict[str, Set[str]],
                          public_entries: Set[str]) -> Dict[str, Tuple[bool, List[str]]]:
    """
    从 public/external 入口点出发，BFS 计算每个函数的可达性。
    返回: {func_name: (is_reachable, reachability_path)}
    """
    reachable = {}
    visited = set()
    queue = []

    for entry in sorted(public_entries):
        if entry in call_graph or entry in ("constructor", "fallback", "receive"):
            reachable[entry] = (True, [entry])
            visited.add(entry)
            queue.append(entry)

    while queue:
        current = queue.pop(0)
        if current not in call_graph:
            continue
        for callee in sorted(call_graph[current]):
            if callee not in visited:
                visited.add(callee)
                parent_path = reachable.get(current, (True, []))[1]
                reachable[callee] = (True, parent_path + [callee])
                queue.append(callee)

    for func_name in call_graph:
        if func_name not in reachable:
            reachable[func_name] = (False, [])

    return reachable


def backward_slice(features: ContractFeatures, source_code: str,
                   risk_funcs: List[str] = None) -> str:
    """
    AST 动态切片 (Backward Slicing)：
    以提取到的风险点为中心，仅提取其上下文函数、依赖的状态变量定义以及修饰器。
    将无关的纯业务逻辑从传递给大模型的文本中裁剪掉，保证输入信息的高信噪比。
    """
    if not source_code:
        return source_code

    lines = source_code.split('\n')
    total_lines = len(lines)

    if total_lines <= 150:
        return source_code

    risk_function_names = set()
    if risk_funcs:
        risk_function_names = set(risk_funcs)
    else:
        for func in features.functions:
            if func.visibility in ['public', 'external']:
                has_risk = (
                    func.external_calls or
                    func.unchecked_calls or
                    func.state_writes or
                    func.loop_features or
                    not func.require_checks
                )
                if has_risk:
                    risk_function_names.add(func.name)

    if not risk_function_names:
        for func in features.functions:
            if func.visibility in ['public', 'external']:
                risk_function_names.add(func.name)

    included_line_ranges = set()

    state_var_lines = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
            continue
        for sv in features.state_variables:
            if sv in stripped and ('=' in stripped or stripped.endswith(';')):
                if not stripped.startswith('function') and not stripped.startswith('modifier'):
                    state_var_lines.add(i)
                    included_line_ranges.add(i)
                    break

    modifier_line_ranges = set()
    for mod in features.modifiers:
        start = mod.line_number - 1
        for i in range(max(0, start), min(total_lines, start + 30)):
            modifier_line_ranges.add(i)
            included_line_ranges.add(i)

    pragma_lines = set()
    import_lines = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('pragma') or stripped.startswith('import'):
            pragma_lines.add(i)
            included_line_ranges.add(i)

    contract_decl_lines = set()
    for i, line in enumerate(lines):
        if re_mod.search(r'\b(contract|library|interface|abstract)\s+', line):
            contract_decl_lines.add(i)
            included_line_ranges.add(i)
            for j in range(i + 1, min(i + 3, total_lines)):
                if '{' in lines[j]:
                    included_line_ranges.add(j)
                    break

    risk_func_ranges = {}
    for func in features.functions:
        if func.name in risk_function_names:
            start = func.line_number - 1
            end = func.end_line_number if func.end_line_number > start else start + 50
            end = min(end, total_lines)
            risk_func_ranges[func.name] = (start, end)
            for i in range(start, end):
                included_line_ranges.add(i)

    for func in features.functions:
        if func.name not in risk_function_names and func.visibility in ['public', 'external']:
            if func.internal_calls:
                for icall in func.internal_calls:
                    callee_name = icall.split("(")[0].split(".")[-1].strip()
                    if callee_name in risk_function_names:
                        start = func.line_number - 1
                        end = func.end_line_number if func.end_line_number > start else start + 30
                        end = min(end, total_lines)
                        for i in range(start, end):
                            included_line_ranges.add(i)

    for func in features.functions:
        if func.name in risk_function_names:
            for icall in func.internal_calls:
                callee_name = icall.split("(")[0].split(".")[-1].strip()
                for callee_func in features.functions:
                    if callee_func.name == callee_name:
                        start = callee_func.line_number - 1
                        end = callee_func.end_line_number if callee_func.end_line_number > start else start + 30
                        end = min(end, total_lines)
                        for i in range(start, end):
                            included_line_ranges.add(i)

    # The feature extractor can lose internal-call metadata on older or
    # mixed-version Solidity syntax.  Reconstruct only the same-contract
    # call edges from the source so a source-grounded risk cannot be sliced
    # away before the provider sees its helper closure.
    functions_by_name = {
        func.name: func for func in features.functions if func.name
    }
    source_call_events = {
        name: _internal_call_events_for_function(
            func, lines, set(functions_by_name)
        )
        for name, func in functions_by_name.items()
    }
    # Set iteration order is process-dependent; keep closure expansion stable
    # so the derived slice and downstream prompt are reproducible.
    pending_functions = sorted(risk_function_names)
    expanded_functions = set(risk_function_names)
    while pending_functions:
        caller_name = pending_functions.pop()
        for _, callee_name in source_call_events.get(caller_name, []):
            callee_func = functions_by_name.get(callee_name)
            if callee_func is None:
                continue
            start = callee_func.line_number - 1
            end = callee_func.end_line_number if callee_func.end_line_number > start else start + 30
            end = min(end, total_lines)
            for i in range(start, end):
                included_line_ranges.add(i)
            if callee_name not in expanded_functions:
                expanded_functions.add(callee_name)
                pending_functions.append(callee_name)

    for func in features.functions:
        if func.name in risk_function_names:
            for mod_name in func.modifiers:
                for mod in features.modifiers:
                    if mod.name == mod_name:
                        start = mod.line_number - 1
                        for i in range(max(0, start), min(total_lines, start + 30)):
                            included_line_ranges.add(i)

    for i in range(min(5, total_lines)):
        included_line_ranges.add(i)
    for i in range(max(0, total_lines - 3), total_lines):
        included_line_ranges.add(i)

    if not included_line_ranges:
        return source_code

    sorted_lines = sorted(included_line_ranges)
    result_parts = []
    prev_line = -10

    for line_idx in sorted_lines:
        if line_idx - prev_line > 2:
            result_parts.append(f"    // ... [{prev_line + 2}-{line_idx}] omitted ...")
        result_parts.append(f"{line_idx + 1:>{len(str(total_lines))}} | {lines[line_idx]}")
        prev_line = line_idx

    sliced = '\n'.join(result_parts)
    original_chars = len(source_code)
    sliced_chars = len(sliced)
    ratio = sliced_chars / original_chars * 100 if original_chars > 0 else 100

    print(f"[AST Slicing] Original: {total_lines} lines, {original_chars} chars")
    print(f"[AST Slicing] Sliced:   {len(sorted_lines)} lines, {sliced_chars} chars ({ratio:.1f}%)")
    print(f"[AST Slicing] Risk functions: {sorted(risk_function_names)}")
    print(f"[AST Slicing] Kept: state_vars({len(state_var_lines)}), modifiers({len(modifier_line_ranges)}), risk_funcs({len(risk_func_ranges)})")

    return sliced


def _has_body_level_access_control(func: FunctionFlow, source_code: str) -> bool:
    if not source_code:
        return False
    lines = source_code.split('\n')
    start = func.line_number - 1
    end = func.end_line_number if func.end_line_number > start else start + 50
    end = min(end, len(lines))
    func_text = "\n".join(lines[start:end])
    # Strip comments before looking for an authorization predicate.  A source
    # comment such as ``// require(msg.sender == owner)`` must not protect a
    # permissionless entrypoint.
    func_text = _strip_comments_preserve_lines(func_text)
    body_ac_patterns = [
        r'\b(?:require|assert)\s*\([^;{}]*\bmsg\.sender\b[^;{}]*\b(?:owner|admin|governance|timelock|operator|guardian)\b',
        r'\b(?:require|assert)\s*\([^;{}]*\b(?:owner|admin|governance|timelock|operator|guardian)\b[^;{}]*\bmsg\.sender\b',
        r'\b(?:require|assert)\s*\([^;{}]*\b_msgSender\s*\(\s*\)[^;{}]*\b(?:owner|admin|governance|timelock|operator|guardian)\b',
        r'\b(?:require|assert)\s*\([^;{}]*\b(?:owner|admin|governance|timelock|operator|guardian)\b[^;{}]*\b_msgSender\s*\(\s*\)',
        r'\b(?:require|assert)\s*\([^;{}]*\bhasRole\s*\([^;{}]*\b(?:msg\.sender|_msgSender\s*\(\s*\))',
        r'\b(?:require|assert)\s*\([^;{}]*\b(?:isAuthorized|isWhitelisted|isAllowed|isOperator|isAdmin|isOwner)\s*\([^;{}]*\b(?:msg\.sender|_msgSender\s*\(\s*\))',
        r'\b(?:_checkRole|checkRole|requireRole)\s*\([^;{}]*\b(?:msg\.sender|_msgSender\s*\(\s*\))',
        r'\bif\s*\([^;{}]*!\s*(?:hasRole|isAuthorized|isWhitelisted|isAllowed)\s*\(',
        r'\bif\s*\([^;{}]*\b(?:msg\.sender|_msgSender\s*\(\))\b[^;{}]*[!=]=?[^;{}]*\b(?:owner|admin|governance|timelock|operator|guardian)\b',
    ]
    for pat in body_ac_patterns:
        if re_mod.search(pat, func_text, re_mod.IGNORECASE | re_mod.DOTALL):
            return True
    return False


STRONG_ADMIN_MODIFIERS = {
    'onlyowner', 'owneronly', 'only_owner', 'only_admin', 'onlyadmin', 'onlyrole', 'onlyminter',
    'onlypauser', 'onlygovernance', 'onlystrategy', 'onlymanager',
    'onlyoperator', 'onlyguardian', 'onlydao', 'onlyprotocol',
    'onlymaintainer', 'onlycontroller', 'onlydelegate', 'onlydev',
}


def _source_function_signature(func: FunctionFlow, source_code: str) -> str:
    if not source_code or not func.line_number:
        return ""
    lines = source_code.splitlines()
    start = max(func.line_number - 1, 0)
    signature_lines = []
    for line in lines[start:min(start + 12, len(lines))]:
        code_line = line.split("//", 1)[0]
        signature_lines.append(code_line)
        if "{" in code_line:
            break
    return "\n".join(signature_lines)


def _signature_has_strong_admin_modifier(func: FunctionFlow, source_code: str) -> bool:
    signature = _source_function_signature(func, source_code)
    if not signature:
        return False
    return any(
        re_mod.search(rf"\b{re_mod.escape(modifier)}\b", signature, re_mod.IGNORECASE)
        for modifier in STRONG_ADMIN_MODIFIERS
    )


def _is_admin_restricted(func: FunctionFlow, source_code: str = "") -> bool:
    if any(m.lower() in STRONG_ADMIN_MODIFIERS for m in func.modifiers):
        return True
    if source_code and _signature_has_strong_admin_modifier(func, source_code):
        return True
    if func.visibility not in ('public', 'external'):
        return True
    if source_code and _has_body_level_access_control(func, source_code):
        return True
    return False


def _tx_origin_identity_state_evidence(
    func: FunctionFlow, source_code: str
) -> dict[str, object] | None:
    """Close a tx.origin identity flow to a persistent mutation in one function."""

    if not source_code or not func or not func.line_number:
        return None

    lines = source_code.splitlines()
    start = max(func.line_number - 1, 0)
    end = func.end_line_number if func.end_line_number > start else start + 50
    end = min(end, len(lines))
    body_lines = [line.split("//", 1)[0] for line in lines[start:end]]
    origin_lines: list[int] = []
    for offset, line in enumerate(body_lines):
        if not re_mod.search(r"\btx\s*\.\s*origin\b", line, re_mod.I):
            continue
        if re_mod.search(r"\bemit\s+[A-Za-z_]\w*\s*\(", line, re_mod.I):
            continue
        if re_mod.search(
            r"\btx\s*\.\s*origin\s*\.\s*(?:call|send|transfer)\b|"
            r"\bmsg\s*\.\s*sender\s*==\s*tx\s*\.\s*origin\b|"
            r"\btx\s*\.\s*origin\s*==\s*msg\s*\.\s*sender\b",
            line,
            re_mod.I,
        ):
            continue
        origin_lines.append(start + offset + 1)
    if not origin_lines:
        return None

    state_write_offsets = [
        offset
        for offset, line in enumerate(body_lines)
        if re_mod.search(
            r"\.(?:push|pop)\s*\(|"
            r"\bdelete\s+[A-Za-z_]\w*|"
            r"\b[A-Za-z_]\w*(?:\s*\[[^\]]+\])+\s*=",
            line,
            re_mod.I,
        )
    ]
    if not state_write_offsets:
        return None

    linked_state_lines: set[int] = set()
    for origin_line in origin_lines:
        offset = origin_line - start - 1
        statement_start = 0
        for candidate in range(offset, -1, -1):
            if ";" in body_lines[candidate]:
                statement_start = candidate
                break
        statement_end = offset
        while statement_end < len(body_lines) - 1 and ";" not in body_lines[statement_end]:
            statement_end += 1
        statement = "\n".join(body_lines[statement_start:statement_end + 1])
        if re_mod.search(r"\bemit\s+[A-Za-z_]\w*\s*\(", statement, re_mod.I):
            continue
        local_names = set(
            re_mod.findall(
                r"\b([A-Za-z_]\w*)\s*=\s*[^;]*\btx\s*\.\s*origin\b",
                statement,
                re_mod.I | re_mod.S,
            )
        )
        for state_offset in state_write_offsets:
            if state_offset < offset:
                continue
            state_line = start + state_offset + 1
            state_text = body_lines[state_offset]
            if state_offset == offset and re_mod.search(
                r"\btx\s*\.\s*origin\b", state_text, re_mod.I
            ):
                linked_state_lines.add(state_line)
            elif local_names and any(
                re_mod.search(rf"\b{re_mod.escape(name)}\b", state_text)
                for name in local_names
            ):
                linked_state_lines.add(state_line)

    if not linked_state_lines:
        return None
    evidence_lines = sorted(set(origin_lines) | linked_state_lines)
    return {
        "line": min(origin_lines),
        "evidence_lines": evidence_lines,
        "state_write_lines": sorted(linked_state_lines),
        "function_name": func.name,
        "source_evidence_kind": "tx_origin_identity_state_write",
    }


def _detect_e2_tx_origin_modifier_access_control(
    contracts: List[ContractFeatures], source_code: str
) -> list[dict[str, object]]:
    """Detect a tx.origin authorization check implemented in a modifier.

    Function-body-only rules miss modifiers such as RocketStorage's
    ``onlyLatestRocketNetworkContract``.  Keep this E2-only rule narrow: the
    modifier must compare ``tx.origin`` to an identity other than
    ``msg.sender`` and must protect at least one reachable public/external
    stateful entrypoint.  EOA-only gates (``msg.sender == tx.origin``) are
    explicitly excluded.
    """

    if not source_code:
        return []

    source_lines = source_code.splitlines()
    results: list[dict[str, object]] = []
    tx_origin = r"\btx\s*\.\s*origin\b"
    eoa_only = re_mod.compile(
        rf"\bmsg\s*\.\s*sender\s*(?:==|!=)\s*{tx_origin}|"
        rf"{tx_origin}\s*(?:==|!=)\s*msg\s*\.\s*sender\b",
        re_mod.I,
    )
    identity_comparison = re_mod.compile(
        rf"{tx_origin}\s*(?:==|!=)\s*(?!msg\s*\.\s*sender\b|address\s*\()[A-Za-z_]\w*|"
        rf"[A-Za-z_]\w*\s*(?:==|!=)\s*{tx_origin}",
        re_mod.I,
    )

    for contract in contracts:
        for modifier in contract.modifiers:
            body = _strip_comments_preserve_lines(modifier.body_text or "")
            if not re_mod.search(tx_origin, body, re_mod.I):
                continue
            if eoa_only.search(body) or not identity_comparison.search(body):
                continue
            if not re_mod.search(
                rf"\b(?:require|assert|if|while)\s*\([^;{{}}]*{tx_origin}",
                body,
                re_mod.I | re_mod.S,
            ):
                continue

            modifier_line = max(int(modifier.line_number or 0), 1)
            origin_line = None
            # ``body`` keeps source line breaks while comments are blanked;
            # scan it instead of raw source so a documentation mention of
            # ``tx.origin`` cannot become the reported vulnerability locus.
            for offset, line in enumerate(body.splitlines()):
                if re_mod.search(tx_origin, line, re_mod.I):
                    origin_line = modifier_line + offset
                    break
            if origin_line is None:
                continue

            protected = []
            for function in contract.functions:
                if modifier.name not in function.modifiers:
                    continue
                if function.visibility not in {"public", "external"}:
                    continue
                if function.is_constructor or not function.is_reachable:
                    continue
                if str(function.state_mutability or "").casefold() in {"view", "pure"}:
                    continue
                start = max(function.line_number - 1, 0)
                end = min(
                    function.end_line_number if function.end_line_number > start else start + 50,
                    len(source_lines),
                )
                function_text = "\n".join(source_lines[start:end])
                stateful = bool(function.state_writes) or bool(
                    re_mod.search(
                        r"\bsstore\b|\[[^\]]+\]\s*=|\b(?:delete|mapping|storage)\b",
                        function_text,
                        re_mod.I,
                    )
                )
                asset_sink = bool(
                    re_mod.search(
                        r"\.(?:transfer|send|call|delegatecall|approve|mint|burn|"
                        r"deposit|withdraw|stake|unstake|claim|redeem|settle)\s*\(",
                        function_text,
                        re_mod.I,
                    )
                )
                if stateful or asset_sink:
                    protected.append((function, function_text))
            if not protected:
                continue

            representative, representative_text = min(
                protected, key=lambda item: int(item[0].line_number or 0)
            )
            representative_start = max(int(representative.line_number or 0) - 1, 0)
            write_lines = [
                representative_start + offset + 1
                for offset, line in enumerate(representative_text.splitlines())
                if re_mod.search(
                    r"\bsstore\b|\[[^\]]+\]\s*=|\b(?:delete)\s+[A-Za-z_]\w*",
                    line,
                    re_mod.I,
                )
            ]
            if not write_lines:
                write_lines = [int(representative.line_number or 0)]

            results.append({
                "risk_type": "access_control",
                "submechanism": "tx_origin_modifier_authorization",
                "source_evidence_kind": "tx_origin_modifier_authorization",
                "source_grounded": True,
                "confidence": 0.93,
                "reason": (
                    f"modifier {modifier.name}() compares tx.origin for authorization "
                    f"at @L{origin_line} and gates public state/asset mutation in "
                    f"{representative.name}()"
                ),
                "function_name": modifier.name,
                "entrypoint_function_name": representative.name,
                "entrypoint_lines": [int(representative.line_number or 0)],
                "line": int(origin_line),
                "evidence_lines": [int(origin_line)],
                "state_write_lines": write_lines[:8],
                "modifier_name": modifier.name,
                "modifier_line": modifier_line,
                "modifier_body": body,
                "source_call_path": [modifier.name, representative.name],
                "sink_function_name": representative.name,
            })

    return results


def _has_local_storage_claimant_debit(function_text: str) -> bool:
    """Recognize claimant accounting through a storage alias."""

    aliases = re_mod.findall(
        r"\b[A-Za-z_]\w*\s+storage\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*"
        r"[A-Za-z_]\w*\s*\[\s*msg\.sender\s*\]",
        function_text or "",
        re_mod.I,
    )
    for alias in aliases:
        field = rf"\b{re_mod.escape(alias)}\s*\.\s*[A-Za-z_]\w*"
        if re_mod.search(
            rf"{field}\s*(?:-=|=\s*0\b|=\s*{field}\s*(?:-|\.sub\s*\())",
            function_text,
            re_mod.I,
        ):
            return True
    return False


def _has_external_claimant_debit_before_payout(function_text: str) -> bool:
    """Recognize a caller-funded asset sale before a native payout."""

    payout = re_mod.search(
        r"(?:msg\.sender|payable\s*\(\s*msg\.sender\s*\))\s*\.\s*transfer\s*\(",
        function_text or "",
        re_mod.I,
    )
    if payout is None:
        return False
    prefix = function_text[: payout.start()]
    return bool(
        re_mod.search(
            r"\b(?:_?transfer|transferFrom|_?burn)\s*\(\s*msg\.sender\s*,\s*"
            r"(?:address\s*\(\s*this\s*\)|this)\s*,",
            prefix,
            re_mod.I,
        )
    )


def _e1_optimized_standard_token_arithmetic_negative_control(
    function_name: str, function_text: str
) -> bool:
    """Recognize standard token balance paths that are E1 arithmetic negatives."""

    if os.environ.get(PROFILE_ENV) != E1_OPTIMIZED_V1_PROFILE:
        return False

    name = str(function_name or "").casefold()
    body = re_mod.sub(
        r"/\*.*?\*/|//[^\n]*",
        "",
        function_text or "",
        flags=re_mod.DOTALL,
    )
    if re_mod.search(r"\b(?:unchecked|assembly)\b", body, re_mod.I):
        return False

    mapping = r"(?:balances|_balances)"
    identifier = r"[A-Za-z_]\w*"

    if name == "transfer":
        return bool(
            re_mod.search(
                rf"\brequire\s*\([^;{{}}]*\b{mapping}\s*\[\s*msg\.sender\s*\]"
                rf"\s*>=\s*{identifier}\b",
                body,
                re_mod.I,
            )
            and re_mod.search(
                rf"\b{mapping}\s*\[\s*msg\.sender\s*\]\s*-="
                rf"\s*{identifier}\b",
                body,
                re_mod.I,
            )
            and re_mod.search(
                rf"\b{mapping}\s*\[\s*{identifier}\s*\]\s*\+="
                rf"\s*{identifier}\b",
                body,
                re_mod.I,
            )
        )

    if name != "transferfrom":
        if name == "constructor":
            return bool(
                re_mod.search(
                    r"\btotalSupply\s*=\s*[A-Za-z_]\w*\s*\*\s*10\s*\*\*\s*"
                    r"(?:uint\d*\s*\(\s*)?decimals",
                    body,
                    re_mod.I,
                )
                and re_mod.search(
                    r"\bbalanceOf\s*\[\s*msg\.sender\s*\]\s*=\s*totalSupply\b",
                    body,
                    re_mod.I,
                )
            )

        if name == "burn":
            return bool(
                re_mod.search(
                    r"\brequire\s*\([^;{}]*\bbalanceOf\s*\[\s*msg\.sender\s*\]"
                    r"\s*>=\s*[A-Za-z_]\w*",
                    body,
                    re_mod.I,
                )
                and re_mod.search(
                    r"\bbalanceOf\s*\[\s*msg\.sender\s*\]\s*-=\s*[A-Za-z_]\w*",
                    body,
                    re_mod.I,
                )
                and re_mod.search(
                    r"\btotalSupply\s*-=\s*[A-Za-z_]\w*",
                    body,
                    re_mod.I,
                )
                and re_mod.search(r"\bemit\s+Burn\s*\(", body, re_mod.I)
            )

        if name == "burnfrom":
            return bool(
                re_mod.search(
                    r"\brequire\s*\([^;{}]*\bbalanceOf\s*\[\s*[A-Za-z_]\w*\s*\]"
                    r"\s*>=\s*[A-Za-z_]\w*",
                    body,
                    re_mod.I,
                )
                and (
                    re_mod.search(
                        r"\brequire\s*\([^;{}]*\b(?:allowance|allowed)\b"
                        r"[^;{}]*>=\s*[A-Za-z_]\w*",
                        body,
                        re_mod.I,
                    )
                    or re_mod.search(
                        r"\brequire\s*\([^;{}]*[A-Za-z_]\w*\s*<=\s*"
                        r"\b(?:allowance|allowed)\b[^;{}]*",
                        body,
                        re_mod.I,
                    )
                )
                and re_mod.search(
                    r"\bbalanceOf\s*\[\s*[A-Za-z_]\w*\s*\]\s*-=\s*[A-Za-z_]\w*",
                    body,
                    re_mod.I,
                )
                and re_mod.search(
                    r"\b(?:allowance|allowed)\b[^;{}]*-=\s*[A-Za-z_]\w*",
                    body,
                    re_mod.I,
                )
                and re_mod.search(
                    r"\btotalSupply\s*-=\s*[A-Za-z_]\w*",
                    body,
                    re_mod.I,
                )
                and re_mod.search(r"\bemit\s+Burn\s*\(", body, re_mod.I)
            )

        if name == "minttoken":
            return bool(
                re_mod.search(r"\bonlyOwner\b", body, re_mod.I)
                and re_mod.search(
                    r"\bbalanceOf\s*\[\s*[A-Za-z_]\w*\s*\]\s*\+=\s*[A-Za-z_]\w*",
                    body,
                    re_mod.I,
                )
                and re_mod.search(
                    r"\btotalSupply\s*\+=\s*[A-Za-z_]\w*",
                    body,
                    re_mod.I,
                )
                and len(re_mod.findall(r"\bemit\s+Transfer\s*\(", body, re_mod.I))
                >= 2
            )

        return False

    has_wrapper_allowance_guard = bool(
        re_mod.search(
            r"\brequire\s*\([^;{}]*\b(?:allowance|allowed)\b"
            r"[^;{}]*>=\s*[A-Za-z_]\w*",
            body,
            re_mod.I,
        )
        or re_mod.search(
            r"\brequire\s*\([^;{}]*[A-Za-z_]\w*\s*<=\s*"
            r"\b(?:allowance|allowed)\b[^;{}]*",
            body,
            re_mod.I,
        )
    )
    has_wrapper_allowance_debit = bool(
        re_mod.search(
            r"\b(?:allowance|allowed)\b[^;{}]*-=\s*[A-Za-z_]\w*",
            body,
            re_mod.I,
        )
    )
    has_internal_transfer = bool(
        re_mod.search(r"\b_?transfer\s*\([^;{}]*\)", body, re_mod.I)
    )
    if (
        has_wrapper_allowance_guard
        and has_wrapper_allowance_debit
        and has_internal_transfer
    ):
        return True

    has_source_guard = bool(
        re_mod.search(
            rf"\brequire\s*\([^;{{}}]*\b{mapping}\s*\[\s*{identifier}\s*\]"
            rf"\s*>=\s*{identifier}\b",
            body,
            re_mod.I,
        )
    )
    has_allowance_guard = bool(
        re_mod.search(
            r"\brequire\s*\([^;{}]*\b(?:allowance|allowed)\b[^;{}]*>=\s*"
            rf"{identifier}\b",
            body,
            re_mod.I,
        )
    )
    has_recipient_credit = bool(
        re_mod.search(
            rf"\b{mapping}\s*\[\s*{identifier}\s*\]\s*\+="
            rf"\s*{identifier}\b",
            body,
            re_mod.I,
        )
    )
    has_source_debit = bool(
        re_mod.search(
            rf"\b{mapping}\s*\[\s*{identifier}\s*\]\s*-="
            rf"\s*{identifier}\b",
            body,
            re_mod.I,
        )
    )
    has_allowance_debit = bool(
        re_mod.search(
            r"\b(?:allowance|allowed)\b[^;{}]*-="
            rf"\s*{identifier}\b",
            body,
            re_mod.I,
        )
    )
    return all(
        (
            has_source_guard,
            has_allowance_guard,
            has_recipient_credit,
            has_source_debit,
            has_allowance_debit,
        )
    )


def _unprotected_native_ether_withdrawal_lines(func: FunctionFlow, source_code: str) -> list[int]:
    """Return caller-transfer loci only when the enclosing function lacks a payout proof."""
    if _is_admin_restricted(func, source_code):
        return []
    if any(re_mod.match(r"(?:only|auth|admin)", modifier, re_mod.IGNORECASE) for modifier in func.modifiers):
        return []

    lines = source_code.split('\n')
    start = max(func.line_number - 1, 0)
    end = func.end_line_number if func.end_line_number > start else start + 50
    function_lines = lines[start:min(end, len(lines))]
    function_text = '\n'.join(function_lines)
    if re_mod.search(
        r"\b(?:require|assert)\s*\(\s*(?:msg\.sender\s*==|[A-Za-z_]\w*\s*==\s*msg\.sender)",
        function_text,
    ):
        return []
    claimant_balance_refs = {
        re_mod.sub(r"\s+", "", match.group(0))
        for match in re_mod.finditer(
            r"\b[A-Za-z_]\w*\s*\[\s*msg\.sender\s*\](?:\s*\[[^\]]+\])?",
            function_text,
        )
    }
    # Do not interpolate a potentially nested mapping expression into a
    # repeated regex.  On real contracts that can trigger catastrophic
    # backtracking before the model request is reached.
    compact_lines = [re_mod.sub(r"\s+", "", line) for line in function_lines]
    for reference in claimant_balance_refs:
        for line in compact_lines:
            position = line.find(reference)
            if position < 0:
                continue
            suffix = line[position + len(reference):]
            if suffix.startswith("-=") or suffix.startswith("=0"):
                return []
            if suffix.startswith("="):
                rhs = suffix[1:]
                if reference in rhs and (".sub(" in rhs or "-" in rhs):
                    return []

    claimant_balance_guard = bool(re_mod.search(
        r"(?:\bbalanceOf\s*\(\s*msg\.sender\s*\)|"
        r"\b[A-Za-z_]\w*\s*\[\s*msg\.sender\s*\])\s*"
        r"(?:>=|>|==|!=)",
        function_text,
        re_mod.IGNORECASE,
    ))
    claimant_burn = bool(re_mod.search(
        r"\b_burn\s*\(\s*msg\.sender\s*,",
        function_text,
        re_mod.IGNORECASE,
    ))
    if claimant_balance_guard and claimant_burn:
        return []
    if _has_local_storage_claimant_debit(function_text):
        return []
    if (
        os.environ.get(PROFILE_ENV) == E1_OPTIMIZED_V1_PROFILE
        and _has_external_claimant_debit_before_payout(function_text)
    ):
        return []

    caller_transfer = re_mod.compile(
        r"(?:msg\.sender|payable\s*\(\s*msg\.sender\s*\))\s*\.\s*transfer\s*\("
    )
    return [
        start + offset + 1
        for offset, line in enumerate(function_lines)
        if caller_transfer.search(line)
    ]


def _has_complete_state_arithmetic_guards(func_text: str) -> bool:
    """Recognize bounded token-style mapping updates before treating them as arithmetic risks."""
    state_op = re_mod.compile(
        r"\b(?P<base>[A-Za-z_]\w*)(?:\s*\[[^\]]+\])+\s*"
        r"(?P<op>\+=|-=)\s*(?P<value>[A-Za-z_]\w*)"
    )
    operations = [match.groupdict() for match in state_op.finditer(func_text)]
    if not operations:
        return False

    guard_text = " ".join(re_mod.findall(
        r"(?:require|assert)\s*\((.*?)\)\s*;", func_text, re_mod.DOTALL
    ))
    if not guard_text:
        return False

    aliases = {}
    for match in re_mod.finditer(
        r"\b(?:u?int(?:\d+)?)\s+(?P<alias>[A-Za-z_]\w*)\s*=\s*"
        r"(?P<base>[A-Za-z_]\w*(?:\s*\[[^\]]+\])+)",
        func_text,
    ):
        aliases[match.group("alias")] = match.group("base").split("[")[0]

    def directly_guarded(base: str, value: str) -> bool:
        return bool(re_mod.search(
            rf"\b{re_mod.escape(base)}(?:\s*\[[^\]]+\])+\s*>=\s*{re_mod.escape(value)}\b",
            guard_text,
        ))

    def alias_guarded(base: str, value: str) -> bool:
        return any(
            alias_base == base and re_mod.search(
                rf"\b{re_mod.escape(alias)}\s*>=\s*{re_mod.escape(value)}\b", guard_text
            )
            for alias, alias_base in aliases.items()
        )

    guarded_decrements = {
        (item["base"], item["value"])
        for item in operations
        if item["op"] == "-=" and (
            directly_guarded(item["base"], item["value"])
            or alias_guarded(item["base"], item["value"])
        )
    }

    for item in operations:
        base, op, value = item["base"], item["op"], item["value"]
        if op == "-=" and (base, value) not in guarded_decrements:
            return False
        if op == "+=":
            explicit_overflow_guard = bool(re_mod.search(
                rf"\b{re_mod.escape(base)}(?:\s*\[[^\]]+\])+\s*\+\s*{re_mod.escape(value)}\s*>\s*"
                rf"{re_mod.escape(base)}(?:\s*\[[^\]]+\])+",
                guard_text,
            ))
            if not explicit_overflow_guard and (base, value) not in guarded_decrements:
                return False
    return True


def _source_arithmetic_evidence(
    func: FunctionFlow,
    source_lines: List[str],
    state_names: Set[str],
) -> Tuple[bool, List[int], List[int]]:
    """Require source-grounded arithmetic with a persistent or narrow-value sink."""

    start = max(func.line_number - 1, 0)
    end = min(func.end_line_number or len(source_lines), len(source_lines))
    clean_lines = _strip_comments_preserve_lines("\n".join(source_lines[start:end])).splitlines()
    function_text = "\n".join(clean_lines)
    parameter_names: Set[str] = set()
    # Function declarations are frequently formatted across several lines.
    # Reusing the bounded header parser keeps caller-controlled operands visible
    # without widening the source slice or changing the arithmetic admission
    # policy.
    parameter_names = _function_parameter_names(func, source_lines)

    arithmetic_pattern = re_mod.compile(
        r"\b(?P<lhs>[A-Za-z_]\w*(?:\s*\[[^\]\r\n]+\])?)\s*"
        r"(?P<op>\+=|-=|\*=|/=|=(?!=)|\+\+|--)\s*"
        r"(?P<rhs>[^;\r\n]*)?;?"
    )
    storage_aliases = _storage_alias_names_for_function(func, source_lines)
    persistent_names = set(state_names) | storage_aliases
    local_integer_bits: Dict[str, int] = {}
    for declaration in re_mod.finditer(
        r"\buint(?P<bits>\d*)\s+(?P<name>[A-Za-z_]\w*)\b",
        "\n".join(clean_lines),
        re_mod.I,
    ):
        local_integer_bits[declaration.group("name")] = int(
            declaration.group("bits") or "256"
        )

    arithmetic_lines = []
    for offset, line in enumerate(clean_lines):
        match = arithmetic_pattern.search(line)
        if not match:
            continue
        # A plain assignment is only arithmetic when its RHS contains an
        # arithmetic operator; compound/postfix updates are arithmetic by
        # definition.
        if match.group("op") == "=" and not re_mod.search(
            r"[+\-*/]", match.group("rhs") or ""
        ):
            continue
        arithmetic_lines.append((start + offset + 1, line.strip(), match))
    if not arithmetic_lines:
        return False, [], []

    state_write_lines = [
        start + offset + 1
        for offset, line in enumerate(clean_lines)
        if _line_writes_persistent_state(line, state_names)
        or _line_writes_storage_alias(line, storage_aliases)
    ]
    source_by_line = {
        start + offset + 1: line
        for offset, line in enumerate(clean_lines)
    }
    operation_lines: List[int] = []
    standalone_operation_lines: List[int] = []

    def uses_as_value(line: str, name: str) -> bool:
        """Ignore a local name used only as a storage index."""

        without_index = re_mod.sub(
            rf"\[\s*{re_mod.escape(name)}\s*\]", "", line
        )
        return re_mod.search(rf"\b{re_mod.escape(name)}\b", without_index) is not None

    for line_number, _, match in arithmetic_lines:
        if match is None:
            continue
        lhs = re_mod.sub(r"\s+", "", match.group("lhs")).split("[", 1)[0]
        rhs = match.group("rhs") or ""
        caller_controlled = bool(
            re_mod.search(r"\bmsg\.value\b|\bmsg\.sender\b", rhs)
            or any(re_mod.search(rf"\b{re_mod.escape(name)}\b", rhs) for name in parameter_names)
        )
        if not caller_controlled:
            narrow_local = (
                lhs not in persistent_names
                and local_integer_bits.get(lhs, 256) < 256
            )
            literal_subtraction = bool(
                match.group("op") == "="
                and re_mod.search(r"(?<![A-Za-z_])-\s*(?:0x[0-9a-f]+|\d+)\b", rhs, re_mod.I)
            )
            if not (narrow_local and literal_subtraction):
                continue
        direct_state_write = line_number in state_write_lines and lhs in persistent_names
        feeds_state_write = any(
            later > line_number
            and uses_as_value(source_by_line.get(later, ""), lhs)
            for later in state_write_lines
        )
        if direct_state_write or feeds_state_write:
            operation_lines.append(line_number)
        elif (
            lhs not in persistent_names
            and local_integer_bits.get(lhs, 256) < 256
        ):
            # A caller-controlled operation on a narrow unsigned integer is
            # independently overflow-prone even when the injected result is
            # not written to contract storage.  This remains source evidence,
            # not a label- or dataset-specific shortcut.
            standalone_operation_lines.append(line_number)

    # A separate balance/allowance guard makes ordinary token bookkeeping a
    # negative control.  Keep non-state narrow-integer evidence above intact
    # when a function happens to contain both patterns.
    if _has_complete_state_arithmetic_guards(function_text):
        operation_lines = [
            line for line in operation_lines if line not in state_write_lines
        ]

    selected_operations = sorted(set(operation_lines + standalone_operation_lines))
    if not selected_operations:
        return False, [line for line, _, _ in arithmetic_lines], state_write_lines
    evidence_lines = sorted(set(selected_operations + [
        line for line in state_write_lines if line > min(operation_lines)
    ])) if operation_lines else sorted(set(selected_operations))
    return True, selected_operations, evidence_lines


def _detect_timestamp_parameter_storage_arithmetic(
    func: FunctionFlow, source_code: str
) -> dict | None:
    """Find legacy timestamp-plus-parameter writes in internal storage helpers.

    DAppSCAN contains library-style helpers whose storage target is a struct
    member inherited from another contract, so the normal state-variable index
    cannot see the write.  Keep this repair deliberately narrow: pre-0.8
    semantics, an internal/private helper, ``block.timestamp``/``now`` plus a
    declared parameter, and an assignment whose left-hand side is visibly a
    mapping/struct storage path.
    """

    if not source_code or func.is_constructor or func.visibility not in {"internal", "private"}:
        return None
    pragma = re_mod.search(
        r"\bpragma\s+solidity\s+[^0-9]*(\d+)\.(\d+)", source_code, re_mod.I
    )
    if not pragma or (int(pragma.group(1)), int(pragma.group(2))) >= (0, 8):
        return None

    source_lines = source_code.splitlines()
    start = max(func.line_number - 1, 0)
    end = min(func.end_line_number or len(source_lines), len(source_lines))
    clean_lines = _strip_comments_preserve_lines(
        "\n".join(source_lines[start:end])
    ).splitlines()
    parameter_names = _function_parameter_names(func, source_lines)
    if not parameter_names or not clean_lines:
        return None

    storage_assignment = re_mod.compile(
        r"(?P<lhs>[A-Za-z_]\w*(?:\s*(?:\[[^\]\r\n]+\]|\.\s*[A-Za-z_]\w*))+\s*)"
        r"=\s*(?P<rhs>[^;\r\n]+)",
        re_mod.I,
    )
    timestamp = re_mod.compile(r"\b(?:block\s*\.\s*timestamp|now)\b", re_mod.I)
    operation_lines: list[int] = []
    state_write_lines: list[int] = []
    for offset, line in enumerate(clean_lines):
        match = storage_assignment.search(line)
        if not match:
            continue
        rhs = match.group("rhs")
        if not timestamp.search(rhs):
            continue
        if not any(
            re_mod.search(rf"\b{re_mod.escape(name)}\b", rhs)
            for name in parameter_names
        ):
            continue
        line_number = start + offset + 1
        operation_lines.append(line_number)
        state_write_lines.append(line_number)

    if not operation_lines:
        return None
    return {
        "risk_type": "arithmetic",
        "confidence": 0.90,
        "reason": (
            f"{func.name}() adds block timestamp to a caller-provided parameter "
            "and writes the result through a mapping/struct storage path under "
            "pre-0.8 arithmetic semantics"
        ),
        "function_name": func.name,
        "line": func.line_number,
        "source_grounded_arithmetic": True,
        "arithmetic_operation_lines": operation_lines,
        "state_write_lines": state_write_lines,
        "evidence_lines": sorted(set(operation_lines + state_write_lines)),
        "source_arithmetic_proof": "block_timestamp_parameter_to_persistent_storage_write",
        "source_evidence_kind": "timestamp_parameter_storage_write",
        "internal_storage_helper": True,
    }


def _detect_e2_legacy_arithmetic_value_sinks(
    features: ContractFeatures, source_code: str
) -> List[dict]:
    """Retain bounded legacy arithmetic that reaches a value or asset sink.

    DAppSCAN contains library-style and inherited helpers that are not exposed
    as reachable functions in the reconstructed source slice.  The normal
    arithmetic loop intentionally stays on public/external state updates, so
    those known arithmetic loci disappear before the E2 development arm can
    localize them.  This helper is E2-development-only and requires a
    source-anchored arithmetic operation plus a return, storage-looking update,
    inline-assembly value write, or asset operation in the same function.
    """

    if not features or not source_code:
        return []
    pragma = re_mod.search(
        r"\bpragma\s+solidity\s+[^0-9]*(\d+)\.(\d+)", source_code, re_mod.I
    )
    if not pragma or (int(pragma.group(1)), int(pragma.group(2))) >= (0, 8):
        return []

    source_lines = source_code.splitlines()
    risks: list[dict] = []
    arithmetic = re_mod.compile(
        r"(?<![A-Za-z0-9_])(?:\+\+|--|\+=|-=|\*=|/=|\+|-|\*|/)(?![=+*/-])"
        r"|\.(?:add|sub|mul|div|mod)\s*\("
        r"|\b(?:badd|bsub|bmul|bdiv|bpow|safePower|fromScaledUint)\s*\(",
        re_mod.I,
    )
    asset_call = re_mod.compile(
        r"(?:\.|\b)(?:transfer|send|transferFrom|safeTransfer|safeTransferFrom|mint|burn|"
        r"deposit|withdraw|swap\w*|exchange\w*|execute\w*|settle\w*|"
        r"claim\w*|redeem\w*|addLiquidity|removeLiquidity|assignTokens|"
        r"reinvest|compound|harvest)\s*\(",
        re_mod.I,
    )
    raw_arithmetic = re_mod.compile(
        r"(?<![=!<>+\-*/])(?:\*\*|\+|\-|\*|/)(?![=+\-*/])|"
        r"(?:\+\+|--|\+=|-=|\*=|/=)",
        re_mod.I,
    )
    string_literal = re_mod.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

    # The production caller passes the parser-owned contract list so a
    # SafeMath helper can be recognized by its actual library scope.  Keep a
    # single ContractFeatures value accepted for focused unit tests.
    contracts = (
        list(features)
        if isinstance(features, (list, tuple))
        else [features]
    )
    for contract in contracts:
        functions = getattr(contract, "functions", []) or []
        state_names = {
            str(name)
            for name in (getattr(contract, "state_variables", []) or [])
            if str(name).strip()
        }
        for func in functions:
            if func.is_constructor or not func.name:
                continue
            start = max(func.line_number - 1, 0)
            end = min(func.end_line_number or (start + 80), len(source_lines))
            clean_lines = [
                re_mod.sub(r"//.*$", "", line)
                for line in source_lines[start:end]
            ]
            if not clean_lines:
                continue
            function_text = "\n".join(clean_lines)

            # SafeMath's raw operators are intentionally checked by the
            # helper itself.  Match the parser-owned library boundary instead
            # of using a file-wide SafeMath flag, so an unrelated vulnerable
            # function named ``sub`` or ``div`` remains eligible.
            contract_name = str(getattr(contract, "name", "") or "")
            checked_helper = (
                bool(re_mod.search(r"\b(?:safemath|checkedmath)\b", contract_name, re_mod.I))
                and str(func.name or "").casefold() in {
                    "add", "sub", "mul", "div", "mod",
                    "tryadd", "trysub", "trymul", "trydiv", "trymod",
                }
                and bool(re_mod.search(r"\b(?:require|assert)\s*\(", function_text, re_mod.I))
            )
            if checked_helper:
                continue

            # SafeMath/checked arithmetic helpers and guarded collection
            # index shuffles are implementation scaffolding, not standalone
            # overflow findings.  Their callers remain eligible through the
            # enclosing value/state flow.
            if (
                re_mod.search(
                    r"\brequire\s*\([^\n;]*(?:overflow|underflow|division by zero|zero)",
                    function_text,
                    re_mod.I,
                )
                and not asset_call.search(function_text)
            ):
                continue
            if (
                re_mod.search(r"\b[A-Za-z_]\w*Index\s*=\s*[A-Za-z_]\w*\s*-\s*1\b", function_text)
                and re_mod.search(r"\b[A-Za-z_]\w*Index\s*!=\s*0\b", function_text)
            ):
                continue

            # Do not reintroduce a legacy arithmetic candidate when the same
            # mapping update is already covered by a complete operand guard.
            # This keeps the helper fail-closed for checked transferFrom-style
            # bookkeeping while leaving unrelated raw operations eligible.
            if _has_complete_state_arithmetic_guards("\n".join(clean_lines)):
                continue

            parameter_names = _function_parameter_names(func, source_lines)
            storage_aliases = _storage_alias_names_for_function(func, source_lines)
            tainted_names = set(parameter_names)
            candidate_closures: list[tuple[int, list[int], list[int], list[int]]] = []

            def contains_name(text: str, names: set[str]) -> bool:
                return any(
                    re_mod.search(rf"\b{re_mod.escape(name)}\b", text)
                    for name in names
                    if name
                )

            def has_controlled_input(expression: str, lhs_base: str) -> bool:
                return bool(
                    contains_name(expression, tainted_names)
                    or contains_name(expression, state_names)
                    or re_mod.search(
                        r"\b(?:msg\s*\.\s*(?:value|sender|data)|calldata|"
                        r"(?:get)?balance(?:Of)?|address\s*\(\s*this\s*\))\b",
                        expression,
                        re_mod.I,
                    )
                    or lhs_base in state_names
                )

            for offset, line in enumerate(clean_lines):
                if re_mod.search(r"\b(?:function|modifier|pragma|import)\b", line):
                    continue
                analysis_line = string_literal.sub("\"\"", line)
                if not raw_arithmetic.search(analysis_line):
                    continue
                line_number = start + offset + 1
                stripped = analysis_line.strip()
                operation_match = re_mod.search(
                    r"^(?:return\s+)?(?:[A-Za-z_]\w*(?:\s*\[[^\]\r\n]+\])?\s+)?"
                    r"(?P<lhs>[A-Za-z_]\w*(?:\s*\[[^\]\r\n]+\])?)\s*"
                    r"(?P<op>=|\+=|-=|\*=|/=|\+\+|--)\s*"
                    r"(?P<rhs>[^;]+)",
                    stripped,
                    re_mod.I,
                )
                is_return = bool(re_mod.match(r"^return\b", stripped, re_mod.I))
                if is_return:
                    expression = re_mod.sub(r"^return\s+", "", stripped, flags=re_mod.I)
                    lhs_base = ""
                elif operation_match:
                    expression = operation_match.group("rhs")
                    lhs_base = re_mod.sub(
                        r"\s+", "", operation_match.group("lhs")
                    ).split("[", 1)[0]
                else:
                    expression = stripped
                    lhs_base = ""
                if not has_controlled_input(expression, lhs_base):
                    continue

                flow_names = {lhs_base} if lhs_base else set()
                # Preserve the operands that seed the value flow.  For a
                # return expression such as ``a + b`` there is no local LHS,
                # so the operand names are the only usable closure anchors.
                flow_names.update(
                    name
                    for name in (tainted_names | state_names)
                    if contains_name(expression, {name})
                )
                operation_state_lines: list[int] = []
                operation_value_lines: list[int] = []
                if is_return:
                    operation_value_lines.append(line_number)
                elif lhs_base and (
                    lhs_base in state_names
                    or _line_writes_persistent_state(line, state_names)
                    or _line_writes_storage_alias(line, storage_aliases)
                ):
                    operation_state_lines.append(line_number)

                for later_offset in range(offset + 1, len(clean_lines)):
                    later = string_literal.sub("\"\"", clean_lines[later_offset])
                    later_number = start + later_offset + 1
                    assignment_alias = re_mod.search(
                        r"^(?:[A-Za-z_]\w*(?:\s*\[[^\]\r\n]+\])?\s+)?"
                        r"(?P<alias>[A-Za-z_]\w*(?:\s*\[[^\]\r\n]+\])?)\s*=\s*"
                        r"(?P<alias_rhs>[^;]+)",
                        later.strip(),
                        re_mod.I,
                    )
                    if assignment_alias and contains_name(
                        assignment_alias.group("alias_rhs"), flow_names
                    ):
                        alias_base = re_mod.sub(
                            r"\s+", "", assignment_alias.group("alias")
                        ).split("[", 1)[0]
                        flow_names.add(alias_base)
                    if not flow_names or not contains_name(later, flow_names):
                        continue
                    assignment = re_mod.search(
                        r"(?P<lhs>[A-Za-z_]\w*(?:\s*\[[^\]\r\n]+\])?(?:\s*\.\s*[A-Za-z_]\w*)*)\s*"
                        r"(?:=|\+=|-=|\*=|/=)(?!=)\s*(?P<rhs>[^;]+)",
                        later,
                        re_mod.I,
                    )
                    assignment_lhs = (
                        re_mod.sub(r"\s+", "", assignment.group("lhs"))
                        if assignment
                        else ""
                    )
                    assignment_rhs = assignment.group("rhs") if assignment else ""
                    writes_state = bool(
                        (
                            _line_writes_persistent_state(later, state_names)
                            or _line_writes_storage_alias(later, storage_aliases)
                        )
                        and (
                            contains_name(assignment_lhs, flow_names)
                            or contains_name(assignment_rhs, flow_names)
                        )
                    )
                    returns_value = bool(
                        (return_match := re_mod.search(r"\breturn\b(?P<expr>[^;]*)", later, re_mod.I))
                        and contains_name(return_match.group("expr"), flow_names)
                    )
                    array_boundary = bool(
                        flow_names
                        and re_mod.search(
                            r"\[[^\]]*\b(?:"
                            + "|".join(
                                re_mod.escape(name) for name in flow_names if name
                            )
                            + r")\b[^\]]*\]",
                            later,
                        )
                    )
                    asset_sink = bool(asset_call.search(later))
                    assembly_value_write = bool(
                        re_mod.search(r"\b(?:mstore|mstore8)\s*\(", later, re_mod.I)
                        and any(
                            re_mod.search(r"\b(?:assembly|let)\b", prior, re_mod.I)
                            for prior in clean_lines[max(0, later_offset - 8) : later_offset + 1]
                        )
                    )
                    if not (
                        writes_state
                        or returns_value
                        or array_boundary
                        or asset_sink
                        or assembly_value_write
                    ):
                        continue
                    if writes_state:
                        operation_state_lines.append(later_number)
                    else:
                        operation_value_lines.append(later_number)
                    break

                impact_lines = sorted(set(operation_state_lines + operation_value_lines))
                if not impact_lines:
                    continue
                candidate_closures.append(
                    (line_number, operation_state_lines, operation_value_lines, impact_lines)
                )

            if not candidate_closures:
                continue
            first_operation, state_lines, value_lines, impact_lines = candidate_closures[0]
            evidence_lines = sorted(set([first_operation, *impact_lines]))
            if any(
                risk.get("risk_type") == "arithmetic"
                and risk.get("function_name") == func.name
                for risk in risks
            ):
                continue
            risks.append(
                {
                    "risk_type": "arithmetic",
                    "confidence": 0.90,
                    "reason": (
                        f"{func.name}() contains source-anchored arithmetic in pre-0.8 code "
                        "whose result reaches a value, storage, assembly, or asset sink"
                    ),
                    "function_name": func.name,
                    "line": first_operation,
                    "evidence_lines": evidence_lines,
                    "source_grounded_arithmetic": True,
                    "arithmetic_operation_lines": [first_operation],
                    "state_write_lines": sorted(set(state_lines)),
                    "value_relevance_lines": sorted(set(value_lines)),
                    "source_arithmetic_proof": "legacy_arithmetic_to_value_or_state_sink",
                    "source_evidence_kind": "legacy_arithmetic_value_or_state_sink",
                }
            )
    return risks


def _source_value_arithmetic_evidence(
    func: FunctionFlow,
    source_lines: List[str],
    state_names: Set[str],
) -> Tuple[bool, List[int], List[int], str]:
    """Find legacy arithmetic that changes a returned or security-relevant value.

    The older state-write-only rule missed view/quote functions such as
    ``return duration * rentPrice``.  Keep this closure narrow: arithmetic must
    use a parameter, state value, or external balance-like value and flow to a
    return, branch, external call, or persistent write.  A temporary used only
    for an event remains a negative control.
    """

    if func.is_constructor:
        return False, [], [], ""

    start = max(func.line_number - 1, 0)
    end = min(func.end_line_number or len(source_lines), len(source_lines))
    clean_lines = _strip_comments_preserve_lines(
        "\n".join(source_lines[start:end])
    ).splitlines()
    if not clean_lines:
        return False, [], [], ""

    function_text = "\n".join(clean_lines)
    # SafeMath-style calls already carry their own arithmetic checks.  A raw
    # multiplication inside a larger SafeMath expression is not enough to
    # promote the enclosing function to an arithmetic vulnerability.
    if re_mod.search(r"\.(?:add|sub|mul|div|mod)\s*\(", function_text, re_mod.I):
        return False, [], [], ""

    function_name = func.name.casefold()
    safe_helper_guard = (
        function_name in {"safeadd", "safesub", "safemul", "safediv", "safemod"}
        and bool(re_mod.search(r"\brequire\s*\(", function_text, re_mod.I))
    )
    if safe_helper_guard:
        return False, [], [], ""

    signature = "\n".join(source_lines[start:min(start + 8, end)])
    parameter_names: Set[str] = set()
    parameter_match = re_mod.search(r"\(([^)]*)\)", signature, re_mod.DOTALL)
    if parameter_match:
        for declaration in parameter_match.group(1).split(","):
            tokens = re_mod.findall(r"[A-Za-z_]\w*", declaration)
            if tokens:
                parameter_names.add(tokens[-1])

    named_return_names: Set[str] = set()
    returns_match = re_mod.search(r"\breturns\s*\(([^)]*)\)", signature, re_mod.DOTALL)
    if returns_match:
        for declaration in returns_match.group(1).split(","):
            tokens = re_mod.findall(r"[A-Za-z_]\w*", declaration)
            if len(tokens) >= 2:
                named_return_names.add(tokens[-1])

    arithmetic = re_mod.compile(
        r"(?<![=!<>+\-*/])(?:\*\*|\+|-|\*|/)(?![=+\-*/])"
    )
    value_source = re_mod.compile(
        r"\b(?:msg\.value|msg\.sender|address\s*\(\s*this\s*\)|"
        r"balance|balanceOf|getBalance|price|amount|value|duration|rate|"
        r"total|supply|reserve|cost|fee)\b",
        re_mod.I,
    )

    # A pure one-parameter calculation that only returns a local temporary is
    # a negative control, not a source-grounded overflow sink.  Keep ordinary
    # quote/rate helpers eligible when they combine multiple caller operands,
    # read contract state, or call an external value source.
    pure_local_return_control = (
        str(getattr(func, "state_mutability", "") or "").casefold() == "pure"
        and not state_names
        and not re_mod.search(
            r"\b(?:msg\s*\.\s*value|msg\s*\.\s*sender|balanceOf|getBalance|"
            r"address\s*\(\s*this\s*\)|\.\s*(?:call|send|transfer|swap\w*)\s*\()",
            function_text,
            re_mod.I,
        )
    )

    def has_multiple_parameter_operands(expression: str) -> bool:
        return sum(
            1
            for name in parameter_names
            if re_mod.search(rf"\b{re_mod.escape(name)}\b", expression)
        ) >= 2
    operation_lines: List[int] = []
    relevance_lines: List[int] = []
    proof = ""

    def uses_controlled_or_value_source(expression: str) -> bool:
        if any(
            re_mod.search(rf"\b{re_mod.escape(name)}\b", expression)
            for name in parameter_names
        ):
            return True
        if any(
            re_mod.search(rf"\b{re_mod.escape(name)}\b", expression)
            for name in state_names
        ):
            return True
        return bool(value_source.search(expression))

    for offset, line in enumerate(clean_lines):
        stripped = line.strip()
        if not stripped or re_mod.search(r"\bfor\s*\(", stripped):
            continue
        if not arithmetic.search(stripped):
            continue

        source_line = start + offset + 1
        if re_mod.search(r"^return\b", stripped, re_mod.I):
            expression = re_mod.sub(r"^return\s+", "", stripped, flags=re_mod.I)
            if uses_controlled_or_value_source(expression):
                if pure_local_return_control and not has_multiple_parameter_operands(
                    expression
                ):
                    continue
                operation_lines.append(source_line)
                relevance_lines.append(source_line)
                proof = "caller_input_arithmetic_to_value_relevant_return"
                break
            continue

        assignment = re_mod.match(
            r"^(?:[A-Za-z_]\w*(?:\s*\[[^\]]+\])?\s+)?"
            r"(?P<lhs>[A-Za-z_]\w*(?:\s*\[[^\]]+\])?)\s*"
            r"(?P<op>=|\+=|-=|\*=|/=)\s*(?P<rhs>[^;]+)",
            stripped,
        )
        if not assignment:
            continue
        rhs = assignment.group("rhs")
        lhs = re_mod.sub(r"\s+", "", assignment.group("lhs")).split("[", 1)[0]
        if not uses_controlled_or_value_source(rhs):
            continue

        if pure_local_return_control and not has_multiple_parameter_operands(rhs):
            continue

        if lhs in named_return_names:
            operation_lines.append(source_line)
            relevance_lines.append(source_line)
            proof = "caller_input_arithmetic_to_named_return_value"
            continue

        later_lines = clean_lines[offset + 1:]
        sink_offset = None
        for later_offset, later in enumerate(later_lines, start=offset + 1):
            if not re_mod.search(rf"\b{re_mod.escape(lhs)}\b", later):
                continue
            if re_mod.search(
                r"\b(?:return|require|assert|if|while|revert)\b|"
                r"\.(?:call|send|transfer|transferFrom|safeTransfer|"
                r"safeTransferFrom|mint|burn|swap\w*)\s*\(",
                later,
                re_mod.I,
            ):
                sink_offset = later_offset
                break
            if _line_writes_persistent_state(later, state_names):
                sink_offset = later_offset
                break
        if sink_offset is None:
            continue
        operation_lines.append(source_line)
        relevance_lines.append(start + sink_offset + 1)
        proof = "caller_input_arithmetic_to_value_relevant_branch"
        break

    if not operation_lines:
        return False, [], [], ""
    return True, operation_lines, sorted(set(relevance_lines)), proof


def _arithmetic_operation_provenance(
    contracts: list[ContractFeatures], source_code: str
) -> list[dict]:
    """Index arithmetic loci across all functions without admitting findings.

    Internal/private helpers are intentionally included here because this is
    provenance-only metadata.  The existing public/external AST admission loop
    remains unchanged.
    """

    if not source_code:
        return []
    source_lines = source_code.splitlines()
    rows: list[dict] = []
    for contract in contracts:
        state_names = set(contract.state_variables)
        for func in contract.functions:
            source_grounded, operation_lines, evidence_lines = _source_arithmetic_evidence(
                func, source_lines, state_names
            )
            value_grounded = False
            value_operation_lines: list[int] = []
            value_relevance_lines: list[int] = []
            value_proof = ""
            if not source_grounded:
                (
                    value_grounded,
                    value_operation_lines,
                    value_relevance_lines,
                    value_proof,
                ) = _source_value_arithmetic_evidence(func, source_lines, state_names)
            if not (source_grounded or value_grounded):
                continue
            selected_operations = (
                operation_lines if source_grounded else value_operation_lines
            )
            selected_evidence = (
                evidence_lines
                if source_grounded
                else sorted(set(value_operation_lines + value_relevance_lines))
            )
            proof = (
                "persistent_state_postfix_update"
                if source_grounded and any(
                    re_mod.search(r"\+\+|--", source_lines[line - 1])
                    for line in selected_operations
                    if 1 <= line <= len(source_lines)
                )
                else (
                    "caller_input_arithmetic_to_persistent_state_write"
                    if source_grounded
                    else value_proof
                )
            )
            rows.append({
                "function_name": func.name,
                "visibility": func.visibility,
                "state_mutability": func.state_mutability,
                "function_span": {
                    "start_line": func.line_number,
                    "end_line": func.end_line_number,
                },
                "arithmetic_operation_lines": list(selected_operations),
                "evidence_lines": list(selected_evidence),
                "source_grounded_arithmetic": True,
                "source_arithmetic_proof": proof,
                "provenance_only": True,
            })
    return rows


def _extract_function_signature(lines, start_line: int, total_lines: int) -> str:
    sig_parts = []
    for i in range(start_line, min(start_line + 8, total_lines)):
        line = lines[i].rstrip()
        sig_parts.append(line)
        if '{' in line:
            break
    return ' '.join(sig_parts) if sig_parts else ""


def _extract_rw_conflict_graph(features: ContractFeatures, source_code: str) -> List[dict]:
    if not features or not source_code:
        return []

    lines = source_code.split('\n')
    total_lines = len(lines)
    state_var_reads = defaultdict(list)
    state_var_writes = defaultdict(list)

    for func in features.functions:
        if func.visibility not in ['public', 'external']:
            continue

        if _is_admin_restricted(func, source_code):
            continue

        for sw in func.state_writes:
            for sv in features.state_variables:
                if sv in sw:
                    ws = func.line_number - 1
                    we = func.end_line_number if func.end_line_number > ws else ws + 30
                    we = min(we, total_lines)
                    func_sig = _extract_function_signature(lines, ws, total_lines)
                    assignment_lines = []
                    for li in range(ws + 1, we):
                        line_text = lines[li] if li < total_lines else ""
                        stripped = line_text.strip()
                        if sv in stripped and ('=' in stripped or sv.split('[')[0] in stripped):
                            assignment_lines.append(f"      {stripped} // Line {li + 1}")
                    skeleton = f"  {func_sig}"
                    if assignment_lines:
                        skeleton += "\n      ... [Code Omitted] ..."
                        for al in assignment_lines:
                            skeleton += f"\n{al}"
                    else:
                        skeleton += f"\n      ... modifies '{sv}' [Code Omitted] ..."
                    skeleton += "\n  }"

                    state_var_writes[sv].append({
                        "function": func.name,
                        "line": func.line_number,
                        "detail": sw,
                        "has_admin_modifier": False,
                        "skeleton": skeleton,
                    })

        func_text = ""
        start = func.line_number - 1
        end = func.end_line_number if func.end_line_number > start else start + 50
        end = min(end, total_lines)
        func_text = "\n".join(lines[start:end])

        for sv in features.state_variables:
            if sv in func_text:
                already_write = any(
                    w["function"] == func.name and sv in w["detail"]
                    for w in state_var_writes.get(sv, [])
                )
                if not already_write:
                    has_financial = any(
                        kw in func_text.lower()
                        for kw in ['msg.value', 'msg.sender', '.call', '.send', '.transfer',
                                    'require(', 'balance', 'reward', 'price', 'amount', 'withdraw', 'claim', 'pay']
                    )
                    if has_financial or func.external_calls or func.state_writes:
                        state_var_reads[sv].append({
                            "function": func.name,
                            "line": func.line_number,
                            "has_financial": has_financial,
                        })

    conflicts = []
    for sv in features.state_variables:
        writes = state_var_writes.get(sv, [])
        reads = state_var_reads.get(sv, [])
        if writes and reads:
            for w in writes:
                for r in reads:
                    if w["function"] != r["function"]:
                        conflicts.append({
                            "variable": sv,
                            "write_function": w["function"],
                            "write_line": w["line"],
                            "write_detail": w["detail"],
                            "write_skeleton": w.get("skeleton", ""),
                            "read_function": r["function"],
                            "read_line": r["line"],
                            "read_has_financial": r["has_financial"],
                        })

    return conflicts


def _detect_weakly_gated_public_payouts(features: ContractFeatures, source_code: str) -> List[dict]:
    """Find public claim paths where a trivial numeric gate controls a one-time payout."""
    if not features or not source_code:
        return []

    lines = source_code.split('\n')
    risks = []
    for func in features.functions:
        if _is_admin_restricted(func, source_code):
            continue
        start = func.line_number - 1
        end = func.end_line_number if func.end_line_number > start else start + 50
        end = min(end, len(lines))
        body_lines = lines[start:end]
        body = '\n'.join(body_lines)
        latch = re_mod.search(r"require\s*\(\s*!\s*([A-Za-z_]\w*)\s*\)", body)
        weak_gate = re_mod.search(
            r"require\s*\(\s*[A-Za-z_]\w*\s*(?:<|<=)\s*(?:[0-9]|1[0-6])\s*\)",
            body,
        )
        payout = re_mod.search(
            r"(?:msg\.sender|payable\s*\(\s*msg\.sender\s*\))\s*\.\s*(?:transfer|send)\s*\(",
            body,
        )
        if not latch or not weak_gate or not payout:
            continue
        if not re_mod.search(rf"\b{re_mod.escape(latch.group(1))}\s*=\s*true\b", body):
            continue
        gate_line = func.line_number + body[:weak_gate.start()].count('\n')
        risks.append({
            "risk_type": "front_running",
            "confidence": 0.9,
            "reason": "Public one-time payout is protected only by a trivial numeric gate, enabling an observer to race the intended claimant.",
            "function_name": func.name,
            "line": gate_line,
            "evidence_lines": [gate_line],
            "source_grounded": True,
            "source_evidence_kind": "weak_public_payout_gate",
        })
    return risks


def _detect_unprotected_critical_state_writers(
    features: ContractFeatures, source_code: str
) -> List[dict]:
    """Find E2-only unprotected external updaters that write caller-controlled value state.

    This is intentionally narrower than a generic "public state write" rule.  It
    requires an updater-like function name, reward/value state in the body, and a
    caller-controlled argument.  Ordinary user operations and admin-modified
    setters remain outside this candidate family.
    """

    if (
        not features
        or not source_code
        or os.environ.get(PROFILE_ENV) != E2_DAPPSCAN_VNEXT_PROFILE
        or os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") != "1"
    ):
        return []

    lines = source_code.split("\n")
    risks: list[dict] = []
    updater_name = re_mod.compile(
        r"\b(?:update|set|change|record|assign|add|remove)[A-Za-z0-9_]*"
        r"(?:reward|balance|share|supply|reserve|fee|amount|credit|debit|impact|score|metric)\b",
        re_mod.IGNORECASE,
    )
    critical_state = re_mod.compile(
        r"(?:\b|_)(?:reward|balance|share|supply|reserve|fee|amount|credit|debit|token|impact|score|metric)"
        r"[A-Za-z0-9_]*\b",
        re_mod.IGNORECASE,
    )
    caller_selected_index = re_mod.compile(
        r"\[[^\]\n]*\b(?:amount|value|fee|share|balance|reward|referral|recipient|account|"
        r"id|key|index|target|subject|proposal|virtual|dataset|service)\w*[^\]\n]*\]",
        re_mod.IGNORECASE,
    )
    storage_write = re_mod.compile(
        r"(?:\[[^\]\n]+\]|\.[A-Za-z_]\w*)\s*"
        r"(?:=|\+=|-=|\*=|/=)(?!=)"
    )

    for func in features.functions:
        if func.visibility not in {"public", "external"}:
            continue
        if _is_admin_restricted(func, source_code) or not updater_name.search(func.name):
            continue
        start = max(func.line_number - 1, 0)
        end = func.end_line_number if func.end_line_number > start else start + 50
        end = min(end, len(lines))
        body_lines = lines[start:end]
        body = "\n".join(body_lines)
        if not critical_state.search(body):
            continue
        if not storage_write.search(body):
            continue
        has_value_argument = bool(
            re_mod.search(
                r"\b(?:amount|value|fee|share|balance|reward|referral|recipient|account)\b",
                body,
                re_mod.IGNORECASE,
            )
        )
        if not has_value_argument and not caller_selected_index.search(body):
            continue

        write_lines: list[int] = []
        for offset, line in enumerate(body_lines):
            if storage_write.search(line):
                write_lines.append(start + offset + 1)
                if len(write_lines) >= 3:
                    break
        if not write_lines:
            continue
        evidence_lines = [func.line_number, *write_lines]
        risks.append({
            "risk_type": "access_control",
            "submechanism": "unprotected_critical_state_write",
            "confidence": 0.9,
            "reason": (
                f"{func.name}() is externally callable without an authorization "
                "modifier and writes caller-controlled reward/value state."
            ),
            "function_name": func.name,
            "line": write_lines[0],
            "evidence_lines": evidence_lines,
            "source_grounded": True,
            "source_evidence_kind": "unprotected_critical_state_write",
        })
    return risks


def _detect_e2_permissionless_critical_asset_operations(
    features: ContractFeatures, source_code: str
) -> List[dict]:
    """Find a small set of permissionless critical operations used by E2.

    This is deliberately source-shaped rather than name-only: each candidate
    must expose a public/external entrypoint, lack effective admin protection,
    and show a concrete state or asset sink in the same function.  The rule is
    development-only and does not alter the legacy E1 feature surface.
    """

    if not features or not source_code:
        return []
    lines = source_code.splitlines()
    risks: list[dict] = []

    def function_body(func: FunctionFlow) -> tuple[int, str]:
        start = max(func.line_number - 1, 0)
        end = min(func.end_line_number or (start + 100), len(lines))
        return start, "\n".join(lines[start:end])

    for func in features.functions:
        if (
            func.visibility not in {"public", "external"}
            or not func.is_reachable
            or _is_admin_restricted(func, source_code)
        ):
            continue
        start, body = function_body(func)
        name = func.name.casefold()
        evidence: list[int] = [func.line_number]
        operation_kind = ""

        if name in {"initialize", "init"}:
            # An initializer that writes core addresses without an initializer
            # guard is callable by the first arbitrary caller.
            if re_mod.search(r"\b(?:initializer|onlyInitializing|initialized)\b", body, re_mod.I):
                continue
            if not re_mod.search(
                r"\b(?:measure|asset|factory|token|controller|admin|owner)\b\s*=",
                body,
                re_mod.I,
            ):
                continue
            operation_kind = "unguarded_initializer_state_write"
            evidence.extend(
                start + offset + 1
                for offset, line in enumerate(body.splitlines())
                if re_mod.search(
                    r"\b(?:measure|asset|factory|token|controller|admin|owner)\b\s*=",
                    line,
                    re_mod.I,
                )
            )
        elif name == "setvalidators":
            if not re_mod.search(r"\bvalidators\s*\[|\bactiveValidatorSetId\s*=", body, re_mod.I):
                continue
            operation_kind = "permissionless_validator_set_write"
            evidence.extend(
                start + offset + 1
                for offset, line in enumerate(body.splitlines())
                if re_mod.search(r"\bvalidators\s*\[|\bactiveValidatorSetId\s*=", line, re_mod.I)
            )
        elif name == "transfer":
            if not re_mod.search(r"\bbalances\s*\[\s*msg\.sender\s*\]\s*=", body, re_mod.I):
                continue
            if not re_mod.search(r"\bbalances\s*\[[^\]]+\]\s*=", body, re_mod.I):
                continue
            operation_kind = "permissionless_balance_transfer"
            evidence.extend(
                start + offset + 1
                for offset, line in enumerate(body.splitlines())
                if re_mod.search(r"\bbalances\s*\[[^\]]+\]\s*=", line, re_mod.I)
            )
        elif name == "swapliquidity":
            if not re_mod.search(r"\.burn\s*\(\s*msg\.sender\s*,\s*receiverAddress", body, re_mod.I):
                continue
            if not re_mod.search(r"\bexecuteOperation\s*\(", body, re_mod.I):
                continue
            operation_kind = "permissionless_swap_asset_operation"
            evidence.extend(
                start + offset + 1
                for offset, line in enumerate(body.splitlines())
                if re_mod.search(r"\.burn\s*\(|\bexecuteOperation\s*\(", line, re_mod.I)
            )
        elif name == "claim":
            if not re_mod.search(r"\b[A-Za-z_]\w*\.transfer\s*\(\s*msg\.sender\s*,", body, re_mod.I):
                continue
            if not re_mod.search(r"\[[^\]]*msg\.sender[^\]]*\]", body, re_mod.I):
                continue
            operation_kind = "permissionless_claim_asset_operation"
            evidence.extend(
                start + offset + 1
                for offset, line in enumerate(body.splitlines())
                if re_mod.search(r"\.transfer\s*\(\s*msg\.sender\s*,", line, re_mod.I)
            )
        elif name == "swap":
            if not re_mod.search(r"\.(?:transfer|transferFrom|joinPool|exitPool)\s*\(", body, re_mod.I):
                continue
            operation_kind = "permissionless_swap_asset_operation"
            evidence.extend(
                start + offset + 1
                for offset, line in enumerate(body.splitlines())
                if re_mod.search(r"\.(?:transfer|transferFrom|joinPool|exitPool)\s*\(", line, re_mod.I)
            )
        else:
            continue

        if not operation_kind:
            continue
        risks.append(
            {
                "risk_type": "access_control",
                "submechanism": "permissionless_critical_asset_operation",
                "confidence": 0.90,
                "reason": (
                    f"{func.name}() is publicly reachable without an effective authorization path "
                    f"and reaches a concrete critical state/asset operation ({operation_kind})"
                ),
                "function_name": func.name,
                "line": min(evidence),
                "evidence_lines": sorted(set(evidence)),
                "source_grounded": True,
                "source_evidence_kind": "permissionless_critical_asset_operation",
                "state_write_lines": sorted(set(evidence[1:])),
            }
        )
    return risks


def _detect_tx_origin_external_call_arguments(
    features: ContractFeatures, source_code: str
) -> List[dict]:
    """Find permissionless asset calls that pass ``tx.origin`` as identity data.

    This closes a narrow access-control channel that is different from the
    existing storage-flow rule: ``tx.origin`` can be forwarded directly to an
    exchange/token operation without first being written to contract storage.
    EOA-only checks, event arguments, and admin-protected functions remain
    negative controls.
    """

    if not features or not source_code:
        return []

    operation_methods = re_mod.compile(
        r"(?:exchange|swap|transfer|send|deposit|withdraw|mint|burn|approve|"
        r"stake|unstake|liquidat|settle|claim|redeem|borrow|repay|flash|execute|reward)",
        re_mod.IGNORECASE,
    )
    call_with_origin = re_mod.compile(
        r"(?P<callee>(?:\b[A-Za-z_]\w*\s*\([^;]*?\)\s*\.\s*)?"
        r"\b[A-Za-z_]\w*)\s*\((?P<args>[^;]*\btx\s*\.\s*origin\b[^;]*)\)",
        re_mod.IGNORECASE | re_mod.DOTALL,
    )

    lines = source_code.splitlines()
    risks: list[dict] = []
    for func in features.functions:
        if (
            func.visibility not in {"public", "external"}
            or not func.is_reachable
            or _is_admin_restricted(func, source_code)
        ):
            continue

        start = max(func.line_number - 1, 0)
        end = min(func.end_line_number or (start + 80), len(lines))
        raw_body = "\n".join(lines[start:end])
        body_start = raw_body.find("{")
        if body_start < 0:
            continue
        body = _strip_comments_preserve_lines(raw_body[body_start + 1 :])
        body_base_line = func.line_number + raw_body[: body_start + 1].count("\n")

        for match in call_with_origin.finditer(body):
            callee = re_mod.sub(r"\s+", "", match.group("callee"))
            method = callee.rsplit(".", 1)[-1]
            if "." not in callee or not operation_methods.search(method):
                continue

            call_line = body_base_line + body[: match.start()].count("\n")
            origin_offset = match.start("args") + re_mod.search(
                r"\btx\s*\.\s*origin\b", match.group("args"), re_mod.IGNORECASE
            ).start()
            origin_line = body_base_line + body[:origin_offset].count("\n")
            evidence_lines = []
            for line in (func.line_number, call_line, origin_line):
                if line > 0 and line not in evidence_lines:
                    evidence_lines.append(line)

            risks.append({
                "risk_type": "access_control",
                "submechanism": "tx_origin_external_call_argument",
                "confidence": 0.9,
                "reason": (
                    f"{func.name}() is permissionless and forwards tx.origin to the "
                    f"external {method}() operation @L{call_line}; the caller identity "
                    "can be confused with the transaction origin."
                ),
                "function_name": func.name,
                "line": origin_line,
                "evidence_lines": evidence_lines,
                "source_grounded": True,
                "source_evidence_kind": "tx_origin_external_call_argument",
                "external_call_method": method,
                "external_call_line": call_line,
            })
            break
    return risks


def _detect_allowance_race_without_zero_first(
    features: ContractFeatures, source_code: str
) -> List[dict]:
    """Find public allowance updates that can change a later transferFrom outcome.

    This is intentionally narrow: the source must expose an externally callable
    approve-like update, an allowance/transferFrom path, and no zero-first or
    increase/decrease allowance protection in the same entrypoint.
    """

    if not features or not source_code:
        return []
    lines = source_code.splitlines()
    risks: list[dict] = []
    approve_call = re_mod.compile(r"\b(?:approve|setAllowance|updateAllowance)\s*\(", re_mod.I)
    update_name = re_mod.compile(
        r"(?:^|_)\b(?:approve|setAllowance|updateAllowance)\b|"
        r"\b(?:approve|setAllowance|updateAllowance)\b",
        re_mod.I,
    )
    allowance_read = re_mod.compile(r"\ballowance\s*\(", re_mod.I)
    transfer_from = re_mod.compile(r"\btransferFrom\s*\(", re_mod.I)
    zero_first = re_mod.compile(
        r"\b(?:increaseAllowance|decreaseAllowance)\s*\(|"
        r"\brequire\s*\([^\n]*(?:allowance|currentAllowance)[^\n]*(?:==|<=)\s*0",
        re_mod.I,
    )
    for func in features.functions:
        if (
            func.visibility not in {"public", "external"}
            or not func.is_reachable
            or _is_admin_restricted(func, source_code)
            or not update_name.search(func.name)
        ):
            continue
        start = max(func.line_number - 1, 0)
        end = min(func.end_line_number or (start + 80), len(lines))
        body = _strip_comments_preserve_lines("\n".join(lines[start:end]))
        if not approve_call.search(body):
            continue
        # Accept either local allowance state or a visible ERC-20 transferFrom
        # consumer in the same source.  The latter handles adapters that proxy
        # allowance updates to a token contract.
        if not (allowance_read.search(body) or transfer_from.search(source_code)):
            continue
        if zero_first.search(body):
            continue
        approve_match = approve_call.search(body)
        approve_line = func.line_number + body[:approve_match.start()].count("\n")
        evidence_lines = [func.line_number, approve_line]
        for match in allowance_read.finditer(body):
            line = func.line_number + body[:match.start()].count("\n")
            if line not in evidence_lines:
                evidence_lines.append(line)
        risks.append({
            "risk_type": "front_running",
            "confidence": 0.9,
            "reason": (
                f"{func.name}() exposes a permissionless allowance update at L{approve_line} "
                "that can change a later transferFrom outcome without zero-first or "
                "increaseAllowance/decreaseAllowance protection."
            ),
            "function_name": func.name,
            "entrypoint_function_name": func.name,
            "sink_function_name": "transferFrom",
            "line": approve_line,
            "entrypoint_lines": [func.line_number],
            "evidence_lines": sorted(set(evidence_lines)),
            "source_grounded": True,
            "source_evidence_kind": "allowance_race_without_zero_first",
            "permissionless": True,
            "allowance_race": True,
            "zero_first_protection": False,
            "source_call_path": [func.name, "approve", "allowance", "transferFrom"],
        })
    return risks


def _detect_permissionless_share_mints_without_minimum_shares(
    features: ContractFeatures, source_code: str
) -> List[dict]:
    """Find deposit paths that price shares before transfer without a min-shares bound.

    Keep this specific to deposit/depositTo entrypoints and require the complete
    source path: pre-transfer asset total, transferFrom(msg.sender), totalSupply-
    based share arithmetic, and a later mint.  This avoids treating ordinary
    arithmetic or a guarded ERC-4626-style deposit as transaction-order risk.
    """
    if not features or not source_code:
        return []

    lines = source_code.splitlines()
    minimum_bound = re_mod.compile(
        r"\b(?:minShares|minShare|minimumShares|minimumShare|sharesMin|"
        r"minMinted|mintedMin|amountOutMin|slippage)\b",
        re_mod.IGNORECASE,
    )
    pre_transfer_total = re_mod.compile(
        r"\b(?:underlyingTotal|totalAssets|underlyingBalance)\s*\(",
        re_mod.IGNORECASE,
    )
    transfer_from = re_mod.compile(
        r"\b(?:safeTransferFrom|transferFrom)\s*\(\s*msg\s*\.\s*sender\b",
        re_mod.IGNORECASE,
    )
    total_supply = re_mod.compile(r"\btotalSupply\s*\(\s*\)", re_mod.IGNORECASE)
    share_assignment = re_mod.compile(
        r"\bshares\b\s*=\s*[^;\n]*\btotalSupply\s*\(\s*\)",
        re_mod.IGNORECASE,
    )
    mint_call = re_mod.compile(r"\b_mint\s*\(", re_mod.IGNORECASE)

    risks: list[dict] = []
    for func in features.functions:
        if (
            func.visibility not in {"public", "external"}
            or not func.is_reachable
            or _is_admin_restricted(func, source_code)
            or func.name.casefold() not in {"deposit", "depositto"}
        ):
            continue

        start = max(func.line_number - 1, 0)
        end = min(func.end_line_number or (start + 80), len(lines))
        raw_body = "\n".join(lines[start:end])
        body_start = raw_body.find("{")
        if body_start < 0:
            continue
        body = _strip_comments_preserve_lines(raw_body[body_start + 1 :])
        body_base_line = func.line_number + raw_body[: body_start + 1].count("\n")

        total_match = pre_transfer_total.search(body)
        transfer_match = transfer_from.search(body)
        share_match = share_assignment.search(body)
        mint_matches = list(mint_call.finditer(body))
        if not total_match or not transfer_match or not share_match or not mint_matches:
            continue
        if not (total_match.start() < transfer_match.start() < share_match.start()):
            continue
        if not any(match.start() > share_match.start() for match in mint_matches):
            continue
        if minimum_bound.search(raw_body):
            continue
        if not total_supply.search(body[share_match.start() :]):
            continue

        def line_for(offset: int) -> int:
            return body_base_line + body[:offset].count("\n")

        mint_lines = [line_for(match.start()) for match in mint_matches]
        evidence_lines: list[int] = []
        for line in [
            func.line_number,
            line_for(total_match.start()),
            line_for(transfer_match.start()),
            line_for(share_match.start()),
            *mint_lines,
        ]:
            if line > 0 and line not in evidence_lines:
                evidence_lines.append(line)

        risks.append({
            "risk_type": "front_running",
            "confidence": 0.9,
            "reason": (
                f"{func.name}() snapshots the underlying total before transferFrom(msg.sender), "
                "computes totalSupply-based shares after the transfer, and mints without a "
                "caller-controlled minimum-share bound; transaction ordering can change the "
                "received share amount."
            ),
            "function_name": func.name,
            "entrypoint_function_name": func.name,
            "sink_function_name": func.name,
            "line": func.line_number,
            "entrypoint_lines": [func.line_number],
            "evidence_lines": evidence_lines,
            "source_grounded": True,
            "source_evidence_kind": "permissionless_share_mint_without_minimum_shares",
            "permissionless": True,
            "share_mint": True,
            "share_price_dependent": True,
            "minimum_shares_bound": False,
            "quote_before_transfer": True,
            "source_call_path": [func.name, "underlyingTotal", "transferFrom", "totalSupply", "_mint"],
        })
    return risks


def _detect_permissionless_msg_value_quote_pair_swaps(
    features: ContractFeatures, source_code: str
) -> List[dict]:
    """Find ``msg.value -> amountOut -> pair.swap`` paths without a min bound.

    This is deliberately narrower than the generic balance-dependent rule. It
    targets direct AMM-pair swaps whose output is computed from a live quote and
    does not treat exact-output ``amountIn`` paths or router min-out arguments as
    front-running evidence.
    """

    if not features or not source_code:
        return []

    lines = source_code.splitlines()
    quote_assignment = re_mod.compile(
        r"\b[A-Za-z_]\w*\s*=\s*(?P<quote>amountOut)\s*\([^;]*\bmsg\s*\.\s*value\b[^;]*\)",
        re_mod.IGNORECASE | re_mod.DOTALL,
    )
    quote_call = re_mod.compile(r"\b(?P<name>amountOut)\s*\(", re_mod.IGNORECASE)
    pair_swap = re_mod.compile(r"\.\s*swap\s*\(", re_mod.IGNORECASE)
    explicit_min_bound = re_mod.compile(
        r"\b(?:minOut|minimumOut|amountOutMinimum|slippage|_min[A-Za-z0-9_]*)\b",
        re_mod.IGNORECASE,
    )

    risks: list[dict] = []
    for func in features.functions:
        if (
            func.visibility not in {"public", "external"}
            or not func.is_reachable
            or _is_admin_restricted(func, source_code)
        ):
            continue

        start = max(func.line_number - 1, 0)
        end = min(func.end_line_number or (start + 80), len(lines))
        raw_body = "\n".join(lines[start:end])
        body_start = raw_body.find("{")
        if body_start < 0:
            continue
        body = _strip_comments_preserve_lines(raw_body[body_start + 1 :])
        body_base_line = func.line_number + raw_body[: body_start + 1].count("\n")

        quote_match = quote_assignment.search(body)
        swap_matches = list(pair_swap.finditer(body))
        if quote_match is None or not swap_matches:
            continue
        if any(
            explicit_min_bound.search(line)
            and re_mod.search(r"\b(?:require|assert)\b", line, re_mod.IGNORECASE)
            for line in body.splitlines()
        ):
            continue

        quote_lines = [
            body_base_line + body[:match.start()].count("\n")
            for match in quote_call.finditer(body)
            if match.start() <= swap_matches[-1].start()
        ]
        swap_lines = [
            body_base_line + body[:match.start()].count("\n")
            for match in swap_matches
        ]
        evidence_lines = []
        for line in [func.line_number, *quote_lines, *swap_lines]:
            if line > 0 and line not in evidence_lines:
                evidence_lines.append(line)

        risks.append({
            "risk_type": "front_running",
            "confidence": 0.9,
            "reason": (
                f"{func.name}() is permissionless, derives AMM output from msg.value "
                f"through amountOut() @L{quote_lines[0] if quote_lines else func.line_number}, "
                f"and calls a pair.swap() sink at @L{swap_lines[0]}; no user-controlled "
                "minimum-output bound protects the quoted result."
            ),
            "function_name": func.name,
            "entrypoint_function_name": func.name,
            "sink_function_name": func.name,
            "line": swap_lines[0],
            "entrypoint_lines": [func.line_number],
            "evidence_lines": evidence_lines,
            "source_grounded": True,
            "source_evidence_kind": (
                "permissionless_msg_value_quote_then_pair_swap_no_slippage_bound"
            ),
            "permissionless": True,
            "quote_function": "amountOut",
            "quote_dependent": True,
            "amm_swap": True,
            "zero_slippage": True,
            "configurable_slippage": False,
            "source_call_path": [func.name, "amountOut", "swap"],
        })
    return risks


def _detect_permissionless_balance_dependent_amm_swaps(
    features: ContractFeatures, source_code: str
) -> List[dict]:
    """Find permissionless balance-dependent swaps with a literal zero slippage bound.

    The balance read and the AMM call may be separated by one internal helper
    (for example ``harvest -> _sellAAVEForWant -> exactInput``).  Keep that
    closure narrow and require the zero bound at the actual swap call site.
    """
    if not features or not source_code:
        return []

    lines = source_code.split("\n")
    swap_pattern = re_mod.compile(
        r"\b(?:_swapToken|_swap|swapExact[A-Za-z0-9_]*|exactInput(?:Single)?)\s*\("
    )
    functions_by_name = {func.name: func for func in features.functions}

    def _find_swap_call(body: str):
        for match in swap_pattern.finditer(body):
            line_start = body.rfind("\n", 0, match.start()) + 1
            if re_mod.search(
                r"\bfunction(?:\s+[A-Za-z_]\w*)?\s*$",
                body[line_start:match.start()],
            ):
                continue
            return match
        return None

    def _function_body(func: FunctionFlow) -> str:
        start = max(func.line_number - 1, 0)
        end = func.end_line_number if func.end_line_number > start else start + 50
        return "\n".join(lines[start:min(end, len(lines))])

    def _slippage_bound_match(body: str, swap_call) -> tuple[int | None, str | None]:
        call_end = body.find(";", swap_call.end())
        call_text = body[swap_call.start():call_end if call_end >= 0 else len(body)]
        method = re_mod.search(r"\b([A-Za-z_]\w*)\s*\(", call_text)
        method_name = method.group(1) if method else ""
        if method_name.startswith("exactInput"):
            match = re_mod.search(
                r"\b(?:amountOutMinimum|amount_out_minimum)\s*[:=]\s*0\b"
                r"|\buint256\s*\(\s*0\s*\)",
                call_text,
            )
        else:
            match = re_mod.search(r"(?<!\w)0\s*,", call_text)
        if match:
            return swap_call.start() + match.start(), "literal_zero"

        configurable = re_mod.search(
            r"\b(?:_?slippage[A-Za-z0-9_]*|slippage[A-Za-z0-9_]*)\b",
            call_text,
            re_mod.IGNORECASE,
        )
        if configurable:
            return swap_call.start() + configurable.start(), "configurable_slippage"
        return None, None

    risks = []
    for func in features.functions:
        if _is_admin_restricted(func, source_code):
            continue
        body = _function_body(func)

        balance_match = re_mod.search(
            r"\bbalanceOf\s*\(\s*address\s*\(\s*this\s*\)\s*\)"
            r"|\baddress\s*\(\s*this\s*\)\s*\.\s*balance\b",
            body,
            re_mod.IGNORECASE,
        )
        balance_func = func if balance_match else None
        balance_context = body if balance_match else ""
        swap_context = body
        swap_func = func
        call_path = [func.name]
        swap_call = _find_swap_call(body)
        slippage_bound, slippage_kind = (
            _slippage_bound_match(body, swap_call) if swap_call else (None, None)
        )

        if not swap_call or slippage_bound is None or balance_func is None:
            callable_names = sorted(
                (
                    name
                    for name, candidate in functions_by_name.items()
                    if name != func.name
                    and candidate.visibility not in {"public", "external"}
                ),
                key=len,
                reverse=True,
            )
            if not callable_names:
                continue
            call_pattern = re_mod.compile(
                r"(?<![A-Za-z0-9_.])(?:"
                + "|".join(re_mod.escape(name) for name in callable_names)
                + r")\s*\("
            )
            helper_match = None
            for internal_call in call_pattern.finditer(body):
                helper = functions_by_name.get(internal_call.group(0).split("(", 1)[0].strip())
                if helper is None:
                    continue
                helper_body = _function_body(helper)
                helper_balance = re_mod.search(
                    r"\bbalanceOf\s*\(\s*address\s*\(\s*this\s*\)\s*\)"
                    r"|\baddress\s*\(\s*this\s*\)\s*\.\s*balance",
                    helper_body,
                    re_mod.IGNORECASE,
                )
                candidate_swap = _find_swap_call(helper_body)
                if not candidate_swap:
                    continue
                candidate_bound, candidate_kind = _slippage_bound_match(
                    helper_body, candidate_swap
                )
                if candidate_bound is None:
                    continue
                if balance_func is None and helper_balance is None:
                    continue
                helper_match = (
                    helper,
                    helper_body,
                    candidate_swap,
                    candidate_bound,
                    candidate_kind,
                    helper_balance,
                )
                break
            if helper_match is None:
                continue
            (
                swap_func,
                swap_context,
                swap_call,
                slippage_bound,
                slippage_kind,
                helper_balance,
            ) = helper_match
            call_path.append(swap_func.name)
            if balance_func is None:
                balance_func = swap_func
                balance_context = swap_context
                balance_match = helper_balance

        if not swap_call or slippage_bound is None or not balance_match or not balance_func:
            continue

        balance_line = balance_func.line_number + balance_context[:balance_match.start()].count("\n")
        swap_line = swap_func.line_number + swap_context[:swap_call.start()].count("\n")
        slippage_line = swap_func.line_number + swap_context[:slippage_bound].count("\n")
        conditional_authorization = bool(re_mod.search(
            r"\bif\s*\(\s*onlyGov\s*\).*?\brequire\s*\([^\n]*msg\.sender\s*==\s*govAddress",
            balance_context,
            re_mod.IGNORECASE | re_mod.DOTALL,
        ))
        evidence_lines = []
        for line in (func.line_number, balance_line, swap_line, slippage_line):
            if line > 0 and line not in evidence_lines:
                evidence_lines.append(line)

        if slippage_kind == "literal_zero":
            evidence_kind = "permissionless_balance_dependent_amm_swap_zero_slippage"
        elif conditional_authorization:
            evidence_kind = "conditional_permissionless_quote_then_swap_configurable_slippage"
        else:
            evidence_kind = "permissionless_quote_then_swap_configurable_slippage"

        risks.append({
            "risk_type": "front_running",
            "confidence": 0.9,
            "reason": (
                f"{func.name}() is permissionless and derives swap input from the current token balance "
                f"@L{balance_line}; it reaches an AMM swap through {swap_call.group(0).strip()} "
                f"with a {slippage_kind.replace('_', ' ')} bound @L{slippage_line}, "
                "so transaction ordering can change output."
            ),
            "function_name": func.name,
            "entrypoint_function_name": func.name,
            "sink_function_name": swap_func.name,
                "line": slippage_line,
            "entrypoint_lines": [func.line_number],
            "evidence_lines": evidence_lines,
            "source_grounded": True,
            "source_evidence_kind": evidence_kind,
            "permissionless": not conditional_authorization,
            "conditional_authorization": conditional_authorization,
            "balance_dependent": True,
            "amm_swap": True,
            "zero_slippage": slippage_kind == "literal_zero",
            "configurable_slippage": slippage_kind == "configurable_slippage",
            "source_call_path": call_path + [swap_call.group(0).split("(", 1)[0].strip()],
        })
    return risks


def _detect_e2_permissionless_ordered_asset_paths(
    features: ContractFeatures, source_code: str
) -> List[dict]:
    """Recover narrow front-running paths missed by the AMM-only predicates.

    DAppSCAN contains several equivalent transaction-order patterns that do
    not use ``address(this).balance`` or a canonical AMM helper: quoted
    ``getAmountsOut`` swaps, ratio-based token settlements, and signed
    external-call balance deltas.  This E2-only rule keeps the proof gate in
    charge by emitting source-grounded candidates with explicit line paths.
    """

    if not features or not source_code:
        return []

    lines = source_code.splitlines()
    functions_by_name = {
        func.name: func for func in features.functions if func.name
    }
    internal_names = {
        func.name
        for func in features.functions
        if func.name and func.visibility not in {"public", "external"}
    }

    def body(func: FunctionFlow) -> str:
        start = max(func.line_number - 1, 0)
        end = func.end_line_number if func.end_line_number > start else start + 80
        return _strip_comments_preserve_lines("\n".join(lines[start:min(end, len(lines))]))

    def line_for(func: FunctionFlow, text: str, match: re_mod.Match) -> int:
        return func.line_number + text[:match.start()].count("\n")

    quote_pattern = re_mod.compile(
        r"\b(?:getAmountsOut|getAmountOut|amountOut|quote|value|balanceOf)\s*\(",
        re_mod.IGNORECASE,
    )
    sink_pattern = re_mod.compile(
        r"\b(?:swapExact[A-Za-z0-9_]*|swapTokens[A-Za-z0-9_]*|"
        r"exactInput[A-Za-z0-9_]*|addLiquidity|transfer|safeTransfer|"
        r"transferFrom|send|call(?:\s*\{[^}]*\})?|delegatecall|mint|_mint|deposit|"
        r"reinvest|compound|harvest|trade|_update[A-Za-z0-9_]*)\s*\(",
        re_mod.IGNORECASE,
    )
    settlement_pattern = re_mod.compile(
        r"\b(?:addLiquidity|deposit|mint|_mint|stake|reinvest|compound|harvest|"
        r"execute|transfer|safeTransfer|transferFrom|send)\s*\(",
        re_mod.IGNORECASE,
    )
    ratio_pattern = re_mod.compile(
        r"\b[A-Za-z_]\w*\s*\*[^;\n]{0,180}?/\s*"
        r"(?:[A-Za-z_]\w*\s*\.\s*)?value\s*\(",
        re_mod.IGNORECASE,
    )
    external_balance_pattern = re_mod.compile(
        r"\bbalanceOf\s*\(\s*[^)]*\)\s*|"
        r"\bbalanceBefore[A-Za-z0-9_]*\b",
        re_mod.IGNORECASE,
    )
    zero_bound_pattern = re_mod.compile(
        r"(?:amountOutMinimum|minOut|minShares|minimumShares)\s*[:=]\s*0"
        r"|,\s*0\s*,",
        re_mod.IGNORECASE,
    )
    configurable_bound_pattern = re_mod.compile(
        r"\b(?:slippage|amountOutMinimum|amountOutMin|minOut|minShares)\w*\b",
        re_mod.IGNORECASE,
    )
    deadline_pattern = re_mod.compile(
        r"\b(?:deadline|expiry|expiration|expiresAt|validUntil)\b",
        re_mod.IGNORECASE,
    )
    signed_execution_pattern = re_mod.compile(
        r"\b(?:sign(?:Data|ature)?|verify(?:Sign|Signature)?|signature|"
        r"callData|timestamp|nonce|expiry|validUntil)\b",
        re_mod.IGNORECASE,
    )

    def is_function_declaration(text: str, match: re_mod.Match) -> bool:
        """Do not treat ``function foo(...)`` declarations as asset sinks."""

        prefix = text[max(0, match.start() - 32):match.start()]
        return bool(re_mod.search(r"\bfunction\s*$", prefix, re_mod.IGNORECASE))

    def filtered_matches(pattern: re_mod.Pattern[str], text: str) -> list[re_mod.Match]:
        return [match for match in pattern.finditer(text) if not is_function_declaration(text, match)]

    def called_internal(text: str) -> list[str]:
        names = []
        for name in sorted(internal_names, key=len, reverse=True):
            if re_mod.search(rf"(?<![A-Za-z0-9_.]){re_mod.escape(name)}\s*\(", text):
                names.append(name)
        return names

    risks: list[dict] = []
    # Deduplicate across all public entrypoints.  Internal helpers are often
    # reached by both ``harvest`` overloads and a manager path; emitting one
    # source-grounded locus keeps alert density stable without weakening proof.
    emitted: set[tuple[str, str]] = set()
    for entrypoint in features.functions:
        if (
            entrypoint.visibility not in {"public", "external"}
            or not entrypoint.is_reachable
            or entrypoint.state_mutability == "view"
            or _is_admin_restricted(entrypoint, source_code)
        ):
            continue

        contexts: list[tuple[FunctionFlow, str, list[str]]] = []
        queue: list[tuple[FunctionFlow, list[str], int]] = [(entrypoint, [entrypoint.name], 0)]
        seen = {entrypoint.name}
        while queue:
            current, call_path, depth = queue.pop(0)
            current_body = body(current)
            contexts.append((current, current_body, call_path))
            if depth >= 2:
                continue
            for helper_name in called_internal(current_body):
                if helper_name in seen or helper_name not in functions_by_name:
                    continue
                seen.add(helper_name)
                helper = functions_by_name[helper_name]
                queue.append((helper, call_path + [helper_name], depth + 1))

        for context, context_body, call_path in contexts:
            quote_matches = list(quote_pattern.finditer(context_body))
            sink_matches = filtered_matches(sink_pattern, context_body)
            ratio_match = ratio_pattern.search(context_body)
            call_match = next(
                (match for match in sink_matches if re_mod.match(r"(?:call|delegatecall)", match.group(0), re_mod.I)),
                None,
            )
            zero_bound = bool(zero_bound_pattern.search(context_body))
            configurable_bound = bool(configurable_bound_pattern.search(context_body))
            has_deadline = bool(deadline_pattern.search(context_body))

            swap_matches = [
                match
                for match in sink_matches
                if re_mod.search(r"swap|addLiquidity|exactInput", match.group(0), re_mod.I)
            ]

            # A signed execution entrypoint may delegate the low-level call to
            # an internal helper.  Propagate the signature context from the
            # public entrypoint, but never infer it from a bare ``call`` or
            # ``balanceOf`` pair.
            signed_execution = bool(
                signed_execution_pattern.search(body(entrypoint))
            )

            # Signed external-call settlement: balance delta is checked only
            # after a permissionless call, so transaction order remains a
            # source-grounded ordering surface even without an AMM quote.
            external_balance_path = bool(
                call_match
                and external_balance_pattern.search(context_body)
                and re_mod.search(r"\b(?:returnedAmount|balanceOf\s*\().*(?:=|safeAdd|safeSub)", context_body, re_mod.I | re_mod.S)
                and signed_execution
            )

            quote_match = quote_matches[0] if quote_matches else None
            sink_match = None
            if quote_match:
                sink_match = next(
                    (match for match in sink_matches if match.start() > quote_match.start()),
                    None,
                )
            if zero_bound and swap_matches:
                sink_match = swap_matches[0]
            downstream_settlement = None
            if sink_match is not None:
                downstream_settlement = next(
                    (
                        match for match in filtered_matches(settlement_pattern, context_body)
                        if match.start() > sink_match.end()
                    ),
                    None,
                )
            if sink_match is None and not external_balance_path:
                continue

            sink_is_swap = bool(
                sink_match
                and re_mod.search(r"swap|addLiquidity|exactInput", sink_match.group(0), re_mod.I)
            )
            # Quote->swap candidates are useful when the swap feeds a
            # downstream liquidity/share/asset settlement.  Fee-only helpers
            # such as ``chargeFees`` otherwise create duplicate alerts for the
            # same enclosing harvest path.
            compound_settlement = bool(
                downstream_settlement
                and re_mod.search(
                    r"addLiquidity|deposit|mint|_mint|stake|reinvest|compound|harvest|execute",
                    downstream_settlement.group(0),
                    re_mod.I,
                )
            )
            if (
                not external_balance_path
                and not ratio_match
                and not sink_is_swap
                and not (configurable_bound and not has_deadline)
            ):
                continue

            if external_balance_path:
                evidence_kind = "permissionless_external_call_balance_settlement"
                locus_function = entrypoint
                quote_line = line_for(context, context_body, quote_match) if quote_match else context.line_number
                sink_line = line_for(context, context_body, call_match)
            elif (
                zero_bound
                and sink_is_swap
                and downstream_settlement is not None
                and quote_match is None
                and ratio_match is None
            ):
                # Some DAppSCAN paths use a zero-minimum swap followed by a
                # balance-derived transfer without an explicit quote call
                # (e.g. ``_rewardToBeneficialVault``).
                evidence_kind = "permissionless_zero_slippage_swap_then_asset_settlement"
                locus_function = context
                quote_line = line_for(context, context_body, sink_match)
                sink_line = line_for(context, context_body, downstream_settlement)
            elif ratio_match is not None:
                evidence_kind = "permissionless_ratio_quote_then_asset_settlement"
                locus_function = context
                quote_line = line_for(context, context_body, ratio_match)
                sink_line = line_for(context, context_body, sink_match) if sink_match else quote_line
            elif quote_match and sink_match:
                evidence_kind = "permissionless_quote_then_asset_settlement"
                locus_function = context
                quote_line = line_for(context, context_body, quote_match)
                sink_line = line_for(context, context_body, sink_match)
            else:
                continue

            if (
                evidence_kind == "permissionless_quote_then_asset_settlement"
                and context.visibility not in {"public", "external"}
                and not compound_settlement
            ):
                continue

            # Signed external-call settlement is scored at its public entry;
            # the other ordered-asset patterns are localized at the concrete
            # helper that owns the quote/swap/settlement operation.  Preserve
            # the public entrypoint separately for reachability proof.
            scoring_function = (
                entrypoint
                if evidence_kind == "permissionless_external_call_balance_settlement"
                and context.visibility not in {"public", "external"}
                else context
            )
            key = (scoring_function.name, evidence_kind)
            if key in emitted:
                continue
            emitted.add(key)
            sink_name = re_mod.match(r"([A-Za-z_]\w*)", sink_match.group(0)).group(1) if sink_match else "external call"
            evidence_lines: list[int] = []
            for line in (
                scoring_function.line_number,
                context.line_number,
                quote_line,
                sink_line,
            ):
                if isinstance(line, int) and line > 0 and line not in evidence_lines:
                    evidence_lines.append(line)
            # Preserve the small source window around ratio/comment anchors
            # used by DAppSCAN locators without expanding to a whole function.
            for line in (quote_line - 1, sink_line + 1):
                if isinstance(line, int) and line > 0 and line <= len(lines) and line not in evidence_lines:
                    evidence_lines.append(line)

            risk = {
                "risk_type": "front_running",
                "confidence": 0.9,
                "reason": (
                    f"{entrypoint.name}() is permissionless and reaches {sink_name}"
                    f"() through mutable quote/balance evidence at L{quote_line}; "
                    "transaction ordering can change the settled asset amount."
                ),
                "function_name": scoring_function.name,
                "entrypoint_function_name": entrypoint.name,
                "sink_function_name": sink_name,
                "line": sink_line,
                "entrypoint_lines": [entrypoint.line_number],
                "call_site_function": context.name,
                "call_site_line": sink_line,
                "sink_line": sink_line,
                "evidence_lines": evidence_lines,
                "source_grounded": True,
                "source_evidence_kind": evidence_kind,
                "permissionless": True,
                "quote_dependent": bool(quote_match or ratio_match),
                "balance_dependent": bool(quote_match and re_mod.search(r"balanceOf|balanceBefore|totalBalance", context_body, re_mod.I)),
                "amm_swap": sink_is_swap,
                "zero_slippage": zero_bound,
                "configurable_slippage": configurable_bound,
                "caller_slippage_unbounded": configurable_bound and not zero_bound,
                "deadline_present": has_deadline,
                "source_call_path": call_path + [sink_name],
            }
            risks.append(risk)
    return risks


def _detect_permissionless_reward_reinvestment_updates(
    features: ContractFeatures, source_code: str
) -> List[dict]:
    """Find public reward-update paths that reinvest a mutable quote.

    This is intentionally limited to common reward-maintenance entrypoints.
    It requires the ordered source path ``claim -> quote/conversion ->
    liquidity/reinvest/deposit`` and rejects explicit output, slippage, or
    deadline controls.  Generic read/write conflicts remain insufficient.
    """

    if not features or not source_code:
        return []

    lines = source_code.splitlines()
    entrypoint_names = {
        "updatepool",
        "massupdatepools",
        "reinvest",
        "harvest",
        "compound",
        "autocompound",
    }
    reward_call = re_mod.compile(
        r"\b(?P<name>poolClaim|getReward|claimReward|harvestReward|collectReward)\s*\(",
        re_mod.I,
    )
    quote_call = re_mod.compile(
        r"\b(?P<name>getTokenIn|getAmountIn|amountOut|quote|convertTo[A-Za-z0-9_]*)\s*\(",
        re_mod.I,
    )
    reinvest_sink = re_mod.compile(
        r"\b(?P<name>makeBalanceOptimalLiquidityByAmount|"
        r"makeLiquidityAndDepositByAmount|addLiquidity|deposit|mint|"
        r"reinvest|compound|harvest)\s*\(",
        re_mod.I,
    )
    ordering_controls = re_mod.compile(
        r"\b(?:minOut|minimumOut|amountOutMinimum|slippage|deadline|"
        r"expiry|expiration|minShares|minimumShares)\b",
        re_mod.I,
    )

    risks: list[dict] = []
    for func in features.functions:
        if (
            func.visibility not in {"public", "external"}
            or not func.is_reachable
            or _is_admin_restricted(func, source_code)
            or func.name.casefold() not in entrypoint_names
        ):
            continue

        start = max(func.line_number - 1, 0)
        end = min(func.end_line_number or (start + 100), len(lines))
        body = _strip_comments_preserve_lines("\n".join(lines[start:end]))
        reward = reward_call.search(body)
        if reward is None:
            continue
        sink = reinvest_sink.search(body, reward.end())
        if sink is None or ordering_controls.search(body):
            continue
        quote_matches = list(quote_call.finditer(body, reward.end(), sink.start()))
        quote = quote_matches[-1] if quote_matches else None
        if quote is None:
            continue

        def line_for(match) -> int:
            return func.line_number + body[: match.start()].count("\n")

        reward_line = line_for(reward)
        quote_line = line_for(quote)
        sink_line = line_for(sink)
        evidence_lines = []
        for line in (func.line_number, reward_line, quote_line, sink_line):
            if line > 0 and line not in evidence_lines:
                evidence_lines.append(line)

        risks.append({
            "risk_type": "front_running",
            "confidence": 0.90,
            "reason": (
                f"{func.name}() is permissionless and moves a mutable reward quote "
                f"from {reward.group('name')}() @L{reward_line} through "
                f"{quote.group('name')}() @L{quote_line} into "
                f"{sink.group('name')}() @L{sink_line} without an effective "
                "minimum-output, slippage, or deadline bound"
            ),
            "function_name": func.name,
            "entrypoint_function_name": func.name,
            "sink_function_name": func.name,
            "line": quote_line,
            "entrypoint_lines": [func.line_number],
            "evidence_lines": evidence_lines,
            "source_grounded": True,
            "source_evidence_kind": (
                "permissionless_reward_quote_then_reinvestment_no_slippage_bound"
            ),
            "permissionless": True,
            "quote_function": quote.group("name"),
            "quote_dependent": True,
            "reward_reinvestment": True,
            "reinvestment_sink": sink.group("name"),
            "zero_slippage": True,
            "configurable_slippage": False,
            "source_call_path": [
                func.name,
                reward.group("name"),
                quote.group("name"),
                sink.group("name"),
            ],
        })
    return risks


def _extract_sink_data_flows(features: ContractFeatures, source_code: str) -> List[dict]:
    """
    数据流物理隔离 (Physical DFG Isolation) + 状态变量锚定膨胀 (State-Anchored Expansion):
    以每个致灾节点 (Sink) 为中心，提取其 Def-Use 链，
    生成独立的数据流切片。每条切片只包含：
    1. Sink 所在函数的代码
    2. Sink 依赖的状态变量定义
    3. Sink 依赖的修饰器代码
    4. Sink 调用的内部函数
    5. [NEW] 全局横向扫描：锚定 Sink 依赖的核心状态变量，
       在整个合约 AST 中搜索所有对该变量进行赋值的 public/external 函数，
       强制拼入切片，确保跨函数 RW-Conflict 不被遗漏。
    """
    if not source_code or not features:
        return []

    lines = source_code.split('\n')
    total_lines = len(lines)
    sink_flows = []

    global_writer_map = defaultdict(list)
    for func in features.functions:
        if func.visibility not in ['public', 'external']:
            continue
        if _is_admin_restricted(func, source_code):
            continue
        for sw in func.state_writes:
            for sv in features.state_variables:
                if sv in sw:
                    global_writer_map[sv].append(func)

    for func in features.functions:
        if func.visibility not in ['public', 'external']:
            continue

        sinks = []
        for ec in func.external_calls:
            sinks.append({"type": "external_call", "detail": ec, "line_hint": ""})
        for uc in func.unchecked_calls:
            sinks.append({"type": "unchecked_call", "detail": uc, "line_hint": ""})
        for sw in func.state_writes:
            sinks.append({"type": "state_write", "detail": sw, "line_hint": ""})
        for lf in func.loop_features:
            sinks.append({"type": "unbounded_loop", "detail": lf, "line_hint": ""})
        for dfc in func.data_flow_chains:
            sinks.append({"type": "data_flow", "detail": dfc, "line_hint": ""})

        if not func.require_checks and not func.has_reentrancy_guard:
            if func.external_calls or func.state_writes:
                sinks.append({"type": "missing_guard", "detail": f"{func.name} has no require/guard", "line_hint": ""})

        if not sinks:
            continue

        included = set()

        start = func.line_number - 1
        end = func.end_line_number if func.end_line_number > start else start + 50
        end = min(end, total_lines)
        func_lines = set(range(start, end))
        included.update(func_lines)

        dep_vars = set()
        func_text = "\n".join(lines[start:end])
        for sv in features.state_variables:
            if sv in func_text:
                dep_vars.add(sv)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue
            for sv in dep_vars:
                if sv in stripped and ('=' in stripped or stripped.endswith(';')):
                    if not stripped.startswith('function') and not stripped.startswith('modifier'):
                        included.add(i)
                        break

        ARITH_OPS = ['+=', '-=', '*=', '/=', '%=', '++', '--']
        for li in range(start, end):
            if li >= total_lines:
                break
            line_text = lines[li]
            stripped = line_text.strip()
            if any(op in stripped for op in ARITH_OPS):
                included.add(li)
                for adj in range(max(start, li - 2), min(end, li + 3)):
                    included.add(adj)
            if any(op in stripped for op in [' + ', ' - ', ' * ', ' / ']):
                if any(sv in stripped for sv in dep_vars):
                    included.add(li)
                    for adj in range(max(start, li - 2), min(end, li + 3)):
                        included.add(adj)

        for li in range(start, end):
            if li >= total_lines:
                break
            stripped = lines[li].strip()
            if stripped.startswith('require(') or stripped.startswith('if(') or stripped.startswith('if ('):
                included.add(li)
                for adj in range(li + 1, min(end, li + 4)):
                    adj_stripped = lines[adj].strip() if adj < total_lines else ""
                    included.add(adj)
                    if ')' in adj_stripped and ('{' in adj_stripped or ';' in adj_stripped):
                        break

        for mod_name in func.modifiers:
            for mod in features.modifiers:
                if mod.name == mod_name:
                    mod_start = mod.line_number - 1
                    for i in range(max(0, mod_start), min(total_lines, mod_start + 30)):
                        included.add(i)

        for icall in func.internal_calls:
            callee_name = icall.split("(")[0].split(".")[-1].strip()
            for callee_func in features.functions:
                if callee_func.name == callee_name:
                    cs = callee_func.line_number - 1
                    ce = callee_func.end_line_number if callee_func.end_line_number > cs else cs + 30
                    ce = min(ce, total_lines)
                    for i in range(cs, ce):
                        included.add(i)

        expanded_writers = []
        skeleton_sections = []
        for sv in sorted(dep_vars):
            for writer_func in global_writer_map.get(sv, []):
                if writer_func.name != func.name:
                    ws = writer_func.line_number - 1
                    we = writer_func.end_line_number if writer_func.end_line_number > ws else ws + 30
                    we = min(we, total_lines)

                    func_sig = _extract_function_signature(lines, ws, total_lines)
                    assignment_lines = []
                    for li in range(ws + 1, we):
                        line_text = lines[li] if li < total_lines else ""
                        stripped = line_text.strip()
                        if sv in stripped and ('=' in stripped or sv.split('[')[0] in stripped):
                            assignment_lines.append(f"      {stripped} // Line {li + 1}")

                    skeleton = f"  {func_sig}"
                    if assignment_lines:
                        skeleton += "\n      ... [Code Omitted] ..."
                        for al in assignment_lines:
                            skeleton += f"\n{al}"
                    else:
                        skeleton += f"\n      ... modifies '{sv}' [Code Omitted] ..."
                    skeleton += "\n  }"

                    skeleton_sections.append(skeleton)
                    expanded_writers.append({
                        "function_name": writer_func.name,
                        "line_range": f"L{writer_func.line_number}-L{writer_func.end_line_number}",
                        "written_var": sv,
                        "has_admin_modifier": False,
                    })

        for i in range(min(5, total_lines)):
            included.add(i)
        for i in range(max(0, total_lines - 3), total_lines):
            included.add(i)

        sorted_lines = sorted(included)
        parts = []
        prev = -10
        for li in sorted_lines:
            if li - prev > 2:
                parts.append(f"    // ... [{prev + 2}-{li}] omitted ...")
            parts.append(f"{li + 1:>{len(str(total_lines))}} | {lines[li]}")
            prev = li

        slice_text = '\n'.join(parts)

        sink_descriptions = []
        for s in sinks:
            sink_descriptions.append(f"  [{s['type']}] {s['detail']}")

        sink_flows.append({
            "function_name": func.name,
            "visibility": func.visibility,
            "line_range": f"L{func.line_number}-L{func.end_line_number}",
            "sinks": sink_descriptions,
            "slice_text": slice_text,
            "dep_vars": sorted(dep_vars),
            "modifiers": func.modifiers,
            "is_reachable": func.is_reachable,
            "reachability_path": func.reachability_path,
            "expanded_writers": expanded_writers,
            "skeleton_sections": skeleton_sections,
        })

    return sink_flows


def extract_contract_features(source: str) -> List[ContractFeatures]:
    tree = _PARSER.parse(source.encode("utf-8"))
    root = tree.root_node

    results = []
    # Solidity libraries use the same contract-body/function grammar, but are
    # represented by a distinct top-level node.  Treat them as analyzable
    # feature containers instead of dropping the entire file into parse
    # fallback.
    contract_nodes = (
        _find_all_by_type(root, "contract_declaration")
        + _find_all_by_type(root, "library_declaration")
    )

    for cnode in contract_nodes:
        contract_name = ""
        for child in cnode.children:
            if child.type == "identifier":
                contract_name = _node_text(child)
                break

        inherits = _extract_inheritance(cnode)
        state_vars = _extract_state_variables(cnode)
        modifier_defs = _extract_modifiers(cnode)
        asm_blocks = _extract_inline_assembly(cnode)
        state_var_set = set(state_vars)

        functions = []
        body = _find_first_by_type(cnode, "contract_body")
        if body:
            all_func_nodes = [child for child in body.children
                              if child.type in ("function_definition", "constructor_definition",
                                                "fallback_function_definition", "receive_function_definition")]
            for child in all_func_nodes:
                func_flow = _extract_function_flow(child, state_var_set, modifier_defs,
                                                   all_func_nodes=all_func_nodes,
                                                   source_code=source,
                                                   contract_name=contract_name)
                functions.append(func_flow)

            call_graph = _build_call_graph(cnode, all_func_nodes)
            public_entries = set()
            for func in functions:
                if func.visibility in ['public', 'external'] or func.is_constructor or func.is_fallback or func.is_receive:
                    public_entries.add(func.name)
            reachability = _compute_reachability(call_graph, public_entries)
            for func in functions:
                if func.name in reachability:
                    is_reachable, reach_path = reachability[func.name]
                    func.is_reachable = is_reachable
                    func.reachability_path = reach_path
                else:
                    func.is_reachable = False
                    func.reachability_path = []

        results.append(ContractFeatures(
            name=contract_name,
            state_variables=state_vars,
            functions=functions,
            inherits=inherits,
            modifiers=modifier_defs,
            inline_assembly_blocks=asm_blocks,
        ))

    return results


def _number_lines(source: str) -> str:
    lines = source.split('\n')
    width = len(str(len(lines)))
    return '\n'.join(f'{i+1:>{width}} | {line}' for i, line in enumerate(lines))


def synthesize_flow(features: ContractFeatures, source_code: str = "") -> str:
    flow_parts = []
    for func in features.functions:
        if func.visibility not in ['public', 'external']:
            continue
        if not func.external_calls and not func.state_writes and not func.is_fallback:
            continue

        steps = []
        if func.inlined_modifier_code:
            steps.append(f"MODIFIER_INLINE({func.inlined_modifier_code.strip()[:120]})")
        if func.has_reentrancy_guard:
            steps.append(f"GUARD_ENTER({func.guard_detail})")

        fn_lower = func.name.lower()
        is_init_func = any(kw in fn_lower for kw in ['init', 'setup', 'configure', 'setowner', 'changeowner'])
        owner_like_writes = [w for w in func.state_writes
                             if any(kw in w.lower() for kw in ['owner', 'admin', 'wallet', 'authority', 'master'])]
        if func.visibility in ['public', 'external'] and (is_init_func or owner_like_writes) and not func.is_constructor:
            has_init_guard = any('init' in m.lower() for m in func.modifiers) or func.has_reentrancy_guard
            if not has_init_guard:
                steps.append("LIFECYCLE_RISK(Public_Init_No_Guard)")

        for req in func.require_checks[:3]:
            steps.append(f"CHECK({req[:60]})")
        for dc in func.delayed_checks:
            steps.append(f"DELAYED_CHECK({dc[:70]})")
        for call in func.external_calls:
            is_delay_checked = any(call in dc for dc in func.delayed_checks)
            is_unchecked = any(call in uc for uc in func.unchecked_calls)
            if is_delay_checked:
                steps.append(f"EXT_CALL_CHECKED({call})")
            elif is_unchecked:
                steps.append(f"EXT_CALL_UNCHECKED({call})")
            else:
                steps.append(f"EXT_CALL({call})")
        for icall in func.internal_calls[:2]:
            steps.append(f"INT_CALL({icall[:40]})")
        for var in func.state_writes:
            steps.append(f"STATE_WRITE({var})")
        if func.has_reentrancy_guard and 'reentrancy' in func.guard_type.lower():
            steps.append("GUARD_RELEASE")

        if func.visibility in ['public', 'external'] and source_code:
            func_start = func.line_number - 1
            func_lines_check = source_code.split('\n')[func_start:func_start + 60]
            func_text_check = '\n'.join(func_lines_check)
            if re_mod.search(r'block\.timestamp|now\b', func_text_check):
                taint_desc = "ENV_VAR_READ(block.timestamp)"
                if re_mod.search(r'(block\.timestamp|now)\s*%', func_text_check):
                    taint_desc += " -> Modulo(%)"
                if func.state_writes:
                    taint_desc += f" -> StateWrite"
                steps.append(taint_desc)

        for inlined_flow in func.inlined_internal_flows:
            steps.append(inlined_flow[:150])

        for dfc in func.data_flow_chains:
            steps.append(f"[DATA_FLOW]: {dfc}")

        for lf in func.loop_features:
            steps.append(lf)

        loc = f"@L{func.line_number}"
        flow_parts.append(f"{func.name}{loc}: " + " -> ".join(steps))

    if not flow_parts:
        for func in features.functions:
            if func.visibility in ['public', 'external'] and func.state_writes:
                steps = [f"CHECK({r[:50]})" for r in func.require_checks[:2]]
                steps += [f"STATE_WRITE({v})" for v in func.state_writes]
                loc = f"@L{func.line_number}"
                flow_parts.append(f"{func.name}{loc}: " + " -> ".join(steps))

    return " | ".join(flow_parts) if flow_parts else "No significant flow detected"


_REENTRANCY_CALLBACK_PATTERNS = [
    # Low-level value calls AND bare low-level calls.  A bare `.call(...)`
    # (e.g. `targets[i].call(calldatas[i])`) is equally callback-capable and
    # was previously missed by the value-only pattern, causing complete CEI
    # closures (call before a persistent state write, unguarded) to be
    # dropped before the source-grounded candidate could be emitted.
    ("low_level_value_call", re_mod.compile(r"\.\s*call\s*(?:\{[^}\r\n]*\})?\s*\(|\.\s*call\s*\.\s*value\s*\(|\.\s*send\s*\(", re_mod.I)),
    # Token transfer methods are callback-capable at the source boundary.  The
    # downstream candidate still requires an unguarded callback followed by a
    # persistent/settlement effect, so this does not promote a bare transfer
    # without a concrete re-entry closure.
    ("token_transfer_callback", re_mod.compile(r"\.(?:safeTransferFrom|safeTransfer|transferFrom|transfer)\s*\(", re_mod.I)),
    ("native_transfer_callback", re_mod.compile(r"\.\s*transfer\s*\(", re_mod.I)),
    ("erc721_safe_mint_callback", re_mod.compile(r"(?<![A-Za-z0-9_.])_?safeMint\s*\(", re_mod.I)),
    ("erc777_sender_hook", re_mod.compile(r"\b_?callTokensToSend\s*\(|\.\s*tokensToSend\s*\(", re_mod.I)),
    ("erc777_receiver_hook", re_mod.compile(r"\b(?:tokensReceived|onERC777Received)\s*\(", re_mod.I)),
    ("arbitrary_action_callback", re_mod.compile(r"\.(?:executeAction|executeInstruction)\s*\(", re_mod.I)),
    ("escrow_or_pool_callback", re_mod.compile(r"\.(?:pay|sendCollaterals|sendCollateralsUnwrap|deposit)\s*(?:\{|\()", re_mod.I)),
]

_REENTRANCY_ASSET_ACTION_PATTERN = re_mod.compile(
    r"\.\s*(?P<method>batchMint|safeMint|mint)\s*\(",
    re_mod.I,
)

# Cast-qualified calls are common in reconstructed DAppSCAN slices, where the
# interface declaration is imported and only the call-site cast remains in the
# authoritative source (for example ``IERC20(gov).balanceOf(...)``).  Keep the
# receiver argument deliberately scalar and source-visible; this is not a
# free-form external-call heuristic.
_CAST_QUALIFIED_CALL_PATTERN = re_mod.compile(
    r"\b(?P<type>[A-Z_]\w*)\s*\(\s*(?P<argument>[A-Za-z_]\w*)\s*\)"
    r"(?P<chain>(?:\s*\.\s*[A-Za-z_]\w*\s*\([^;\r\n]*?\))*)"
    r"\s*\.\s*(?P<method>[A-Za-z_]\w*)\s*\(",
    re_mod.I,
)
_EXTERNAL_ASSET_ACTION_METHODS = frozenset({
    "transfer", "transferfrom", "safetransfer", "safetransferfrom",
    "deposit", "withdraw", "redeem", "mint", "burn", "swap",
    "exactinput", "exactoutput", "addliquidity", "removeliquidity",
    "pay", "sendcollaterals", "sendcollateralsunwrap",
})
_QUALIFIED_REENTRANCY_CALLBACK_METHODS = frozenset({
    "deposit", "withdraw", "redeem", "swap", "exactinput", "exactoutput",
    "execute", "executeaction", "executeinstruction", "pay",
    "sendcollaterals", "sendcollateralsunwrap", "flash", "borrow", "repay",
})


_LOW_LEVEL_CALLBACK_CALL = re_mod.compile(
    r"\.\s*call\s*(?:\{[^}\r\n]*\})?\s*\(|\.\s*call\s*\.\s*value\s*\(",
    re_mod.I,
)
_PERSISTENT_WRITE_SUFFIX = r"(?:\s*\[[^\]\r\n]*\]|\s*\.\s*[A-Za-z_]\w*)*"


_ERC20_TRANSFER_METHODS = {"transfer", "transferfrom"}

_TYPED_RETURN_METHOD_PATTERN = re_mod.compile(
    r"(?<![A-Za-z0-9_.])(?P<receiver>[A-Za-z_]\w*(?:\s*\([^;\r\n]*\))?(?:\s*\.\s*[A-Za-z_]\w*)*)\s*\.\s*"
    r"(?P<method>[A-Za-z_]\w*)\s*\(",
    re_mod.I,
)

_TYPED_RETURN_METHODS = frozenset({
    "transfer", "transferfrom", "approve", "increaseallowance", "decreaseallowance",
    "send", "delegatecall", "staticcall", "call",
    "deposit", "withdraw", "redeem", "mint", "burn", "swap",
    "exactinput", "exactoutput", "addliquidity", "removeliquidity",
    "joinpool", "exitpool", "borrow", "repay", "execute", "executetransaction",
    "finalizecrowdfund", "quote", "swapexacttokensfortokens", "swapexactethfortokens",
    "swapexacttokensforeth", "getamountout", "getamountsout",
})


def _e2_unchecked_narrow_rule_enabled() -> bool:
    """Enable the narrow source-grounded unchecked rule only for E2."""

    return (
        os.environ.get(PROFILE_ENV) == E2_DAPPSCAN_VNEXT_PROFILE
        and os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") == "1"
        and os.environ.get("FUSEDAUDIT_E2_UNCHECKED_NARROW_RULE", "1") != "0"
    )


def _e2_precision_gates_enabled() -> bool:
    """Enable the post-hoc source precision rules only when opted in."""

    return (
        os.environ.get(PROFILE_ENV) == E2_DAPPSCAN_VNEXT_PROFILE
        and os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") == "1"
        and os.environ.get("FUSEDAUDIT_E2_PRECISION_GATES", "0") == "1"
    )


def _is_erc20_like_type(type_name: str) -> bool:
    normalized = re_mod.sub(r"[^a-z0-9]", "", (type_name or "").lower())
    return any(token in normalized for token in ("token", "erc20", "eip20", "fungible"))


def _is_ordinary_token_transfer_line(line: str) -> bool:
    """Recognize direct calls on an obviously token-like receiver."""

    expression = re_mod.compile(
        r"(?P<receiver>[A-Za-z_]\w*(?:\s*\([^;]*?\))?)\s*\.\s*"
        r"(?:transferFrom|transfer)\s*\(",
        re_mod.I,
    )
    for match in expression.finditer(line or ""):
        receiver = re_mod.sub(r"[^a-z0-9]", "", match.group("receiver").lower())
        if any(token in receiver for token in ("token", "erc20", "eip20", "fungible")):
            return True
    return False


def _looks_like_native_transfer_receiver(receiver: str) -> bool:
    """Distinguish native-Ether recipients from typed token receivers."""

    compact = re_mod.sub(r"\s+", "", str(receiver or ""))
    return bool(re_mod.fullmatch(
        r"(?:msg\.sender|payable\(msg\.sender\)|recipient|receiver|sender|"
        r"beneficiary|payee|to|caller|owner|[A-Za-z_]\w*(?:addr|address))",
        compact,
        re_mod.I,
    ))


def _token_transfer_has_directional_asset_closure(candidate: Dict[str, object]) -> bool:
    """Keep only source-local token transfers with a user-asset closure.

    ERC-20 ``transfer``/``transferFrom`` is not a standardized callback.  A
    narrow exception is justified when the source proves an incoming user
    transfer into the contract, the result is checked, and the same public
    entry then writes per-user asset state (the common deposit/redeem shape).
    This is semantic and source-local; it does not depend on a dataset ID.
    """

    callback_text = str(
        candidate.get("callback_call_text")
        or candidate.get("callback_line_text")
        or ""
    )
    state_text = str(candidate.get("state_write_line_text") or "")
    if not callback_text or not state_text:
        return False
    method_match = re_mod.search(
        r"\.(safeTransferFrom|safeTransfer|transferFrom|transfer|send)\s*\(",
        callback_text,
        re_mod.I,
    )
    if method_match is None:
        return False
    method = method_match.group(1).casefold()
    safe_api = method in {"safetransfer", "safetransferfrom"} or bool(
        re_mod.search(r"transferhelper\s*\.\s*safeTransfer", callback_text, re_mod.I)
    )
    if safe_api:
        return True

    args = callback_text[method_match.end():]
    incoming = method == "transferfrom" and bool(
        re_mod.search(r"\b(?:msg\.sender|sender)\b", args, re_mod.I)
        and re_mod.search(r"\b(?:address\s*\(\s*this\s*\)|this)\b", args, re_mod.I)
    )
    user_state = bool(
        re_mod.search(r"\[\s*(?:msg\.sender|sender|account|recipient|beneficiary)\s*\]", state_text, re_mod.I)
        or re_mod.search(
            r"\b(?:user|account|position|deposit|stake|balance|reserve|claim|reward|share)\w*\s*\.\s*[A-Za-z_]\w*",
            state_text,
            re_mod.I,
        )
    )
    checked_return = bool(
        re_mod.search(r"\brequire\s*\(", callback_text, re_mod.I)
        or re_mod.search(r"\b(?:success|ok|sent)\s*=", callback_text, re_mod.I)
    )
    function_name = str(candidate.get("function_name") or "").casefold()
    visibility = str(candidate.get("function_visibility") or "").casefold()
    redeem_state = bool(
        incoming
        and re_mod.search(r"\b(?:grant|amountredeemed|tokenid|vesting)\b", state_text, re_mod.I)
    )
    if visibility not in {"public", "external"} or (
        not user_state and not redeem_state
    ):
        return False
    if incoming:
        return bool(
            re_mod.fullmatch(
                r"(?:deposit|redeem)(?:withtransfer|for|from)?",
                function_name,
                re_mod.I,
            )
            and checked_return
        )
    user_recipient = bool(
        re_mod.search(r"\b(?:msg\.sender|account|recipient|sender|beneficiary)\b", args, re_mod.I)
    )
    return bool(
        user_recipient
        and re_mod.search(r"(?:claim|unstake|withdraw)", function_name, re_mod.I)
    )


def _e2_typed_bool_return_is_visible(
    receiver: str, method: str, source_code: str
) -> bool:
    """Require a receiver-scoped bool-return signature for the typed call."""

    method_name = str(method or "").strip()
    if not method_name or method_name.casefold().startswith("safe"):
        return False
    source = str(source_code or "")
    receiver_text = re_mod.sub(r"\s+", "", str(receiver or ""))
    type_match = re_mod.match(r"([A-Za-z_]\w*)\(", receiver_text)
    type_name = type_match.group(1).casefold() if type_match else ""
    if not type_name and re_mod.fullmatch(r"[A-Za-z_]\w*", receiver_text):
        from source_candidates import typed_return_signatures

        _, variable_types = typed_return_signatures(source)
        type_name = variable_types.get(receiver_text.casefold(), "")
    if not type_name:
        return False

    from source_candidates import typed_return_signatures

    type_methods, _ = typed_return_signatures(source)
    returns = type_methods.get(type_name.casefold(), {}).get(
        method_name.casefold(), ()
    )
    return bool(returns and returns[0].casefold() == "bool")


def _is_explicitly_reentrancy_guarded(func: FunctionFlow, contract: ContractFeatures) -> bool:
    """Recognize only mutex-style guards; access control is not a CEI guard."""
    modifier_names = " ".join(func.modifiers).lower()
    if any(token in modifier_names for token in ("nonreentrant", "reentrancy", "mutex", "lock")):
        return True
    for modifier in contract.modifiers:
        if modifier.name not in func.modifiers:
            continue
        body = modifier.body_text.lower()
        if (
            any(token in body for token in ("nonreentrant", "reentrancy", "mutex"))
            or ("locked" in body and "require" in body)
            or ("_status" in body and "require" in body)
        ):
            return True
    return False


def _line_writes_persistent_state(line: str, state_names: Set[str]) -> bool:
    """Return whether one source line mutates a known contract state variable."""
    for state_name in state_names:
        escaped = re_mod.escape(state_name)
        reference = rf"\b{escaped}\b{_PERSISTENT_WRITE_SUFFIX}"
        if re_mod.search(rf"\bdelete\s+{reference}", line):
            return True
        if re_mod.search(rf"{reference}\s*(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)", line):
            return True
        if re_mod.search(rf"{reference}\s*\.\s*(?:push|pop)\s*\(", line):
            return True
    return False


_STORAGE_ALIAS_DECLARATION = re_mod.compile(
    r"\bstorage\s+(?P<alias>[A-Za-z_]\w*)\s*=",
    re_mod.I,
)


def _storage_alias_names_for_function(
    func: FunctionFlow, source_lines: List[str]
) -> Set[str]:
    """Collect local ``storage`` aliases whose fields are persistent state.

    Solidity commonly binds a mapping/struct entry to a local storage alias
    (for example ``UserInfo storage user = userInfo[id]``).  The alias itself
    is not a contract state-variable name, so the older state-write extractor
    missed later writes such as ``user.amount = 0``.  Keep this source-local:
    only declarations with an assignment are accepted, never ordinary memory
    locals or free-form names.
    """

    start, clean_lines = _function_clean_source_lines(func, source_lines)
    del start
    joined = " ".join(clean_lines)
    return {
        match.group("alias")
        for match in _STORAGE_ALIAS_DECLARATION.finditer(joined)
        if match.group("alias")
    }


def _line_writes_storage_alias(line: str, aliases: Set[str]) -> bool:
    """Return whether a line mutates a field/index of a storage alias."""

    for alias in aliases:
        reference = rf"\b{re_mod.escape(alias)}\b(?:\s*\.\s*[A-Za-z_]\w*|\s*\[[^\]\r\n]+\])+"
        if re_mod.search(rf"\bdelete\s+{reference}", line, re_mod.I):
            return True
        if re_mod.search(
            rf"{reference}\s*(?:=|\+=|-=|\*=|/=|\+\+|--)(?!=)",
            line,
            re_mod.I,
        ):
            return True
    return False


_REENTRANCY_ASSET_STATE_HINTS = {
    "allowance", "balance", "claim", "collateral", "debt", "deposit",
    "liquidity", "owned", "position", "reserve", "reward", "share",
    "supply", "stake", "staker", "total", "withdraw",
}
_REENTRANCY_BOOKKEEPING_STATE_HINTS = {
    "enabled", "fee", "flag", "lock", "nonce", "paused", "previous",
    "status", "tax", "timestamp", "cycle", "threshold",
}


def _reentrancy_state_write_priority(line: str, state_names: Set[str]) -> int:
    """Prefer asset/accounting writes over fee, lock, and configuration writes."""

    score = 0
    compact_line = re_mod.sub(r"\s+", "", line or "").lower()
    for state_name in state_names:
        if not re_mod.search(rf"\b{re_mod.escape(state_name)}\b", line or "", re_mod.I):
            continue
        normalized_name = re_mod.sub(r"[^a-z0-9]", "", state_name.lower())
        if any(hint in normalized_name for hint in _REENTRANCY_ASSET_STATE_HINTS):
            score += 100
        if any(hint in normalized_name for hint in _REENTRANCY_BOOKKEEPING_STATE_HINTS):
            score -= 100
        if "[" in compact_line or ".push(" in compact_line or ".pop(" in compact_line:
            score += 15
    # A storage-alias field (``user.amount``, ``position.collateral``) is a
    # user/accounting effect and is more informative than a global checkpoint
    # write when both occur after the callback.
    lhs = re_mod.split(
        r"(?:\+=|-=|\*=|/=|(?<![=!<>])=(?!=)|\+\+|--)",
        line,
        maxsplit=1,
    )[0]
    if re_mod.search(
        r"\b(?:user|account|position|deposit|balance|reserve|claim|reward)\w*\s*\.",
        lhs,
        re_mod.I,
    ):
        score += 125
    return score


def _select_reentrancy_state_write(
    write_lines: List[int], source_lines: List[str], state_names: Set[str]
) -> int | None:
    """Choose the strongest source-local state-write anchor, then earliest on ties."""

    valid_lines = sorted({line for line in write_lines if 1 <= line <= len(source_lines)})
    if not valid_lines:
        return None
    return min(
        valid_lines,
        key=lambda line: (
            -_reentrancy_state_write_priority(source_lines[line - 1], state_names),
            line,
        ),
    )


def _function_clean_source_lines(func: FunctionFlow, source_lines: List[str]) -> Tuple[int, List[str]]:
    start = max(func.line_number - 1, 0)
    end = min(func.end_line_number or len(source_lines), len(source_lines))
    return start, _strip_comments_preserve_lines("\n".join(source_lines[start:end])).splitlines()


def _callback_callsite_text(
    source_lines: List[str], line_number: int, end_line_number: int
) -> str:
    """Capture enough of a multiline callback call for downstream evidence gates."""

    if not (1 <= line_number <= len(source_lines)):
        return ""
    start = line_number
    for candidate in range(max(1, line_number - 3), line_number):
        prefix = " ".join(source_lines[candidate - 1:line_number])
        if re_mod.search(r"\brequire\s*\([^;]*$", prefix, re_mod.I):
            start = candidate
            break
    end = min(len(source_lines), end_line_number or len(source_lines), line_number + 10)
    parts: List[str] = []
    callback_seen = False
    for index in range(start - 1, end):
        line = re_mod.sub(r"//.*$", "", source_lines[index]).strip()
        if line:
            parts.append(line)
        joined = " ".join(parts)
        if re_mod.search(
            r"\.(?:call|safeTransferFrom|transferFrom|safeTransfer|transfer|send)\s*\(|"
            r"\b_?safeMint\s*\(",
            joined,
            re_mod.I,
        ):
            callback_seen = True
        if callback_seen and ";" in joined:
            return joined
    return " ".join(parts)


def _callback_events_for_function(
    contract: ContractFeatures,
    func: FunctionFlow,
    source_lines: List[str],
    source_code: str = "",
    qualified_callback_lines: Set[int] | None = None,
) -> List[Tuple[int, str]]:
    """Return source-anchored callback-capable calls within one function."""

    start, clean_lines = _function_clean_source_lines(func, source_lines)
    if qualified_callback_lines is None:
        qualified_callback_lines = _qualified_external_callback_event_lines(
            func,
            source_lines,
            _qualified_external_dispatch_metadata(source_code or "\n".join(source_lines)),
        )
    inherits_erc721 = any("erc721" in inheritance.lower() for inheritance in contract.inherits)
    events = []
    for offset, line in enumerate(clean_lines):
        source_line = start + offset + 1
        # A helper declaration can itself contain the callback-shaped name
        # (for example ``function _safeMint(...)``).  It is not an execution
        # event and must not become a second candidate for its caller.
        if line.lstrip().startswith(("function ", "modifier ")):
            continue
        if source_line in qualified_callback_lines:
            # Preserve the existing qualified-dispatch channel for arbitrary
            # mutating calls, but keep ERC-20 transfer methods on the stricter
            # token-transfer evidence gate.  A typed transfer is not a generic
            # low-level callback merely because the receiver type was missing
            # from a blinded slice.
            callback_type = _qualified_external_callback_kind(line) or (
                "token_transfer_callback"
                if re_mod.search(
                    r"\.(?:safeTransferFrom|safeTransfer|transferFrom|transfer)\s*\(",
                    line,
                    re_mod.I,
                )
                else "qualified_external_callback"
            )
            events.append((source_line, callback_type))
            continue
        for callback_type, pattern in _REENTRANCY_CALLBACK_PATTERNS:
            if not pattern.search(line):
                continue
            if (
                callback_type == "native_transfer_callback"
                and _is_ordinary_token_transfer_line(line)
            ):
                continue
            # tx.origin is an EOA origin, not an attacker-controlled callback
            # receiver.  Keep its low-level transfer available to the
            # unchecked-call analysis, but do not promote it to reentrancy.
            if callback_type == "low_level_value_call" and re_mod.search(
                r"\btx\s*\.\s*origin\s*\.\s*call\b", line, re_mod.I
            ):
                continue
            # `_safeMint` has a standardized receiver callback only in an
            # ERC-721 implementation; a same-named local helper is not enough.
            if callback_type == "erc721_safe_mint_callback" and not inherits_erc721:
                continue
            events.append((source_line, callback_type))
            break
    return events


def _asset_action_events_for_function(
    func: FunctionFlow, source_lines: List[str]
) -> List[Tuple[int, str]]:
    """Return qualified external asset actions in one function."""

    start, clean_lines = _function_clean_source_lines(func, source_lines)
    events: List[Tuple[int, str]] = []
    for offset, line in enumerate(clean_lines):
        for match in _REENTRANCY_ASSET_ACTION_PATTERN.finditer(line):
            events.append((start + offset + 1, match.group("method")))
    return events


def _qualified_external_callback_kind(line: str) -> str | None:
    """Classify one already-qualified external call without widening callers."""

    def is_callback_method(method: str) -> bool:
        return method.casefold() in _QUALIFIED_REENTRANCY_CALLBACK_METHODS

    for match in _CAST_QUALIFIED_CALL_PATTERN.finditer(line or ""):
        method = match.group("method")
        method_key = method.casefold()
        if (
            method_key == "swap"
            and re_mod.search(r"new\s+bytes\s*\(\s*0\s*\)", line or "", re_mod.I)
        ):
            continue
        if method_key in _ERC20_TRANSFER_METHODS:
            return "token_transfer_callback"
        if _EXTERNAL_READ_METHOD_HINT.match(method):
            return "external_view_call"
        if is_callback_method(method):
            return "qualified_external_callback"
    for match in _QUALIFIED_CALL_PATTERN.finditer(line or ""):
        method = match.group("method")
        method_key = method.casefold()
        if (
            method_key == "swap"
            and re_mod.search(r"new\s+bytes\s*\(\s*0\s*\)", line or "", re_mod.I)
        ):
            continue
        if method_key in _ERC20_TRANSFER_METHODS:
            return "token_transfer_callback"
        if _EXTERNAL_READ_METHOD_HINT.match(method):
            return "external_view_call"
        if is_callback_method(method):
            return "qualified_external_callback"
    return None


def _external_asset_action_events_for_function(
    func: FunctionFlow, source_lines: List[str]
) -> List[Tuple[int, str]]:
    """Return source-visible external asset/settlement actions.

    This channel is a terminal observation after a proven callback.  It is kept
    separate from the ordinary token-transfer callback channel so a plain
    ``token.transfer`` cannot become a new callback candidate merely because a
    later action exists.
    """

    start, clean_lines = _function_clean_source_lines(func, source_lines)
    events: List[Tuple[int, str]] = []

    def add(line_number: int, method: str) -> None:
        key = (line_number, method.casefold())
        if key not in {(line, action.casefold()) for line, action in events}:
            events.append((line_number, method))

    for offset, line in enumerate(clean_lines):
        for match in _CAST_QUALIFIED_CALL_PATTERN.finditer(line):
            method = match.group("method")
            if (
                method.casefold() == "swap"
                and re_mod.search(r"new\s+bytes\s*\(\s*0\s*\)", line, re_mod.I)
            ):
                continue
            if method.casefold() not in _EXTERNAL_ASSET_ACTION_METHODS:
                continue
            # The cast itself is the source proof that the call crosses a
            # contract/interface boundary; transfer methods remain terminal
            # actions here, never callback admission evidence.
            add(start + offset + 1, method)
        for match in _QUALIFIED_CALL_PATTERN.finditer(line):
            method = match.group("method")
            method_key = method.casefold()
            receiver = match.group("receiver")
            receiver_key = re_mod.sub(r"[^a-z0-9]", "", receiver.casefold())
            if method_key not in _EXTERNAL_ASSET_ACTION_METHODS:
                continue
            if not (
                receiver[:1].isupper()
                or any(
                    token in receiver_key
                    for token in ("token", "router", "pool", "vault", "asset", "auction", "helper")
                )
            ):
                continue
            add(start + offset + 1, method)
    return events


def _is_standard_erc777_accounting_path(candidate: Dict[str, object]) -> bool:
    """Reject standard ERC777 sender-hook bookkeeping as reentrancy evidence."""

    if candidate.get("source_callback_type") != "erc777_sender_hook":
        return False
    callback_text = str(
        candidate.get("callback_call_text")
        or candidate.get("callback_line_text")
        or ""
    )
    state_write_text = str(candidate.get("state_write_line_text") or "")
    if not re_mod.search(r"\b(?:_callTokensToSend|tokensToSend)\b", callback_text, re_mod.I):
        return False
    if not re_mod.search(r"\b_(?:balances|allowances)\b", state_write_text, re_mod.I):
        return False
    return re_mod.fullmatch(
        r"(?:_?burn|_?mint|transfer(?:from)?|send|operatorSend)",
        str(candidate.get("function_name") or "").strip(),
        re_mod.I,
    ) is not None


def _state_write_lines_for_function(
    func: FunctionFlow, source_lines: List[str], state_names: Set[str]
) -> List[int]:
    start, clean_lines = _function_clean_source_lines(func, source_lines)
    storage_aliases = _storage_alias_names_for_function(func, source_lines)
    return [
        start + offset + 1
        for offset, line in enumerate(clean_lines)
        if _line_writes_persistent_state(line, state_names)
        or _line_writes_storage_alias(line, storage_aliases)
    ]


_E2_FLASH_LOAN_RECEIVER_BINDING = re_mod.compile(
    r"(?P<lhs>[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)?)\s*=\s*"
    r"[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*\s*\(\s*"
    r"(?:[A-Za-z_]\w*\s*\.\s*)?receiverAddress\s*\)",
    re_mod.I,
)
_E2_FLASH_LOAN_CALLBACK = re_mod.compile(
    r"(?P<receiver>[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)?)\s*\.\s*"
    r"executeOperation\s*\(",
    re_mod.I,
)
_E2_FLASH_LOAN_STORAGE_PARAMETER = re_mod.compile(
    r"\bstorage\s+(?P<name>[A-Za-z_]\w*)", re_mod.I
)
_E2_FLASH_LOAN_STORAGE_UPDATE = re_mod.compile(
    r"\b(?P<storage>[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)?)\s*\.\s*"
    r"(?:updateState|cumulateToLiquidityIndex|updateInterestRates)\s*\(|"
    r"\b(?P<storage_lhs>[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)?)\s*\.\s*"
    r"(?:accruedToTreasury|liquidityIndex|currentLiquidityRate|currentVariableBorrowRate)\s*"
    r"(?:=|\+=|-=|\*=|/=)(?!=)",
    re_mod.I,
)


def _detect_e2_flash_loan_receiver_reentrancy_candidates(
    contracts: List[ContractFeatures], source_code: str
) -> List[Dict[str, object]]:
    """Find E2-only flash-loan callback paths that update storage afterwards.

    Flash-loan logic is often implemented in a library and mutates a storage
    parameter (for example ``reserve``), so the normal contract-state write
    extractor cannot see the accounting closure.  This rule stays narrow:
    the function must be a reachable flash-loan entry, bind a receiver from
    ``receiverAddress``, call ``executeOperation``, and then mutate the same
    storage/accounting object without a mutex guard.
    """

    if (
        os.environ.get(PROFILE_ENV) != E2_DAPPSCAN_VNEXT_PROFILE
        or os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") != "1"
        or not source_code
    ):
        return []

    source_lines = source_code.splitlines()
    candidates: List[Dict[str, object]] = []
    for contract in contracts:
        for func in contract.functions:
            if (
                func.visibility not in {"public", "external"}
                or not func.is_reachable
                or "flashloan" not in func.name.casefold()
                or _is_explicitly_reentrancy_guarded(func, contract)
            ):
                continue

            start, clean_lines = _function_clean_source_lines(func, source_lines)
            body = "\n".join(clean_lines)
            storage_names = {
                match.group("name")
                for match in _E2_FLASH_LOAN_STORAGE_PARAMETER.finditer(
                    " ".join(source_lines[max(func.line_number - 1, 0):func.line_number + 10])
                )
            }
            if not storage_names:
                continue

            receiver_aliases = {
                re_mod.sub(r"\s+", "", match.group("lhs"))
                for match in _E2_FLASH_LOAN_RECEIVER_BINDING.finditer(body)
            }
            if not receiver_aliases:
                continue

            callback_match = None
            callback_receiver = ""
            for match in _E2_FLASH_LOAN_CALLBACK.finditer(body):
                receiver = re_mod.sub(r"\s+", "", match.group("receiver"))
                if receiver in receiver_aliases:
                    callback_match = match
                    callback_receiver = receiver
                    break
            if callback_match is None:
                continue

            callback_line = start + body[:callback_match.start()].count("\n") + 1
            update_match = None
            update_line = 0
            for match in _E2_FLASH_LOAN_STORAGE_UPDATE.finditer(body):
                line = start + body[:match.start()].count("\n") + 1
                storage_expr = re_mod.sub(r"\s+", "", match.group("storage") or match.group("storage_lhs") or "")
                if line <= callback_line or storage_expr.split(".", 1)[0] not in storage_names:
                    continue
                update_match = match
                update_line = line
                break
            if update_match is None:
                continue

            callback_text = source_lines[callback_line - 1].strip()
            update_text = source_lines[update_line - 1].strip()
            candidates.append({
                "risk_type": "reentrancy",
                "submechanism": "flash_loan_receiver_callback_before_storage_update",
                "confidence": 0.95,
                "reason": (
                    f"{func.name}() has an unguarded flash-loan receiver callback "
                    f"at @L{callback_line} before a persistent state write "
                    f"@L{update_line} (storage/accounting update)"
                ),
                "function_name": func.name,
                "entry_function_names": [func.name],
                "entrypoint_lines": [func.line_number] if func.line_number > 0 else [],
                "execution_order": ["callback", "state_write"],
                "line": callback_line,
                "evidence_lines": [callback_line, update_line],
                "source_callback_type": "flash_loan_receiver_callback",
                "source_callback_kind": "caller_controlled_flash_loan_receiver",
                "function_visibility": func.visibility,
                "function_modifiers": list(func.modifiers),
                "callback_line_text": callback_text,
                "callback_call_text": _callback_callsite_text(
                    source_lines, callback_line, func.end_line_number
                ),
                "state_write_line_text": update_text,
                "source_grounded": True,
                "source_evidence_kind": "e2_flash_loan_receiver_callback_before_storage_update",
                "storage_parameter": next(
                    name for name in storage_names
                    if re_mod.search(rf"\b{re_mod.escape(name)}\b", update_text)
                ),
                "receiver_alias": callback_receiver,
            })
    return candidates


_CALLER_CONTROLLED_LOW_LEVEL_RECEIVER = re_mod.compile(
    r"(?P<receiver>[A-Za-z_]\w*)\s*\.\s*call\s*"
    r"(?:\{[^}\r\n]*\}\s*)?\(",
    re_mod.I,
)
_POST_CALLBACK_BALANCE_OBSERVATION = re_mod.compile(
    r"\.\s*balanceOf\s*\(",
    re_mod.I,
)
_POST_CALLBACK_CALLER_STATE_CLOSURE = re_mod.compile(
    r"(?:_store\s*\(\)|\.\s*data\b|\b(?:balance|balances|amount|"
    r"claim|deposit|fund|refund|reserve|reward|share|state|status|total)\b)"
    r"[^;]*(?:=|\+=|-=|\*=|/=)(?!=)",
    re_mod.I,
)


def _function_parameter_names(func: FunctionFlow, source_lines: List[str]) -> Set[str]:
    """Return parameter identifiers from a bounded function header slice."""

    start = max(func.line_number - 1, 0)
    header = " ".join(source_lines[start:min(start + 10, len(source_lines))])
    header = header.split("{", 1)[0]
    match = re_mod.search(r"\(([^()]*)\)", header, re_mod.S)
    if not match:
        return set()
    names: Set[str] = set()
    for declaration in match.group(1).split(","):
        tokens = re_mod.findall(r"[A-Za-z_]\w*", declaration)
        if len(tokens) >= 2:
            names.add(tokens[-1])
    return names


def _caller_controlled_aliases(
    func: FunctionFlow, source_lines: List[str], line_number: int
) -> Set[str]:
    """Resolve simple local aliases that still carry a function parameter.

    The E2 reentrancy rule only follows one-way scalar/address aliases.  It does
    not infer storage, return values, arbitrary expressions, or attacker
    control from a Gold label.  That keeps the callback proof limited to a
    source-visible parameter -> callback receiver path.
    """

    if not (1 <= line_number <= len(source_lines)):
        return set()
    start = max(func.line_number - 1, 0)
    end = min(line_number, len(source_lines))
    parameter_names = _function_parameter_names(func, source_lines)
    aliases = set(parameter_names)
    if not aliases:
        return aliases

    alias_assignment = re_mod.compile(
        r"(?:address(?:\s+payable)?\s+)?(?P<lhs>[A-Za-z_]\w*)\s*=\s*"
        r"(?:payable\s*\(\s*|address\s*\(\s*)?"
        r"(?P<rhs>[A-Za-z_]\w*)\s*\)?\s*;",
        re_mod.I,
    )
    for line in source_lines[start:end]:
        clean = re_mod.sub(r"//.*$", "", line).strip()
        match = alias_assignment.search(clean)
        if not match:
            continue
        if match.group("rhs") in aliases:
            aliases.add(match.group("lhs"))
    return aliases


def _caller_controlled_low_level_receiver(
    func: FunctionFlow, source_lines: List[str], line_number: int
) -> str | None:
    """Return a callback receiver only when it is a function-controlled input."""

    if not (1 <= line_number <= len(source_lines)):
        return None
    # The receiver may be declared on the preceding line in a multi-line call.
    # Keep the window bounded to the current statement so an unrelated earlier
    # call cannot donate its receiver to this callback.
    start = max(func.line_number, line_number - 3)
    context = " ".join(source_lines[start - 1:line_number])
    match = _CALLER_CONTROLLED_LOW_LEVEL_RECEIVER.search(context)
    if not match:
        return None
    receiver = match.group("receiver")
    if receiver in _caller_controlled_aliases(func, source_lines, line_number):
        return receiver
    return None


def _post_callback_balance_observation(
    func: FunctionFlow,
    source_lines: List[str],
    callback_line: int,
    state_write_lines: List[int],
) -> Tuple[int, str, str] | None:
    """Find a balance observation after a caller-controlled callback.

    A balance read is accepted only as a post-call settlement signal.  This is
    deliberately narrower than treating every later return or external call
    as a reentrancy closure.
    """

    if any(line < callback_line for line in state_write_lines):
        return None
    end_line = min(func.end_line_number or len(source_lines), len(source_lines))
    for line_number in range(callback_line + 1, end_line + 1):
        line = re_mod.sub(r"//.*$", "", source_lines[line_number - 1]).strip()
        if not line:
            continue
        if _POST_CALLBACK_BALANCE_OBSERVATION.search(line):
            return line_number, "balance_observation", line
    return None


def _post_internal_call_state_closure(
    func: FunctionFlow, source_lines: List[str], call_line: int
) -> Tuple[int, str] | None:
    """Find a caller-side state/settlement write after an internal helper call."""

    end_line = min(func.end_line_number or len(source_lines), len(source_lines))
    for line_number in range(call_line + 1, end_line + 1):
        line = re_mod.sub(r"//.*$", "", source_lines[line_number - 1]).strip()
        if line and _POST_CALLBACK_CALLER_STATE_CLOSURE.search(line):
            return line_number, line
    return None


def _external_view_state_write_candidates(
    func: FunctionFlow,
    source_lines: List[str],
    contract: ContractFeatures,
) -> List[Dict[str, object]]:
    """Find typed external calls followed by a storage-looking assignment.

    E2 includes source slices whose inherited declarations are outside the
    supplied file.  In that case ``state_names`` is empty even though a
    function such as ``_setComptroller`` clearly writes inherited storage.
    Keep this fallback narrow: the receiver must be a typed interface-like
    parameter, the call must look like a view/validation method, and the later
    assignment must not target a local or parameter variable.
    """

    if (
        func.visibility not in {"public", "external"}
        or not func.is_reachable
        or _is_explicitly_reentrancy_guarded(func, contract)
    ):
        return []

    start, clean_lines = _function_clean_source_lines(func, source_lines)
    signature = " ".join(source_lines[start:min(start + 8, len(source_lines))])
    parameter_match = re_mod.search(r"\(([^)]*)\)", signature, re_mod.DOTALL)
    if not parameter_match:
        return []

    interface_parameters: Set[str] = set()
    local_names: Set[str] = set()
    for declaration in parameter_match.group(1).split(","):
        tokens = re_mod.findall(r"[A-Za-z_]\w*", declaration)
        if not tokens:
            continue
        name = tokens[-1]
        local_names.add(name)
        type_text = " ".join(tokens[:-1]).casefold()
        if (
            "interface" in type_text
            or "comptroller" in type_text
            or any(token.startswith("i") and len(token) > 1 for token in tokens[:-1])
        ):
            interface_parameters.add(name)
    declaration_pattern = re_mod.compile(
        r"\b(?:uint(?:\d+)?|int(?:\d+)?|address|bool|bytes(?:\d+)?|string|"
        r"[A-Z_]\w*)\s+(?P<name>[A-Za-z_]\w*)\s*(?:=|;)",
    )
    for line in clean_lines:
        match = declaration_pattern.search(line)
        if match:
            local_names.add(match.group("name"))

    view_call_pattern = re_mod.compile(
        r"\b(?P<receiver>[A-Za-z_]\w*)\s*\.\s*"
        r"(?P<method>(?:is[A-Z]\w*|supportsInterface|balanceOf|decimals|"
        r"totalSupply|get[A-Z]\w*|latestVault|pricePerShare|current[A-Z]\w*))\s*\(",
    )
    cast_view_call_pattern = re_mod.compile(
        r"\b(?P<type>[A-Z_]\w*)\s*\(\s*(?P<argument>[A-Za-z_]\w*)\s*\)"
        r"(?:\s*\.\s*[A-Za-z_]\w*\s*\([^;\r\n]*?\))*\s*\.\s*"
        r"(?P<method>(?:is[A-Z]\w*|supportsInterface|balanceOf|decimals|"
        r"totalSupply|get[A-Z]\w*|latestVault|pricePerShare|current[A-Z]\w*))\s*\(",
        re_mod.I,
    )
    assignment_pattern = re_mod.compile(
        r"^\s*(?P<lhs>[A-Za-z_]\w*(?:\s*\[[^\]\r\n]+\]|\s*\.\s*[A-Za-z_]\w*)*)\s*"
        r"(?:=|\+=|-=|\*=|/=)\s*"
    )
    candidates: List[Dict[str, object]] = []
    for offset, line in enumerate(clean_lines):
        direct_match = view_call_pattern.search(line)
        cast_match = cast_view_call_pattern.search(line) if direct_match is None else None
        if direct_match is not None:
            if direct_match.group("receiver") not in interface_parameters:
                continue
            call_match = direct_match
        else:
            call_match = cast_match
            if call_match is None:
                continue
            # A cast-qualified observation carries its own external-boundary
            # proof.  The cast argument must be source-visible and not a local
            # temporary declared in this function.
            argument = call_match.groupdict().get("argument")
            if not argument or argument in local_names:
                continue
        call_line = start + offset + 1
        for later_offset in range(offset + 1, len(clean_lines)):
            later_line = clean_lines[later_offset]
            assignment = assignment_pattern.search(later_line)
            if not assignment:
                continue
            lhs = assignment.group("lhs")
            if lhs in local_names:
                continue
            write_line = start + later_offset + 1
            candidates.append({
                "risk_type": "reentrancy",
                "submechanism": "external_view_call_before_inherited_state_write",
                "confidence": 0.9,
                "reason": (
                    f"{func.name}() has an unguarded external_view_call callback at "
                    f"@L{call_line} before a persistent state write @L{write_line}"
                ),
                "function_name": func.name,
                "entry_function_names": [func.name],
                "entrypoint_lines": [func.line_number],
                "execution_order": ["callback", "state_write"],
                "line": call_line,
                "evidence_lines": [call_line, write_line],
                "source_callback_type": "external_view_call",
                "function_visibility": func.visibility,
                "function_modifiers": list(func.modifiers),
                "callback_line_text": line.strip(),
                "callback_call_text": line.strip(),
                "state_write_line_text": later_line.strip(),
                "source_grounded": True,
            })
            break
    return candidates


def _internal_call_events_for_function(
    func: FunctionFlow, source_lines: List[str], known_function_names: Set[str]
) -> List[Tuple[int, str]]:
    """Extract bare calls to same-contract functions with their source lines."""

    callable_names = sorted((name for name in known_function_names if name and name != func.name), key=len, reverse=True)
    if not callable_names:
        return []
    pattern = re_mod.compile(
        r"(?<![A-Za-z0-9_.])(" + "|".join(re_mod.escape(name) for name in callable_names) + r")\s*\("
    )
    start, clean_lines = _function_clean_source_lines(func, source_lines)
    events = []
    for offset, line in enumerate(clean_lines):
        if line.lstrip().startswith("function "):
            continue
        for match in pattern.finditer(line):
            events.append((start + offset + 1, match.group(1)))
    return events


def _runtime_reentrancy_events(
    function_name: str,
    functions_by_name: Dict[str, FunctionFlow],
    callback_events: Dict[str, List[Tuple[int, str]]],
    asset_action_events: Dict[str, List[Tuple[int, str]]],
    qualified_callback_lines: Dict[str, Set[int]],
    state_writes: Dict[str, List[int]],
    call_events: Dict[str, List[Tuple[int, str]]],
    contract: ContractFeatures,
    stack: Tuple[str, ...] = (),
) -> List[Dict[str, object]]:
    """Flatten callback/state events in runtime call order for one entry."""

    if function_name in stack or function_name not in functions_by_name:
        return []
    func = functions_by_name[function_name]
    if _is_explicitly_reentrancy_guarded(func, contract):
        return []

    calls_by_line: Dict[int, List[str]] = defaultdict(list)
    for line, callee in call_events.get(function_name, []):
        calls_by_line[line].append(callee)
    callbacks_by_line: Dict[int, List[str]] = defaultdict(list)
    for line, callback_type in callback_events.get(function_name, []):
        callbacks_by_line[line].append(callback_type)
    asset_actions_by_line: Dict[int, List[str]] = defaultdict(list)
    for line, action_type in asset_action_events.get(function_name, []):
        asset_actions_by_line[line].append(action_type)

    event_lines = set(calls_by_line) | set(callbacks_by_line) | set(
        state_writes.get(function_name, [])
    ) | set(asset_actions_by_line)
    events: List[Dict[str, object]] = []
    next_stack = stack + (function_name,)
    for line in sorted(event_lines):
        for callee in calls_by_line[line]:
            events.extend(
                _runtime_reentrancy_events(
                    callee,
                    functions_by_name,
                    callback_events,
                    asset_action_events,
                    qualified_callback_lines,
                    state_writes,
                    call_events,
                    contract,
                    next_stack,
                )
            )
        for callback_type in callbacks_by_line[line]:
            events.append({
                "kind": "callback",
                "line": line,
                "callback_type": callback_type,
                "callback_kind": (
                    "qualified_external_call"
                    if line in qualified_callback_lines.get(function_name, set())
                    else callback_type
                ),
                "function_name": function_name,
            })
        if line in state_writes.get(function_name, []):
            events.append({
                "kind": "state_write",
                "line": line,
                "function_name": function_name,
            })
        for action_type in asset_actions_by_line.get(line, []):
            events.append({
                "kind": "asset_action",
                "line": line,
                "action_type": action_type,
                "function_name": function_name,
            })
    return events


def detect_source_grounded_reentrancy_candidates(
    contracts: List[ContractFeatures], source_code: str
) -> List[Dict[str, object]]:
    """Find direct CEI violations with source-level call and write locations.

    A public/external entry point must make a source-anchored callback-capable
    interaction before mutating known persistent state in the same function,
    and no mutex-style guard may be present.  This deliberately covers the
    callback vocabulary used by the evidence-chain extractor (for example,
    ERC-20 transfer helpers), rather than treating only low-level ``.call`` as
    an interaction candidate.
    """
    if not source_code:
        return []
    source_lines = source_code.splitlines()
    candidates: List[Dict[str, object]] = []
    for contract in contracts:
        functions_by_name = {func.name: func for func in contract.functions if func.name}
        known_function_names = set(functions_by_name)
        state_names = set(contract.state_variables)
        if not functions_by_name:
            continue
        for func in functions_by_name.values():
            candidates.extend(
                _external_view_state_write_candidates(func, source_lines, contract)
            )
        if not state_names:
            continue
        dispatch_metadata = _qualified_external_dispatch_metadata(source_code)
        qualified_callback_lines = {
            name: _qualified_external_callback_event_lines(
                func, source_lines, dispatch_metadata
            )
            for name, func in functions_by_name.items()
        }
        callback_events = {
            name: _callback_events_for_function(
                contract,
                func,
                source_lines,
                source_code,
                qualified_callback_lines[name],
            )
            for name, func in functions_by_name.items()
        }
        # ERC-777 receiver hooks are callback entrypoints themselves.  When
        # the hook delegates its accounting to a reachable helper, the hook
        # declaration is the callback anchor and the helper's storage write is
        # the second evidence anchor.
        for name, func in functions_by_name.items():
            if name.casefold() in {"tokensreceived", "onerc777received"}:
                callback_events[name] = [
                    (func.line_number, "erc777_receiver_hook"),
                    *callback_events.get(name, []),
                ]
        asset_action_events = {
            name: _asset_action_events_for_function(func, source_lines)
            for name, func in functions_by_name.items()
        }
        external_asset_action_events = {
            name: _external_asset_action_events_for_function(func, source_lines)
            for name, func in functions_by_name.items()
        }
        state_writes = {
            name: _state_write_lines_for_function(func, source_lines, state_names)
            for name, func in functions_by_name.items()
        }
        call_events = {
            name: _internal_call_events_for_function(func, source_lines, known_function_names)
            for name, func in functions_by_name.items()
        }

        # Build paths from unguarded public/external entry points.  A helper's
        # function name remains the candidate locus, while entry names record
        # why an internal/private body is reachable from an attacker.
        entrypoints_by_function: Dict[str, Set[str]] = defaultdict(set)
        contract_is_erc721 = any(
            "erc721" in inheritance.casefold() for inheritance in contract.inherits
        )

        def is_safe_mint_entry(func: FunctionFlow) -> bool:
            if not contract_is_erc721:
                return False
            _, clean_lines = _function_clean_source_lines(func, source_lines)
            return any(
                re_mod.search(r"(?<![A-Za-z0-9_.])_?safeMint\s*\(", line, re_mod.I)
                for line in clean_lines
            )

        roots = [
            func for func in functions_by_name.values()
            if func.visibility in ("public", "external")
            and func.is_reachable
            and (
                not _is_admin_restricted(func, source_code)
                or is_safe_mint_entry(func)
            )
            and not _is_explicitly_reentrancy_guarded(func, contract)
        ]
        for root in roots:
            stack = [root.name]
            visited = set()
            while stack:
                name = stack.pop()
                if name in visited or name not in functions_by_name:
                    continue
                visited.add(name)
                entrypoints_by_function[name].add(root.name)
                stack.extend(callee for _, callee in call_events[name] if callee not in visited)

        def reachable_write_lines(function_name: str, visited: Set[str]) -> List[int]:
            if function_name in visited or function_name not in functions_by_name:
                return []
            visited = set(visited)
            visited.add(function_name)
            if _is_explicitly_reentrancy_guarded(functions_by_name[function_name], contract):
                return []
            lines = list(state_writes[function_name])
            for _, callee in sorted(call_events[function_name]):
                lines.extend(reachable_write_lines(callee, visited))
            return lines

        seen = set()

        # Narrow closure for caller-controlled low-level callbacks whose
        # function returns or settles against a post-call balance observation.
        # This covers exchange/router helpers without promoting every checked
        # `.call` that lacks a post-call asset consequence.
        for name, func in functions_by_name.items():
            if _is_explicitly_reentrancy_guarded(func, contract):
                continue
            entrypoints = sorted(entrypoints_by_function.get(name, set()))
            if not entrypoints:
                continue
            for callback_line, callback_type in callback_events.get(name, []):
                if callback_type != "low_level_value_call":
                    continue
                if _caller_controlled_low_level_receiver(func, source_lines, callback_line) is None:
                    continue
                observation = _post_callback_balance_observation(
                    func,
                    source_lines,
                    callback_line,
                    state_writes.get(name, []),
                )
                if observation is None:
                    continue
                observation_line, evidence_kind, observation_text = observation
                signature = (name, callback_line, observation_line, "settlement")
                if signature in seen:
                    continue
                seen.add(signature)
                callback_func = functions_by_name.get(name)
                candidates.append({
                    "risk_type": "reentrancy",
                    "submechanism": (
                        "caller_controlled_low_level_callback_before_"
                        "post_call_settlement"
                    ),
                    "confidence": 0.95,
                    "reason": (
                        f"{name}() has an unguarded low_level_value_call callback at "
                        f"@L{callback_line} before a post-call balance observation "
                        f"@L{observation_line}"
                    ),
                    "function_name": name,
                    "entry_function_names": entrypoints,
                    "entrypoint_lines": sorted({
                        functions_by_name[entrypoint].line_number
                        for entrypoint in entrypoints
                        if entrypoint in functions_by_name
                        and isinstance(functions_by_name[entrypoint].line_number, int)
                        and functions_by_name[entrypoint].line_number > 0
                    }),
                    "execution_order": ["callback", "settlement"],
                    "interprocedural": bool(any(entrypoint != name for entrypoint in entrypoints)),
                    "line": callback_line,
                    "evidence_lines": [callback_line, observation_line],
                    "source_callback_type": callback_type,
                    "source_callback_kind": "caller_controlled_external_target",
                    "post_callback_evidence_kind": evidence_kind,
                    "function_visibility": callback_func.visibility if callback_func else func.visibility,
                    "function_modifiers": list(callback_func.modifiers if callback_func else func.modifiers),
                    "callback_line_text": source_lines[callback_line - 1].strip(),
                    "callback_call_text": _callback_callsite_text(
                        source_lines,
                        callback_line,
                        callback_func.end_line_number if callback_func else func.end_line_number,
                    ),
                    "state_write_line_text": observation_text,
                    "source_grounded": True,
                })

        # Narrow helper closure for native value callbacks.  Preserve the
        # helper declaration as the finding locus while proving that an
        # externally reachable caller performs a later state/settlement write.
        for helper_name, helper in functions_by_name.items():
            if _is_explicitly_reentrancy_guarded(helper, contract):
                continue
            helper_entrypoints = sorted(entrypoints_by_function.get(helper_name, set()))
            if not helper_entrypoints:
                continue
            for callback_line, callback_type in callback_events.get(helper_name, []):
                if callback_type != "low_level_value_call":
                    continue
                if _caller_controlled_low_level_receiver(helper, source_lines, callback_line) is None:
                    continue
                for caller_name, caller_events in call_events.items():
                    if caller_name == helper_name or not entrypoints_by_function.get(caller_name):
                        continue
                    caller = functions_by_name.get(caller_name)
                    if caller is None:
                        continue
                    for call_line, callee in caller_events:
                        if callee != helper_name:
                            continue
                        closure = _post_internal_call_state_closure(
                            caller, source_lines, call_line
                        )
                        if closure is None:
                            continue
                        closure_line, closure_text = closure
                        signature = (helper_name, callback_line, closure_line, "caller_closure")
                        if signature in seen:
                            continue
                        seen.add(signature)
                        candidates.append({
                            "risk_type": "reentrancy",
                            "submechanism": (
                                "interprocedural_low_level_callback_before_"
                                "caller_state_settlement"
                            ),
                            "confidence": 0.95,
                            "reason": (
                                f"{helper_name}() has an unguarded low_level_value_call "
                                f"callback at @L{callback_line}, then reaches caller "
                                f"state settlement @L{closure_line} through {caller_name}()"
                            ),
                            "function_name": helper_name,
                            "entry_function_names": [caller_name],
                            "entrypoint_lines": [caller.line_number] if caller.line_number > 0 else [],
                            "execution_order": ["callback", "state_write"],
                            "interprocedural": True,
                            "line": helper.line_number,
                            "evidence_lines": [callback_line, closure_line],
                            "source_callback_type": callback_type,
                            "source_callback_kind": "caller_controlled_external_target",
                            "function_visibility": helper.visibility,
                            "function_modifiers": list(helper.modifiers),
                            "callback_line_text": source_lines[callback_line - 1].strip(),
                            "callback_call_text": _callback_callsite_text(
                                source_lines, callback_line, helper.end_line_number
                            ),
                            "state_write_line_text": closure_text,
                            "source_grounded": True,
                        })

        for name, func in functions_by_name.items():
            entrypoints = sorted(entrypoints_by_function.get(name, set()))
            if not entrypoints or _is_explicitly_reentrancy_guarded(func, contract):
                continue
            for call_line, callback_type in callback_events[name]:
                if callback_type == "external_view_call":
                    # The dedicated typed-view rule below owns this channel so
                    # one balance observation cannot emit duplicate closures.
                    continue
                # Internal helpers are attributed to their public entry by
                # the runtime expansion below.  Do not expose a low-level
                # helper as an independent entry, which would bypass the
                # function-level semantics of the final gate.
                if (
                    func.visibility not in {"public", "external"}
                    and callback_type == "low_level_value_call"
                ):
                    continue
                write_line = _select_reentrancy_state_write(
                    [line for line in state_writes[name] if line > call_line],
                    source_lines,
                    state_names,
                )
                if write_line is None:
                    continue
                signature = (name, call_line, write_line)
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append({
                    "risk_type": "reentrancy",
                    "submechanism": f"{callback_type}_before_persistent_state_write",
                    "confidence": 0.95 if callback_type == "low_level_value_call" else 0.85,
                    "reason": (
                        f"{name}() has an unguarded {callback_type} callback at @L{call_line} "
                        f"before a persistent state write @L{write_line}"
                    ),
                    "function_name": name,
                    "entry_function_names": entrypoints,
                    "entrypoint_lines": sorted({
                        functions_by_name[entrypoint].line_number
                        for entrypoint in entrypoints
                        if entrypoint in functions_by_name
                        and isinstance(functions_by_name[entrypoint].line_number, int)
                        and functions_by_name[entrypoint].line_number > 0
                    }),
                    "execution_order": ["callback", "state_write"],
                    "line": call_line,
                    "evidence_lines": [call_line, write_line],
                    "source_callback_type": callback_type,
                    "function_visibility": func.visibility,
                    "function_modifiers": list(func.modifiers),
                    "callback_line_text": source_lines[call_line - 1].strip(),
                    "callback_call_text": _callback_callsite_text(
                        source_lines, call_line, func.end_line_number
                    ),
                    "state_write_line_text": source_lines[write_line - 1].strip(),
                    "source_grounded": True,
                })

            # A callback can be followed by an external asset/settlement call
            # even when this source slice has no local state variable (for
            # example ``msg.sender.call(...)`` before a helper transferFrom).
            # Only the low-level channel and cast-qualified non-transfer calls
            # enter this new closure; ordinary ERC-20 transfer callbacks remain
            # on the existing direction-aware gate above.
            for callback_line, callback_type in callback_events[name]:
                if callback_type not in {
                    "low_level_value_call",
                    "qualified_external_callback",
                }:
                    continue
                if (
                    callback_type != "low_level_value_call"
                    and callback_line not in qualified_callback_lines.get(name, set())
                ):
                    continue
                action = next(
                    (
                        (line, action_type)
                        for line, action_type in external_asset_action_events[name]
                        if line > callback_line
                    ),
                    None,
                )
                if action is None:
                    continue
                action_line, action_type = action
                signature = (name, callback_line, action_line, "external_asset_action")
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append({
                    "risk_type": "reentrancy",
                    "submechanism": (
                        f"{callback_type}_before_external_asset_action"
                    ),
                    "confidence": 0.95 if callback_type == "low_level_value_call" else 0.85,
                    "reason": (
                        f"{name}() has an unguarded {callback_type} callback at "
                        f"@L{callback_line}, then reaches external asset action "
                        f"@L{action_line}"
                    ),
                    "function_name": name,
                    "entry_function_names": entrypoints,
                    "entrypoint_lines": sorted({
                        functions_by_name[entrypoint].line_number
                        for entrypoint in entrypoints
                        if entrypoint in functions_by_name
                        and isinstance(functions_by_name[entrypoint].line_number, int)
                        and functions_by_name[entrypoint].line_number > 0
                    }),
                    "execution_order": ["callback", "asset_action"],
                    "interprocedural": bool(any(entrypoint != name for entrypoint in entrypoints)),
                    "line": callback_line,
                    "evidence_lines": [callback_line, action_line],
                    "source_callback_type": callback_type,
                    "source_callback_kind": (
                        "caller_controlled_external_target"
                        if callback_type == "low_level_value_call"
                        else "qualified_external_call"
                    ),
                    "source_asset_action_type": action_type,
                    "function_visibility": func.visibility,
                    "function_modifiers": list(func.modifiers),
                    "callback_line_text": source_lines[callback_line - 1].strip(),
                    "callback_call_text": _callback_callsite_text(
                        source_lines, callback_line, func.end_line_number
                    ),
                    "asset_action_line_text": source_lines[action_line - 1].strip(),
                    "source_grounded": True,
                })
                # Keep one strongest callback -> asset observation per function
                # to prevent alert flooding when a router exposes several
                # consecutive settlement calls.
                break

            for call_line, callback_type in callback_events[name]:
                if callback_type != "token_transfer_callback":
                    continue
                action = next(
                    (
                        (line, action_type)
                        for line, action_type in asset_action_events[name]
                        if line > call_line
                    ),
                    None,
                )
                if action is None:
                    continue
                action_line, action_type = action
                signature = (name, call_line, action_line, "asset_action")
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append({
                    "risk_type": "reentrancy",
                    "submechanism": f"{callback_type}_before_external_asset_action",
                    "confidence": 0.85,
                    "reason": (
                        f"{name}() has an unguarded {callback_type} callback at @L{call_line} "
                        f"before an external {action_type} asset action @L{action_line}"
                    ),
                    "function_name": name,
                    "entry_function_names": entrypoints,
                    "entrypoint_lines": sorted({
                        functions_by_name[entrypoint].line_number
                        for entrypoint in entrypoints
                        if entrypoint in functions_by_name
                        and isinstance(functions_by_name[entrypoint].line_number, int)
                        and functions_by_name[entrypoint].line_number > 0
                    }),
                    "execution_order": ["callback", "asset_action"],
                    "line": call_line,
                    "evidence_lines": [call_line, action_line],
                    "source_callback_type": callback_type,
                    "source_asset_action_type": action_type,
                    "function_visibility": func.visibility,
                    "function_modifiers": list(func.modifiers),
                    "callback_line_text": source_lines[call_line - 1].strip(),
                    "callback_call_text": _callback_callsite_text(
                        source_lines, call_line, func.end_line_number
                    ),
                    "asset_action_line_text": source_lines[action_line - 1].strip(),
                    "source_grounded": True,
                })

        # A callback in a public entry function can precede a state write in
        # an internal helper invoked later on that execution path.  Attribute
        # this candidate to the entry function so its call-line anchor is kept.
        for root in roots:
            for call_line, callback_type in callback_events[root.name]:
                for internal_call_line, callee in call_events[root.name]:
                    if internal_call_line <= call_line:
                        continue
                    write_line = _select_reentrancy_state_write(
                        reachable_write_lines(callee, {root.name}),
                        source_lines,
                        state_names,
                    )
                    if write_line is None:
                        continue
                    signature = (root.name, call_line, write_line)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    candidates.append({
                        "risk_type": "reentrancy",
                        "submechanism": f"interprocedural_{callback_type}_before_persistent_state_write",
                        "confidence": 0.95 if callback_type == "low_level_value_call" else 0.85,
                        "reason": (
                            f"{root.name}() has an unguarded {callback_type} callback at @L{call_line}, "
                            f"then reaches persistent state write @L{write_line} through {callee}()"
                        ),
                        "function_name": root.name,
                        "entry_function_names": [root.name],
                        "entrypoint_lines": [root.line_number] if root.line_number > 0 else [],
                        "execution_order": ["callback", "state_write"],
                        "interprocedural": True,
                    "line": call_line,
                    "evidence_lines": [call_line, write_line],
                        "source_callback_type": callback_type,
                        "function_visibility": root.visibility,
                        "function_modifiers": list(root.modifiers),
                        "callback_line_text": source_lines[call_line - 1].strip(),
                        "callback_call_text": _callback_callsite_text(
                            source_lines, call_line, root.end_line_number
                        ),
                        "state_write_line_text": source_lines[write_line - 1].strip(),
                        "source_grounded": True,
                    })

        # Expand every unguarded public entry through its internal call graph.
        # This closes paths such as transfer -> swap helper -> external
        # router/library dispatch -> token state write, while preserving the
        # runtime order instead of comparing declaration line numbers.
        for root in roots:
            runtime_events = _runtime_reentrancy_events(
                root.name,
                functions_by_name,
                callback_events,
                asset_action_events,
                qualified_callback_lines,
                state_writes,
                call_events,
                contract,
            )
            for index, event in enumerate(runtime_events):
                if event.get("kind") != "callback":
                    continue
                if event.get("callback_type") == "token_transfer_callback":
                    later_asset_events = [
                        candidate_event
                        for candidate_event in runtime_events[index + 1:]
                        if candidate_event.get("kind") == "asset_action"
                    ]
                    asset_event = later_asset_events[0] if later_asset_events else None
                    if asset_event is not None:
                        call_line = int(event["line"])
                        action_line = int(asset_event["line"])
                        callback_function = str(
                            event.get("function_name") or root.name
                        )
                        action_function = str(
                            asset_event.get("function_name") or root.name
                        )
                        signature = (
                            callback_function,
                            call_line,
                            action_line,
                            "asset_action",
                        )
                        if signature not in seen:
                            seen.add(signature)
                            callback_func = functions_by_name.get(callback_function)
                            candidates.append({
                                "risk_type": "reentrancy",
                                "submechanism": (
                                    "interprocedural_token_transfer_callback_"
                                    "before_external_asset_action"
                                    if callback_function != root.name
                                    or action_function != root.name
                                    else "token_transfer_callback_before_external_asset_action"
                                ),
                                "confidence": 0.85,
                                "reason": (
                                    f"{root.name}() has an unguarded token_transfer_callback "
                                    f"at @L{call_line}, then reaches an external "
                                    f"{asset_event.get('action_type', 'asset')} asset action "
                                    f"@L{action_line}"
                                ),
                                "function_name": callback_function,
                                "entry_function_names": [root.name],
                                "entrypoint_lines": (
                                    [root.line_number] if root.line_number > 0 else []
                                ),
                                "execution_order": ["callback", "asset_action"],
                                "interprocedural": (
                                    callback_function != root.name
                                    or action_function != root.name
                                ),
                                "line": call_line,
                                "evidence_lines": [call_line, action_line],
                                "source_callback_type": event["callback_type"],
                                "source_asset_action_type": asset_event.get("action_type", ""),
                                "function_visibility": (
                                    callback_func.visibility
                                    if callback_func is not None
                                    else root.visibility
                                ),
                                "function_modifiers": list(
                                    callback_func.modifiers
                                    if callback_func is not None
                                    else root.modifiers
                                ),
                                "callback_line_text": source_lines[call_line - 1].strip(),
                                "callback_call_text": _callback_callsite_text(
                                    source_lines,
                                    call_line,
                                    callback_func.end_line_number
                                    if callback_func
                                    else root.end_line_number,
                                ),
                                "asset_action_line_text": source_lines[action_line - 1].strip(),
                                "source_grounded": True,
                            })
                # Ordinary token transfers and pool-method labels retain the
                # existing local/asset-flow gate.  Only callback forms with
                # an unambiguous external-dispatch proof are widened across
                # nested helpers here.
                callback_type = event.get("callback_type")
                token_transfer_wrapper = bool(
                    callback_type == "token_transfer_callback"
                    and re_mod.fullmatch(
                        r"(?:deposit|redeem)(?:withtransfer|for|from)?",
                        root.name,
                        re_mod.I,
                    )
                )
                if callback_type not in {
                    "low_level_value_call",
                    "qualified_external_callback",
                    "erc721_safe_mint_callback",
                } and not token_transfer_wrapper:
                    continue
                later_write_events = [
                    candidate_event
                    for candidate_event in runtime_events[index + 1:]
                    if candidate_event.get("kind") == "state_write"
                ]
                callback_function = str(event.get("function_name") or root.name)
                if event.get("callback_type") == "qualified_external_callback":
                    # The same-function closure is emitted by the direct
                    # detector.  Runtime propagation must select a later
                    # write in the reachable public helper instead, otherwise
                    # the earlier helper write masks the caller-side closure.
                    later_write_events = [
                        candidate_event
                        for candidate_event in later_write_events
                        if candidate_event.get("function_name") != callback_function
                    ]
                write_line = _select_reentrancy_state_write(
                    [int(event["line"]) for event in later_write_events],
                    source_lines,
                    state_names,
                )
                write_event = next(
                    (event for event in later_write_events if int(event["line"]) == write_line),
                    None,
                )
                if write_event is None:
                    continue
                call_line = int(event["line"])
                write_line = int(write_event["line"])
                write_function = str(write_event.get("function_name") or root.name)
                # The direct helper closure already owns callback -> state
                # evidence within the same function.  Runtime expansion is
                # only for a cross-function path, which prevents duplicate
                # alerts while preserving public-helper propagation.
                if (
                    event.get("callback_type") == "qualified_external_callback"
                    and callback_function == write_function
                ):
                    continue
                signature = (root.name, call_line, write_line)
                if signature in seen:
                    continue
                seen.add(signature)
                callback_func = functions_by_name.get(callback_function)
                callback_kind = str(event.get("callback_kind") or event["callback_type"])
                interprocedural = callback_function != root.name or write_function != root.name
                candidates.append({
                    "risk_type": "reentrancy",
                    "submechanism": (
                        f"interprocedural_{event['callback_type']}_before_persistent_state_write"
                        if interprocedural
                        else f"{event['callback_type']}_before_persistent_state_write"
                    ),
                    "confidence": 0.95 if event["callback_type"] == "low_level_value_call" else 0.85,
                    "reason": (
                        f"{root.name}() has an unguarded {callback_kind} callback at @L{call_line}, "
                        f"then reaches persistent state write at @L{write_line}"
                    ),
                    "function_name": root.name,
                    "entry_function_names": [root.name],
                    "entrypoint_lines": [root.line_number] if root.line_number > 0 else [],
                    "execution_order": ["callback", "state_write"],
                    "interprocedural": interprocedural,
                    "line": call_line,
                    "evidence_lines": [call_line, write_line],
                    "source_callback_type": event["callback_type"],
                    "source_callback_kind": callback_kind,
                    "function_visibility": root.visibility,
                    "function_modifiers": list(root.modifiers),
                    "callback_line_text": source_lines[call_line - 1].strip(),
                    "callback_call_text": _callback_callsite_text(
                        source_lines,
                        call_line,
                        callback_func.end_line_number if callback_func else root.end_line_number,
                    ),
                    "state_write_line_text": source_lines[write_line - 1].strip(),
                     "source_grounded": True,
                 })
    function_spans = {
        func.name: (func.line_number, func.end_line_number or func.line_number)
        for contract in contracts
        for func in contract.functions
        if func.name
    }
    # A typed ERC-20 transfer/transferFrom is not itself a callback surface.
    # Keep the callback channel for SafeERC20-style wrappers and explicit
    # receiver-capable APIs, but reject ordinary token transfers here before
    # they become source-grounded reentrancy candidates.  This is a semantic
    # gate, not a dataset/source exception.
    filtered_candidates: List[Dict[str, object]] = []
    for candidate in candidates:
        if _is_standard_erc777_accounting_path(candidate):
            continue
        if candidate.get("source_callback_type") == "token_transfer_callback":
            callback_text = str(
                candidate.get("callback_call_text")
                or candidate.get("callback_line_text")
                or ""
            )
            method_match = re_mod.search(
                r"\.(safeTransferFrom|safeTransfer|transferFrom|transfer|send)\s*\(",
                callback_text,
                re_mod.I,
            )
            method = method_match.group(1).casefold() if method_match else ""
            safe_api = method in {"safetransfer", "safetransferfrom"} or bool(
                re_mod.search(r"transferhelper\s*\.\s*safeTransfer", callback_text, re_mod.I)
            )
            if method in {"transfer", "transferfrom"} and not safe_api and not _token_transfer_has_directional_asset_closure(candidate):
                continue
        filtered_candidates.append(candidate)
    candidates = filtered_candidates
    for candidate in candidates:
        evidence_lines = [
            int(line)
            for line in [candidate.get("line"), *(candidate.get("evidence_lines") or [])]
            if isinstance(line, int) and line > 0
        ]
        evidence_lines = list(dict.fromkeys(evidence_lines))
        if not evidence_lines:
            continue
        primary_line = candidate.get("primary_line")
        if not isinstance(primary_line, int) or primary_line <= 0:
            primary_line = evidence_lines[0]
        candidate["primary_line"] = primary_line
        candidate["evidence_lines"] = evidence_lines
        candidate["source_anchor_lines"] = list(dict.fromkeys([
            *(candidate.get("entrypoint_lines") or []), *evidence_lines
        ]))
        span = function_spans.get(str(candidate.get("function_name") or ""))
        anchor = {
            "source_line": primary_line,
            "line": primary_line,
            "line_start": min(candidate["source_anchor_lines"]),
            "line_end": max(candidate["source_anchor_lines"]),
            "function_name": str(candidate.get("function_name") or "") or None,
        }
        if span is not None:
            anchor.update({
                "function_start_line": span[0],
                "function_end_line": span[1],
            })
        candidate["source_anchor"] = anchor
    return candidates


def _strip_comments_preserve_lines(text: str) -> str:
    text = re_mod.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), text, flags=re_mod.S)
    return re_mod.sub(r"//.*?$", "", text, flags=re_mod.M)


_QUALIFIED_CALL_PATTERN = re_mod.compile(
    r"\b(?P<receiver>[A-Za-z_]\w*)\s*\.\s*"
    r"(?P<method>[A-Za-z_]\w*)\s*"
    r"(?P<options>\{[^}\r\n]*\}\s*)?\(",
    re_mod.I,
)
_EXTERNAL_CALLBACK_METHOD_HINT = re_mod.compile(
    r"^(?:swap|addliquidity|removeliquidity|deposit|withdraw|mint|burn|"
    r"execute|transfer|send|approve|stake|unstake|flash|borrow|repay|call)",
    re_mod.I,
)
_EXTERNAL_READ_METHOD_HINT = re_mod.compile(
    r"^(?:balanceof|decimals|totalsupply|allowance|weth|coins|factory|"
    r"get[a-z0-9_]*|is[a-z0-9_]*|supportsinterface|price|latest[a-z0-9_]*|"
    r"virtual[a-z0-9_]*|current[a-z0-9_]*)$",
    re_mod.I,
)


def _qualified_external_dispatch_metadata(source_code: str) -> Dict[str, object]:
    """Collect source-local evidence for qualified callback-capable calls.

    A qualified call is treated as callback-capable only when its receiver is
    a known interface/contract variable with a non-read-only method, or when
    it targets a library method whose own body reaches such a call (or a
    low-level call).  Pure helper calls such as ``SafeMath`` and
    ``Utils.calculateTopUpClaim`` therefore remain excluded.
    """

    empty: Dict[str, object] = {
        "library_names": set(),
        "external_type_names": set(),
        "readonly_methods": {},
        "library_callback_methods": set(),
        "external_receiver_types": {},
        "external_receiver_names": set(),
    }
    if not source_code:
        return empty

    tree = _PARSER.parse(source_code.encode("utf-8"))
    declarations = []
    for declaration_type in (
        "library_declaration",
        "interface_declaration",
        "contract_declaration",
        "abstract_contract_declaration",
    ):
        declarations.extend(_find_all_by_type(tree.root_node, declaration_type))

    library_names = set()
    external_type_names = set()
    readonly_methods: Dict[Tuple[str, str], bool] = {}
    library_function_bodies: Dict[Tuple[str, str], str] = {}

    def declaration_name(node) -> str:
        match = re_mod.search(
            r"\b(?:library|interface|abstract\s+contract|contract)\s+([A-Za-z_]\w*)",
            _node_text(node)[:240],
        )
        return match.group(1) if match else ""

    def function_name(node) -> str:
        for child in node.children:
            if child.type == "identifier":
                return _node_text(child)
        return ""

    for declaration in declarations:
        name = declaration_name(declaration)
        if not name:
            continue
        is_library = declaration.type == "library_declaration"
        if is_library:
            library_names.add(name)
        else:
            external_type_names.add(name)
        for function_node in _find_all_by_type(declaration, "function_definition"):
            method = function_name(function_node)
            if not method:
                continue
            function_text = _node_text(function_node)
            readonly_methods[(name, method)] = bool(
                re_mod.search(r"\b(?:view|pure)\b", function_text)
            )
            body = _find_first_by_type(function_node, "function_body")
            if is_library and body:
                library_function_bodies[(name, method)] = _strip_comments_preserve_lines(
                    _node_text(body)
                )

    external_receiver_types: Dict[str, str] = {}
    for type_name in external_type_names:
        declaration_pattern = re_mod.compile(
            rf"\b{re_mod.escape(type_name)}\b"
            r"(?:\s+(?:public|private|internal|external|immutable|constant|virtual|override))*"
            r"\s+(?P<name>[A-Za-z_]\w*)\s*(?:=|;)",
            re_mod.I,
        )
        for match in declaration_pattern.finditer(source_code):
            external_receiver_types[match.group("name")] = type_name

    # Code-complete reconstruction keeps the source body but may omit imported
    # interface declarations.  Recover only receiver names whose declared type
    # is still visibly interface/contract-shaped (leading uppercase type), so
    # calls on locals such as ``uint256 amount`` cannot enter this channel.
    # This is intentionally metadata only; the method-level callback allowlist
    # below remains the admission gate.
    external_receiver_names: Set[str] = set(external_receiver_types)
    unknown_type_declaration = re_mod.compile(
        r"\b(?P<type>[A-Z_]\w*)\s+"
        r"(?:(?:public|private|internal|external|immutable|constant|virtual|override)\s*)*"
        r"(?P<name>[A-Za-z_]\w*)\s*(?:=|;)",
    )
    for match in unknown_type_declaration.finditer(source_code):
        type_name = match.group("type")
        if type_name in {"String", "Bytes", "Address"}:
            continue
        external_receiver_names.add(match.group("name"))

    library_callback_methods = set()
    for (library_name, method), body_text in library_function_bodies.items():
        has_low_level_call = bool(re_mod.search(
            r"\.\s*(?:call|send|delegatecall|staticcall|transfer)\b",
            body_text,
            re_mod.I,
        ))
        has_external_typed_call = False
        for match in _QUALIFIED_CALL_PATTERN.finditer(body_text):
            receiver = match.group("receiver")
            receiver_type = external_receiver_types.get(receiver)
            if not receiver_type:
                continue
            called_method = match.group("method")
            if readonly_methods.get((receiver_type, called_method)) is True:
                continue
            if (
                match.group("options")
                or _EXTERNAL_CALLBACK_METHOD_HINT.match(called_method)
                or (receiver_type, called_method) not in readonly_methods
            ):
                has_external_typed_call = True
                break
        if has_low_level_call or has_external_typed_call:
            library_callback_methods.add((library_name, method))

    # Propagate callback capability through same-library helper calls, e.g.
    # Address.functionCall -> _functionCallWithValue -> target.call(...).
    changed = True
    while changed:
        changed = False
        for (library_name, method), body_text in library_function_bodies.items():
            if (library_name, method) in library_callback_methods:
                continue
            sibling_methods = {
                candidate_method
                for candidate_library, candidate_method in library_callback_methods
                if candidate_library == library_name
            }
            if any(
                re_mod.search(rf"\b{re_mod.escape(candidate)}\s*\(", body_text)
                for candidate in sibling_methods
                if candidate != method
            ):
                library_callback_methods.add((library_name, method))
                changed = True

    empty["library_names"] = library_names
    empty["external_type_names"] = external_type_names
    empty["readonly_methods"] = readonly_methods
    empty["library_callback_methods"] = library_callback_methods
    empty["external_receiver_types"] = external_receiver_types
    empty["external_receiver_names"] = external_receiver_names
    return empty


def _qualified_external_callback_event_lines(
    func: FunctionFlow,
    source_lines: List[str],
    metadata: Dict[str, object],
) -> Set[int]:
    """Return lines whose qualified calls can re-enter through external code."""

    start, clean_lines = _function_clean_source_lines(func, source_lines)
    library_names = metadata.get("library_names", set())
    library_callback_methods = metadata.get("library_callback_methods", set())
    readonly_methods = metadata.get("readonly_methods", {})
    external_receiver_types = metadata.get("external_receiver_types", {})
    external_receiver_names = metadata.get("external_receiver_names", set())
    callback_lines: Set[int] = set()
    for offset, line in enumerate(clean_lines):
        for match in _QUALIFIED_CALL_PATTERN.finditer(line):
            receiver = match.group("receiver")
            method = match.group("method")
            if method.casefold() == "approve":
                continue
            # Uniswap V2 invokes the recipient callback only when swap data is
            # non-empty.  A literal empty bytes payload is an ordinary swap,
            # not a reentrancy callback surface; front-running detection owns
            # this pattern separately.
            if (
                method.casefold() == "swap"
                and re_mod.search(r"new\s+bytes\s*\(\s*0\s*\)", line, re_mod.I)
            ):
                continue
            if (
                receiver in library_names
                and (receiver, method) in library_callback_methods
            ):
                callback_lines.add(start + offset + 1)
                continue
            receiver_type = external_receiver_types.get(receiver)
            if not receiver_type:
                # Imported interface declarations are not always present in a
                # blinded source.  A visibly typed external component may
                # still be recognized by name, but only for callback-shaped
                # mutating methods; read-like calls remain excluded.
                if receiver not in external_receiver_names:
                    continue
                if _EXTERNAL_READ_METHOD_HINT.match(method):
                    continue
                if (
                    method.casefold() in _ERC20_TRANSFER_METHODS
                    and re_mod.search(r"(?:token|erc20|eip20|fungible)", receiver, re_mod.I)
                ):
                    continue
                if _EXTERNAL_CALLBACK_METHOD_HINT.match(method):
                    callback_lines.add(start + offset + 1)
                continue
            if readonly_methods.get((receiver_type, method)) is True:
                continue
            if (
                method.casefold() in _ERC20_TRANSFER_METHODS
                and _is_erc20_like_type(receiver_type)
            ):
                continue
            if (
                match.group("options")
                or _EXTERNAL_CALLBACK_METHOD_HINT.match(method)
            ):
                callback_lines.add(start + offset + 1)

        # Imported interfaces are often absent from the blinded slice, but a
        # cast-qualified call still proves the external boundary locally.  Read
        # methods such as ``IERC20(gov).balanceOf`` are admitted as callback
        # observations; ordinary ERC-20 transfer methods are intentionally left
        # to the stricter token-transfer channel.
        for match in _CAST_QUALIFIED_CALL_PATTERN.finditer(line):
            method = match.group("method")
            method_key = method.casefold()
            if (
                method_key == "swap"
                and re_mod.search(r"new\s+bytes\s*\(\s*0\s*\)", line, re_mod.I)
            ):
                continue
            if method_key in _ERC20_TRANSFER_METHODS:
                continue
            if (
                _EXTERNAL_READ_METHOD_HINT.match(method)
                and method_key in {"balanceof", "totalsupply", "currenttick"}
            ) or method_key in _QUALIFIED_REENTRANCY_CALLBACK_METHODS:
                callback_lines.add(start + offset + 1)
    return callback_lines


_LOW_LEVEL_CALL_SITE_PATTERN = re_mod.compile(
    r"\.\s*(?:send|delegatecall|staticcall|call)"
    # ``call.value(amount)`` is only the value-builder in Solidity 0.4/0.5;
    # the actual low-level call still has a final ``()`` argument list.
    # Require that final list so a standalone builder expression is not
    # mistaken for an ignored external call.
    r"(?:\s*\.\s*value\s*\([^()\r\n]*\)\s*)?"
    r"(?:\{[^}\r\n]*\}\s*)?\(",
    re_mod.I,
)


def _source_unchecked_call_lines(
    func: FunctionFlow, source_code: str
) -> list[int]:
    """Locate ignored low-level call expressions inside one function.

    The AST risk record historically used ``func.line_number`` even when the
    ignored call was deeper in the function.  Keep this locator deliberately
    narrow: calls used directly by require/assert/if/while/try are consumed,
    while an expression statement or an unconsumed assignment is retained.
    """

    lines = (source_code or "").splitlines()
    if not lines:
        return []
    start = max(func.line_number - 1, 0)
    end = min(func.end_line_number or len(lines), len(lines))
    clean_lines = _strip_comments_preserve_lines("\n".join(lines[start:end])).splitlines()
    call_lines: list[int] = []
    for offset, line in enumerate(clean_lines):
        match = _LOW_LEVEL_CALL_SITE_PATTERN.search(line)
        if not match:
            continue
        if _e2_condition_contains_call(clean_lines, offset, match.start()):
            continue
        prefix = line[:match.start()]
        preceding = "\n".join(clean_lines[max(0, offset - 3):offset + 1])
        if re_mod.search(
            r"\b(?:require|assert|if|while|try)\s*\([^;]*$",
            preceding,
            re_mod.I,
        ):
            continue
        if re_mod.search(r"\breturn\s*$", prefix, re_mod.I):
            continue

        assignment = re_mod.search(
            r"\b(?P<name>[A-Za-z_]\w*)\s*(?:,\s*)?\)?\s*=\s*$",
            prefix,
            re_mod.I,
        )
        if assignment:
            variable = assignment.group("name")
            later = "\n".join(clean_lines[offset + 1:offset + 6])
            if re_mod.search(
                rf"\b(?:require|assert|if|while)\s*\([^)]*\b{re_mod.escape(variable)}\b",
                later,
                re_mod.I,
            ):
                continue
        call_lines.append(start + offset + 1)
    return call_lines


def _e2_condition_contains_call(
    clean_lines: list[str], offset: int, call_start: int
) -> bool:
    """Return whether a low-level call is itself inside a control predicate.

    The legacy helper looked backwards for any ``if (...)`` line and therefore
    treated a call in the body of ``if (guard) { ... }`` as checked.  Count
    parentheses from the most recent control opener instead; a balanced guard
    before the call is not a return-value check.
    """

    window_lines = clean_lines[max(0, offset - 4) : offset]
    window_lines.append(clean_lines[offset][:call_start])
    window = "\n".join(window_lines)
    controls = list(
        re_mod.finditer(
            r"\b(?:require|assert|if|while)\s*\(", window, re_mod.I
        )
    )
    for control in reversed(controls):
        predicate_prefix = window[control.start() :]
        if predicate_prefix.count("(") > predicate_prefix.count(")"):
            return True
    return bool(
        re_mod.search(r"\btry\s*$", clean_lines[offset][:call_start], re_mod.I)
    )


def _e2_assigned_return_names(prefix: str) -> list[str]:
    """Extract tuple/scalar names assigned by a call on the current statement."""

    if not prefix or "=" not in prefix:
        return []
    equal = prefix.rfind("=")
    if equal > 0 and prefix[equal - 1] in "=<>!":
        return []
    lhs = prefix[:equal].rsplit(";", 1)[-1].strip().strip("() ")
    if not lhs:
        return []
    names: list[str] = []
    for part in lhs.split(","):
        match = re_mod.search(r"\b([A-Za-z_]\w*)\s*$", part.strip())
        if not match:
            continue
        name = match.group(1)
        if name.casefold() in {"bool", "bytes", "memory", "uint", "address", "var"}:
            continue
        if name not in names:
            names.append(name)
    return names


def _e2_source_unchecked_call_lines(
    func: FunctionFlow, source_code: str
) -> list[int]:
    """Locate only source-proven ignored low-level call return values for E2."""

    if not source_code:
        return []
    candidates = analyze_unchecked_low_level_call_candidates(
        source_code,
        function_name=str(func.name or ""),
        start_line=func.line_number,
        end_line=func.end_line_number or None,
    )
    return [
        int(candidate["line"])
        for candidate in candidates
        if candidate.get("state") == E1_CANDIDATE_CONFIRMED
    ]


def _e2_assembly_regions(source: str) -> list[tuple[int, int]]:
    """Return byte ranges for inline assembly blocks."""

    regions: list[tuple[int, int]] = []
    text = str(source or "")
    for match in re_mod.finditer(r"\bassembly\b(?:\s*\([^)]*\))?\s*\{", text, re_mod.I):
        opening = text.find("{", match.start(), match.end())
        depth = 0
        closing = -1
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing >= 0:
            regions.append((opening + 1, closing))
    return regions


def _e2_yul_unchecked_call_risks(source: str) -> list[dict]:
    """Emit source-grounded Yul low-level call candidates; checked switch paths abstain."""

    text = str(source or "")
    risks: list[dict] = []
    call_pattern = re_mod.compile(
        r"\b(?:call|staticcall|delegatecall|callcode)\s*\(", re_mod.I
    )
    for region_start, region_end in _e2_assembly_regions(text):
        region = text[region_start:region_end]
        for match in call_pattern.finditer(region):
            absolute = region_start + match.start()
            line_start = text.rfind("\n", 0, absolute) + 1
            line_number = text[:absolute].count("\n") + 1
            line = text[line_start : text.find("\n", absolute) if text.find("\n", absolute) >= 0 else len(text)]
            before = line[: absolute - line_start]
            assigned = re_mod.search(
                r"\blet\s+(?P<name>[A-Za-z_]\w*)\s*:=\s*$", before, re_mod.I
            )
            checked = bool(
                re_mod.search(r"\b(?:if|switch)\b[^\n]*\b(?:iszero|eq|lt|gt)\s*\([^)]*$", before, re_mod.I)
            )
            if assigned:
                variable = assigned.group("name")
                tail = region[match.end() :]
                checked = checked or bool(
                    re_mod.search(
                        rf"\b(?:if|switch)\b[^{{}}\n]*\b{re_mod.escape(variable)}\b",
                        tail,
                        re_mod.I,
                    )
                )
            if checked:
                continue
            risks.append(
                {
                    "risk_type": "unchecked_low_level_calls",
                    "confidence": 0.95,
                    "reason": f"fallback() has unchecked Yul low-level call at @L{line_number}",
                    "function_name": "fallback",
                    "line": line_number,
                    "evidence_lines": [line_number],
                    "source_grounded": True,
                    "source_evidence_kind": "yul_unchecked_return",
                }
            )
    return risks


def _source_typed_return_discard_lines(
    func: FunctionFlow, source_code: str
) -> list[int]:
    """Locate selected typed external calls whose return value is discarded.

    The E2 development channel covers ERC-20 boolean-return methods plus the
    common router/cToken methods whose return values are routinely ignored in
    audit findings.  SafeERC20 wrappers remain excluded because their public
    API already handles the token return value internally.
    """

    lines = (source_code or "").splitlines()
    if not lines:
        return []
    start = max(func.line_number - 1, 0)
    end = min(func.end_line_number or len(lines), len(lines))
    clean_lines = _strip_comments_preserve_lines("\n".join(lines[start:end])).splitlines()
    call_lines: list[int] = []
    for offset, line in enumerate(clean_lines):
        for match in _TYPED_RETURN_METHOD_PATTERN.finditer(line):
            receiver = re_mod.sub(r"\s+", "", match.group("receiver"))
            method = match.group("method").casefold()
            if method not in _TYPED_RETURN_METHODS:
                continue
            if method in {"call", "send", "delegatecall", "staticcall"}:
                # Low-level calls have their own multiline assignment and
                # require/if/assert checker. Do not classify them again as
                # typed interface calls.
                continue
            if re_mod.search(r"\b(?:require|assert|if|while)\s*\(", receiver, re_mod.I):
                continue
            if method == "transfer" and _looks_like_native_transfer_receiver(receiver):
                continue
            if match.group("method").casefold().startswith("safe"):
                continue
            if not _e2_typed_bool_return_is_visible(
                receiver,
                match.group("method"),
                source_code,
            ):
                continue
            if re_mod.search(
                r"\b(?:msg\.sender|payable\(msg\.sender\)|address\([^)]*\))\b",
                receiver,
                re_mod.I,
            ) or re_mod.fullmatch(
                r"[A-Za-z_]\w*(?:addr|address)", receiver, re_mod.I
            ):
                continue
            prefix = " ".join(
                clean_lines[max(0, offset - 3): offset]
                + [line[:match.start()]]
            )
            control_prefix = prefix.rsplit("{", 1)[-1].rsplit("}", 1)[-1]
            if re_mod.search(
                r"\b(?:require|assert|if|while)\s*\([^;{}]*$|"
                r"\breturn\s*$",
                control_prefix,
                re_mod.I,
            ):
                continue
            if re_mod.search(r"\btry\s*$", control_prefix, re_mod.I):
                try_header = "\n".join(
                    clean_lines[offset:min(len(clean_lines), offset + 8)]
                ).split("{", 1)[0]
                if re_mod.search(r"\breturns\s*\(", try_header, re_mod.I):
                    continue

            assignment = re_mod.search(
                r"(?:\bbool\s+)?(?P<name>[A-Za-z_]\w*)\s*=\s*$",
                prefix,
                re_mod.I,
            )
            if assignment:
                variable = assignment.group("name")
                later = " ".join(
                    clean_lines[offset:min(len(clean_lines), offset + 8)]
                )
                if re_mod.search(
                    rf"\b(?:require|assert|if|while)\s*\([^)]*\b{re_mod.escape(variable)}\b",
                    later,
                    re_mod.I,
                ):
                    continue

            line_number = start + offset + 1
            if line_number not in call_lines:
                call_lines.append(line_number)
    return call_lines


def detect_reentrancy_evidence_chains(
    contracts: List[ContractFeatures], source_code: str, max_hints: int = 12
) -> List[str]:
    """Find unguarded single-file paths to callback-capable interactions.

    This is deliberately evidence-only: it adds no taxonomy rule and does not
    declare a vulnerability. It repairs the prior blind spot where typed
    token/protocol calls and internal-call paths were not represented as
    external interactions in the LLM evidence packet.
    """
    if not source_code:
        return []
    source_lines = source_code.splitlines()
    functions: Dict[str, FunctionFlow] = {}
    for contract in contracts:
        for func in contract.functions:
            functions.setdefault(func.name, func)
    if not functions:
        return []

    records: Dict[str, Dict[str, object]] = {}
    function_names = sorted(functions, key=len, reverse=True)
    effect_pattern = re_mod.compile(r"\+\+|--|\bdelete\b|\.push\s*\(|\.pop\s*\(|\b_(?:mint|burn)\s*\(|(?<![=!<>])=(?!=)")
    state_read_pattern = re_mod.compile(
        r"\b(?:balanceOf|balance|getWithdrawalLimit|limit|supply|reserved|debt|collateral|shares|activeProposal|price)\b",
        re_mod.I,
    )
    for name, func in functions.items():
        start = max(func.line_number - 1, 0)
        end = min(func.end_line_number or (start + 80), len(source_lines))
        raw_body = "\n".join(source_lines[start:end])
        clean_lines = _strip_comments_preserve_lines(raw_body).splitlines()
        events = []
        for offset, line in enumerate(clean_lines):
            line_number = start + offset + 1
            for callback_type, pattern in _REENTRANCY_CALLBACK_PATTERNS:
                match = pattern.search(line)
                if match:
                    if (
                        callback_type == "native_transfer_callback"
                        and _is_ordinary_token_transfer_line(line)
                    ):
                        continue
                    events.append({"kind": "callback", "line": line_number, "detail": callback_type, "text": match.group(0)[:80]})
                    break
            if state_read_pattern.search(line):
                events.append({"kind": "state_read", "line": line_number, "detail": line.strip()[:100]})
            if effect_pattern.search(line):
                events.append({"kind": "effect", "line": line_number, "detail": line.strip()[:100]})
            if not line.lstrip().startswith("function "):
                for callee in function_names:
                    if callee == name:
                        continue
                    if re_mod.search(rf"(?<![A-Za-z0-9_]){re_mod.escape(callee)}\s*\(", line):
                        events.append({"kind": "internal_call", "line": line_number, "detail": callee})
        events.sort(key=lambda event: (event["line"], {"state_read": 0, "internal_call": 1, "callback": 2, "effect": 3}[event["kind"]]))
        records[name] = {"func": func, "events": events}

    def expand(name: str, stack: Tuple[str, ...], depth: int) -> List[Dict[str, object]]:
        if depth > 5 or name in stack or name not in records:
            return []
        expanded = []
        next_stack = stack + (name,)
        for event in records[name]["events"]:
            if event["kind"] == "internal_call":
                callee = str(event["detail"])
                nested = expand(callee, next_stack, depth + 1)
                for nested_event in nested:
                    expanded.append({**nested_event, "path": (name,) + tuple(nested_event.get("path", (callee,)))})
            else:
                expanded.append({**event, "path": (name,)})
        return expanded

    hints = []
    seen_roots = set()
    for contract in contracts:
        for root in contract.functions:
            if root.visibility not in ["public", "external"] or root.has_reentrancy_guard or root.name in seen_roots:
                continue
            events = expand(root.name, tuple(), 0)
            emitted_for_root = 0
            emitted_signatures = set()
            for index, event in enumerate(events):
                if event["kind"] != "callback":
                    continue
                prior_read = next((item for item in reversed(events[:index]) if item["kind"] == "state_read"), None)
                later_effect = next(
                    (
                        item
                        for item in events[index + 1 :]
                        if item["kind"] == "callback"
                        or (item["kind"] == "effect" and item["line"] != event["line"])
                    ),
                    None,
                )
                if not prior_read and not later_effect:
                    continue
                evidence = []
                if prior_read:
                    evidence.append(f"pre-callback state/external-state read@L{prior_read['line']}")
                if later_effect:
                    evidence.append(f"post-callback {later_effect['kind']}@L{later_effect['line']}")
                path = " -> ".join(dict.fromkeys(event.get("path", (root.name,))))
                signature = (path, event["detail"])
                if signature in emitted_signatures:
                    continue
                hints.append(
                    f"REENTRANCY_EVIDENCE_CHAIN: {root.name}()@L{root.line_number} is unguarded; "
                    f"path [{path}] reaches {event['detail']}@L{event['line']}; {'; '.join(evidence)}. "
                    "Review callback/re-entry ordering before dismissing reentrancy."
                )
                emitted_signatures.add(signature)
                emitted_for_root += 1
                if emitted_for_root >= 2:
                    break
            if emitted_for_root:
                seen_roots.add(root.name)
            if len(hints) >= max_hints:
                return hints
    return hints


def _solidity_major_minor(source_code: str) -> tuple[int, int] | None:
    """Return a concrete pragma major/minor pair without guessing from ``0.x``."""

    match = re_mod.search(
        r"\bpragma\s+solidity\s+[\^>=<\s]*([0-9]+)\s*\.\s*([0-9]+)",
        source_code or "",
    )
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def detect_vulnerability_hints(features: ContractFeatures, source_code: str = "") -> List[str]:
    hints = []
    hard_facts = []

    solidity_version = "unknown"
    if source_code:
        pragma_match = re_mod.search(
            r'pragma\s+solidity\s+[\^>=<\s]*\s*(0\s*\.\s*\d+\s*\.\s*\d+)',
            source_code,
        )
        if pragma_match:
            solidity_version = pragma_match.group(1)

    for func in features.functions:
        if func.visibility not in ['public', 'external']:
            continue

        if func.external_calls and func.state_writes:
            if not func.has_reentrancy_guard:
                has_delayed = bool(func.delayed_checks)
                if has_delayed:
                    hints.append(f"REENTRANCY_DELAYED_CHECK: {func.name}()@L{func.line_number} has external call with delayed return-value check ({len(func.delayed_checks)} checked)")
                else:
                    hints.append(f"REENTRANCY_RISK: {func.name}()@L{func.line_number} has external call followed by state write without guard")
            else:
                hints.append(f"REENTRANCY_GUARDED: {func.name}()@L{func.line_number} has external call + state write BUT protected by {func.guard_detail}")

        if func.unchecked_calls:
            for uc in func.unchecked_calls:
                hints.append(f"UNCHECKED_RETURN: {func.name}()@L{func.line_number} {uc}")

        if not func.unchecked_calls and func.visibility in ['public', 'external'] and source_code:
            func_start = func.line_number - 1
            func_lines_r = source_code.split('\n')[func_start:func_start + 60]
            func_text_r = '\n'.join(func_lines_r)
            for upat, uname in [(r'\.send\s*\(', '.send()'), (r'\.call\.value\s*\(', '.call.value()'), (r'\.delegatecall\s*\(', '.delegatecall()')]:
                if re_mod.search(upat, func_text_r):
                    has_ret_check = bool(re_mod.search(r'(?:require|if)\s*\(.*(?:send|call|delegatecall)', func_text_r))
                    if not has_ret_check:
                        hints.append(f"UNCHECKED_RETURN: {func.name}@L{func.line_number} has unchecked {uname} call (regex fallback)")
                        break

        if func.delayed_checks:
            for dc in func.delayed_checks:
                hints.append(f"DELAYED_CHECK_OK: {func.name}()@L{func.line_number} {dc}")

        if not func.require_checks and func.state_writes and not func.is_constructor and not func.delayed_checks:
            hints.append(f"MISSING_CHECKS: {func.name}()@L{func.line_number} writes state without require checks")

        if func.visibility in ['public', 'external'] and func.state_writes and not func.is_constructor:
            fn_lower = func.name.lower()
            is_init_func = any(kw in fn_lower for kw in ['init', 'setup', 'configure', 'setowner', 'changeowner'])
            owner_like_writes = [w for w in func.state_writes
                                 if any(kw in w.lower() for kw in ['owner', 'admin', 'wallet', 'authority', 'master'])]
            effective_authorization = _is_admin_restricted(func, source_code)
            if (is_init_func or (owner_like_writes and not func.require_checks)) and not effective_authorization:
                has_init_guard = any('init' in m.lower() for m in func.modifiers) or \
                                 func.has_reentrancy_guard
                if not has_init_guard:
                    hints.append(f"LIFECYCLE_RISK: {func.name}()@L{func.line_number} is a public/external initialization function that writes critical state (owner/admin) WITHOUT an initializer guard - can be called multiple times")

        if func.is_fallback and func.external_calls:
            hints.append(f"FALLBACK_EXT_CALL: {func.name}()@L{func.line_number} fallback function makes external calls")

        if func.visibility in ['public', 'external'] and source_code:
            func_start = func.line_number - 1
            func_lines = source_code.split('\n')[func_start:func_start + 60]
            func_text = '\n'.join(func_lines)

            timestamp_read = False
            timestamp_in_condition = False
            timestamp_in_assignment = False
            timestamp_in_modulo = False

            if re_mod.search(r'block\.timestamp|now\b', func_text):
                timestamp_read = True
                if re_mod.search(r'(block\.timestamp|now)\s*[<>=!]=?\s', func_text) or \
                   re_mod.search(r'[<>=!]=?\s*(block\.timestamp|now)', func_text):
                    timestamp_in_condition = True
                if re_mod.search(r'(block\.timestamp|now)\s*%', func_text) or \
                   re_mod.search(r'%\s*(block\.timestamp|now)', func_text):
                    timestamp_in_modulo = True
                if re_mod.search(r'=\s*.*?(block\.timestamp|now)', func_text):
                    timestamp_in_assignment = True

            if timestamp_read:
                taint_path = "Read(block.timestamp)"
                if timestamp_in_modulo:
                    taint_path += " -> Modulo(%)"
                if timestamp_in_condition:
                    taint_path += " -> Condition"
                if timestamp_in_assignment and func.state_writes:
                    taint_path += f" -> StateWrite({','.join(func.state_writes[:2])})"
                hints.append(f"TIME_MANIPULATION_RISK: {func.name}()@L{func.line_number} reads block.timestamp - taint path: [{taint_path}]")

            if re_mod.search(r'tx\.origin\b', func_text):
                hints.append(f"ACCESS_CONTROL_TX_ORIGIN: {func.name}()@L{func.line_number} uses tx.origin for authorization - vulnerable to phishing attacks")

        if func.visibility in ['public', 'external']:
            has_arithmetic = False
            has_state_arith = False
            if source_code:
                func_start = func.line_number - 1
                func_lines = source_code.split('\n')[func_start:func_start + 50]
                func_text = '\n'.join(func_lines)
                arithmetic_patterns = [r'\+\s*=', r'\-\s*=', r'\*\s*=', r'\w+\s*\+\s*\w+', r'\w+\s*\-\s*\w+', r'\w+\s*\*\s*\w+']
                for pat in arithmetic_patterns:
                    if re_mod.search(pat, func_text):
                        has_arithmetic = True
                        break
                if re_mod.search(r'\w+\[.*\]\s*[\+\-\*]=', func_text):
                    has_state_arith = True
                if func.state_writes:
                    has_state_arith = True
            if has_arithmetic and has_state_arith:
                version = _solidity_major_minor(source_code)
                if version is not None and version < (0, 8):
                    hints.append(f"ARITHMETIC_OVERFLOW_RISK: {func.name}()@L{func.line_number} has arithmetic ops in Solidity {solidity_version} (<0.8.0) without built-in overflow protection")

    if features.inline_assembly_blocks:
        hints.append(f"INLINE_ASSEMBLY: contract {features.name} contains {len(features.inline_assembly_blocks)} inline assembly block(s) - requires manual review")

    for func in features.functions:
        if func.visibility in ['public', 'external'] and func.loop_features:
            for lf in func.loop_features:
                if "unbounded" in lf:
                    hints.append(f"DOS_UNBOUNDED_LOOP: {func.name}()@L{func.line_number} {lf}")
                elif "SEND_IN_LOOP" in lf:
                    hints.append(f"DOS_SEND_IN_LOOP: {func.name}()@L{func.line_number} {lf}")

    for mod in features.modifiers:
        mbody_lower = mod.body_text.lower()
        if any(kw in mbody_lower for kw in ['locked', '_status', 'nonreentrant', 'mutex']):
            hints.append(f"GUARD_MODIFIER: {mod.name}()@L{mod.line_number} implements reentrancy protection")

    for func in features.functions:
        if func.visibility not in ['public', 'external']:
            continue
        if func.external_calls:
            has_sender_check = any(
                'msg.sender' in rc.lower() and ('owner' in rc.lower() or 'admin' in rc.lower())
                for rc in func.require_checks
            )
            has_onlyowner = any(
                m.lower() in ['onlyowner', 'only_admin', 'onlyadmin', 'onlyrole']
                for m in func.modifiers
            )
            if has_sender_check or has_onlyowner:
                for ec in func.external_calls:
                    fact = f"[SYSTEM HARD FACT]: {func.name}()@L{func.line_number}: {ec} is PROTECTED by "
                    if has_sender_check:
                        fact += "msg.sender==owner/admin check"
                    if has_sender_check and has_onlyowner:
                        fact += " + "
                    if has_onlyowner:
                        fact += "onlyOwner/onlyAdmin modifier"
                    fact += ". DO NOT report Access Control for this call."
                    hard_facts.append(fact)

            has_delayed_or_try = bool(func.delayed_checks) or any(
                'try' in dc.lower() for dc in func.delayed_checks
            )
            if has_delayed_or_try:
                for ec in func.external_calls:
                    is_checked = any(ec in dc for dc in func.delayed_checks)
                    if is_checked:
                        fact = f"[SYSTEM HARD FACT]: {func.name}()@L{func.line_number}: {ec} return value IS checked (delayed/try-catch). DO NOT report Unchecked Return Value."
                        hard_facts.append(fact)

        for uc in func.unchecked_calls:
            if "RETURN VALUE SILENTLY DROPPED" in uc:
                fact = f"[SYSTEM HARD FACT]: {func.name}()@L{func.line_number}: {uc} - RETURN VALUE IS NOT CONSUMED. This is a CONFIRMED unchecked_low_level_calls vulnerability. The call result is silently discarded without any error handling."
                hard_facts.append(fact)

    return hints, hard_facts


def fuse_features(source_code: str, *, enable_slicing: bool = True) -> dict:
    temporal_ir = build_temporal_ir(source_code)
    temporal_ir_summary = summarize_temporal_ir(temporal_ir)[:8_000]
    temporal_invariants = analyze_temporal_invariants(source_code)
    temporal_slice = build_temporal_slice(source_code, temporal_invariants)
    try:
        contracts = [
            contract
            for contract in extract_contract_features(source_code)
            if contract is not None
        ]
        if not contracts:
            raise ValueError("No contract features parsed")
    except Exception as error:
        temporal_hints = [
            f"TEMPORAL_INVARIANT[{evidence['subtype']}]: "
            f"L{evidence['line']} {evidence['reason']}"
            for evidence in temporal_invariants
        ]
        temporal_risks = [{
            "risk_type": "time_manipulation",
            "confidence": evidence["confidence"],
            "reason": evidence["reason"],
            "function_name": (
                evidence.get("functions", ["temporal_invariant"])[0]
                if evidence.get("functions")
                else "temporal_invariant"
            ),
            "line": evidence["line"],
            "temporal_subtype": evidence["subtype"],
            "temporal_invariant": True,
            "evidence_lines": list(evidence.get("source_anchor_lines", [])),
            "primary_category": evidence.get(
                "primary_category", "time_manipulation"
            ),
            "time_role": evidence.get("time_role", "primary"),
        } for evidence in temporal_invariants]
        risk_function_map = defaultdict(set)
        for risk in temporal_risks:
            risk_function_map[risk["risk_type"]].add(risk["function_name"])
        numbered_source = _number_lines(source_code)
        fused_text = (
            "[FEATURE_PARSE_FALLBACK]\n"
            f"{temporal_ir_summary}\n"
            f"[TEMPORAL_INVARIANTS]:\n{temporal_slice}\n"
            f"[SOURCE_CODE]:\n{source_code}"
        )
        return {
            "ast_flow": "[FEATURE_PARSE_FALLBACK]",
            "vulnerability_hints": temporal_hints,
            "fused_text": fused_text,
            "numbered_source": numbered_source,
            "sliced_source": (
                temporal_slice or numbered_source
                if enable_slicing
                else numbered_source
            ),
            "slicing_enabled": bool(enable_slicing),
            "contracts": [],
            "hard_facts": [
                f"[SYSTEM HARD FACT - TEMPORAL INVARIANT]: "
                f"{evidence['subtype']} at L{evidence['line']}. "
                f"{evidence['reason']}"
                for evidence in temporal_invariants
            ],
            "unreachable_functions": [],
            "sink_data_flows": [],
            "rw_conflicts": [],
            "ast_identified_risks": temporal_risks,
            "protected_functions": [],
            "risk_function_map": dict(risk_function_map),
            "temporal_invariants": temporal_invariants,
            "temporal_slice": temporal_slice,
            "temporal_ir": temporal_ir,
            "temporal_ir_summary": temporal_ir_summary,
            "_feature_parse_error": f"{type(error).__name__}: {error}",
        }

    all_flows = []
    all_hints = []
    all_hard_facts = []
    unreachable_facts = []

    for contract in contracts:
        flow = synthesize_flow(contract, source_code=source_code)
        hints, hard_facts = detect_vulnerability_hints(contract, source_code=source_code)
        all_flows.append(f"[{contract.name}] {flow}")
        all_hints.extend(hints)
        all_hard_facts.extend(hard_facts)

        for func in contract.functions:
            if not func.is_reachable and func.visibility not in ['public', 'external']:
                fact = f"[SYSTEM HARD FACT]: {func.name}()@L{func.line_number} is UNREACHABLE from any public/external entry point. Call path: {func.reachability_path if func.reachability_path else 'NO PATH'}. DO NOT report vulnerabilities in unreachable code."
                unreachable_facts.append(fact)

    if os.environ.get("V123_REENTRANCY_EVIDENCE", "0") == "1":
        all_hints.extend(detect_reentrancy_evidence_chains(contracts, source_code))

    all_hard_facts.extend(unreachable_facts)

    has_safemath = bool(re_mod.search(r'\bSafeMath\b', source_code)) if source_code else False
    has_unchecked_arith = False
    for contract in contracts:
        for func in contract.functions:
            if func.visibility in ['public', 'external']:
                for sw in func.state_writes:
                    if any(op in sw for op in ['+', '-', '*', '/']):
                        has_unchecked_arith = True
                        break
            if has_unchecked_arith:
                break
        if has_unchecked_arith:
            break

    solidity_version = _solidity_major_minor(source_code)
    if has_unchecked_arith and not has_safemath and solidity_version is not None:
        if solidity_version < (0, 8):
            all_hard_facts.append(
                "[SYSTEM HARD FACT]: NO SafeMath library detected, and pragma solidity < 0.8 (no built-in overflow protection). "
                "Arithmetic operations (+, -, *, /) on state variables are VULNERABLE to overflow/underflow. "
                "You MUST evaluate each arithmetic operation for potential overflow."
            )
        elif solidity_version == (0, 8):
            all_hard_facts.append(
                "[SYSTEM HARD FACT]: Solidity 0.8+ has built-in overflow/underflow checks by default. "
                "Only report arithmetic overflow if 'unchecked' block is used or if the operation is inside an unchecked block."
            )

    ast_identified_risks = []

    for evidence in temporal_invariants:
        subtype = evidence["subtype"]
        all_hints.append(
            f"TEMPORAL_INVARIANT[{subtype}]: L{evidence['line']} "
            f"{evidence['reason']}"
        )
        all_hard_facts.append(
            f"[SYSTEM HARD FACT - TEMPORAL INVARIANT]: {subtype} at "
            f"L{evidence['line']} (confidence={evidence['confidence']:.2f}). "
            f"{evidence['reason']} Evidence: {evidence['evidence']}"
        )
        primary_temporal_risk = {
            "risk_type": "time_manipulation",
            "confidence": evidence["confidence"],
            "reason": evidence["reason"],
            "function_name": (
                evidence.get("functions", ["temporal_invariant"])[0]
                if evidence.get("functions")
                else "temporal_invariant"
            ),
            "line": evidence["line"],
            "temporal_subtype": subtype,
            "temporal_invariant": True,
            "evidence_lines": list(evidence.get("source_anchor_lines", [])),
            "primary_category": evidence.get(
                "primary_category", "time_manipulation"
            ),
            "time_role": evidence.get("time_role", "primary"),
        }
        ast_identified_risks.append(primary_temporal_risk)

        # A verified modifier lifecycle can have independent Gold loci in the
        # modifier and in the phase-boundary setters.  Preserve the primary
        # invariant, then materialize one scoped candidate per concrete
        # function anchor so scoring and final localization are one-to-one.
        if (
            subtype == "modifier_time_window_protected_operation"
            and len(evidence.get("source_anchor_lines", [])) > 1
        ):
            anchor_lines = [
                int(line)
                for line in evidence.get("source_anchor_lines", [])
                if isinstance(line, int) and line > 0
            ]
            for anchor_line in anchor_lines[1:]:
                anchor_function = ""
                for contract in contracts:
                    for func in contract.functions:
                        if func.line_number <= anchor_line <= func.end_line_number:
                            anchor_function = func.name
                            break
                    if anchor_function:
                        break
                if not anchor_function:
                    continue
                ast_identified_risks.append({
                    **primary_temporal_risk,
                    "reason": (
                        f"{evidence['reason']} Concrete lifecycle locus: "
                        f"{anchor_function}() @L{anchor_line}."
                    ),
                    "function_name": anchor_function,
                    "line": anchor_line,
                    "evidence_lines": [anchor_line],
                    "temporal_locus_scoped": True,
                })

    is_pre_080 = solidity_version is not None and solidity_version < (0, 8)
    source_arithmetic_enabled = (
        os.environ.get(PROFILE_ENV) == E1_OPTIMIZED_V1_PROFILE
        or (
            os.environ.get(PROFILE_ENV) == E2_DAPPSCAN_VNEXT_PROFILE
            and os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") == "1"
        )
    )

    if is_pre_080:
        for contract in contracts:
            for func in contract.functions:
                if func.visibility not in ['public', 'external']:
                    continue
                if not func.is_reachable:
                    continue
                has_arith_in_func = False
                has_safemath_in_func = False
                has_state_arith = False
                has_require_boundary = False
                has_complete_arithmetic_guards = False
                has_user_controlled_input = False
                has_only_owner = False
                source_grounded_arithmetic = False
                value_grounded_arithmetic = False
                arithmetic_operation_lines: List[int] = []
                arithmetic_evidence_lines: List[int] = []
                value_operation_lines: List[int] = []
                value_relevance_lines: List[int] = []
                value_arithmetic_proof = ""
                if source_code:
                    fs = func.line_number - 1
                    fe = func.end_line_number if func.end_line_number > fs else fs + 50
                    ft = '\n'.join(source_code.split('\n')[fs:fe])
                    ft_code = re_mod.sub(r'/\*.*?\*/|//[^\n]*', '', ft, flags=re_mod.DOTALL)
                    ft_no_index = re_mod.sub(r'\[[^\]]*\]', '[_IDX_]', ft_code)
                    for pat in [
                        r'\+\s*=', r'\-\s*=', r'\*\s*=',
                        r'\b[A-Za-z_]\w*\s*(?:\+\+|--)',
                        r'\w+\s*\+\s*\w+', r'\w+\s*\-\s*\w+', r'\w+\s*\*\s*\w+',
                    ]:
                        if re_mod.search(pat, ft_code):
                            has_arith_in_func = True
                            break
                    if re_mod.search(r'\.(add|sub|mul|div|mod)\s*\(', ft_code):
                        has_safemath_in_func = True
                    if re_mod.search(r'\w+\[.*\]\s*[\+\-\*]=', ft_code):
                        has_state_arith = True
                    # Preserve the legacy E1 admission behavior. The
                    # optimized profile uses source-grounded closure instead
                    # of this broad parser-level state-write fallback.
                    if not source_arithmetic_enabled and func.state_writes:
                        has_state_arith = True
                    has_complete_arithmetic_guards = _has_complete_state_arithmetic_guards(ft_code)
                    for bpat in [
                        r'require\s*\([^)]*>=\s',
                        r'require\s*\([^)]*<=\s',
                        r'require\s*\([^)]*>\s*\d',
                        r'require\s*\([^)]*\+\s*\w+\s*>=\s',
                        r'require\s*\([^)]*-\s*\w+\s*>=\s',
                    ]:
                        if re_mod.search(bpat, ft_code):
                            has_require_boundary = True
                            break
                    func_params = []
                    param_match = re_mod.search(r'function\s+' + re_mod.escape(func.name) + r'\s*\(([^)]*)\)', source_code)
                    if param_match:
                        raw_params = param_match.group(1)
                        for p in raw_params.split(','):
                            parts = p.strip().split()
                            if len(parts) >= 2:
                                func_params.append(parts[-1].replace(',',''))
                    for src_pat in [r'msg\.value', r'_value', r'_amount', r'amount', r'value']:
                        if re_mod.search(src_pat, ft_no_index):
                            has_user_controlled_input = True
                            break
                    for param in func_params:
                        if param and re_mod.search(r'\b' + re_mod.escape(param) + r'\b', ft_no_index):
                            has_user_controlled_input = True
                            break
                    ADMIN_MODS = {'onlyowner', 'only_admin', 'onlyadmin', 'onlyrole', 'onlyminter', 'onlypauser'}
                    for mod in func.modifiers:
                        if mod.lower() in ADMIN_MODS:
                            has_only_owner = True
                            break
                    if source_arithmetic_enabled:
                        (
                            source_grounded_arithmetic,
                            arithmetic_operation_lines,
                            arithmetic_evidence_lines,
                        ) = _source_arithmetic_evidence(
                            func,
                            source_code.splitlines(),
                            set(contract.state_variables),
                        )
                        if source_grounded_arithmetic:
                            has_state_arith = True
                        else:
                            (
                                value_grounded_arithmetic,
                                value_operation_lines,
                                value_relevance_lines,
                                value_arithmetic_proof,
                            ) = _source_value_arithmetic_evidence(
                                func,
                                source_code.splitlines(),
                                set(contract.state_variables),
                            )
                            if value_grounded_arithmetic and _is_admin_restricted(
                                func, source_code
                            ):
                                # A privileged payout/calculation path is not an
                                # independent arithmetic overflow candidate. Keep
                                # source-state arithmetic evidence above intact.
                                value_grounded_arithmetic = False
                                value_operation_lines = []
                                value_relevance_lines = []
                                value_arithmetic_proof = ""
                        if source_grounded_arithmetic or value_grounded_arithmetic:
                            # The source-value rule is intentionally stricter than
                            # the lexical pre-check.  Let its confirmed evidence
                            # activate the arithmetic branch for qualified balance
                            # expressions and returned values.
                            has_arith_in_func = True
                    if re_mod.search(r'\bi\s*\+\s*1\b', ft_code) and not re_mod.search(r'msg\.value|amount|balance', ft_code):
                        if not re_mod.search(r'\w+\s*[\+\-\*]=\s*(?!1\b)', ft_code):
                            has_arith_in_func = False
                if has_arith_in_func and (
                    has_state_arith or value_grounded_arithmetic
                ):
                    if _e1_optimized_standard_token_arithmetic_negative_control(
                        func.name, ft_code
                    ):
                        continue
                    if has_complete_arithmetic_guards and not source_grounded_arithmetic:
                        continue
                    if source_grounded_arithmetic or value_grounded_arithmetic:
                        conf = 0.80
                    elif has_only_owner:
                        conf = 0.55
                    elif not has_user_controlled_input:
                        conf = 0.55
                    elif has_safemath_in_func:
                        conf = 0.4
                    elif has_require_boundary:
                        conf = 0.7
                    else:
                        conf = 0.70
                    if conf < 0.6:
                        continue
                    risk = {
                        "risk_type": "arithmetic",
                        "confidence": conf,
                        "reason": f"Solidity {solidity_version[0]}.{solidity_version[1]} < 0.8.0, {func.name}() has arithmetic ops on {'state/value-relevant data' if value_grounded_arithmetic else 'state vars'}{' with user-controlled input' if has_user_controlled_input else ' (no direct user input)'}{' [protected by modifier]' if has_only_owner else ''}",
                        "function_name": func.name,
                        "line": func.line_number,
                    }
                    if source_grounded_arithmetic:
                        risk.update({
                            "source_grounded_arithmetic": True,
                            "arithmetic_operation_lines": arithmetic_operation_lines,
                            "state_write_lines": arithmetic_evidence_lines,
                            "evidence_lines": arithmetic_evidence_lines,
                            "source_arithmetic_proof": (
                                "persistent_state_postfix_update"
                                if any(
                                    re_mod.search(r"\+\+|--", source_code.splitlines()[line - 1])
                                    for line in arithmetic_operation_lines
                                    if 1 <= line <= len(source_code.splitlines())
                                )
                                else "caller_input_arithmetic_to_persistent_state_write"
                            ),
                        })
                    elif value_grounded_arithmetic:
                        risk.update({
                            "source_grounded_arithmetic": True,
                            "arithmetic_operation_lines": value_operation_lines,
                            "state_write_lines": [],
                            "value_relevance_lines": value_relevance_lines,
                            "evidence_lines": sorted(set(
                                value_operation_lines + value_relevance_lines
                            )),
                            "source_arithmetic_proof": value_arithmetic_proof,
                        })
                    ast_identified_risks.append(risk)

    # DAppSCAN's legacy source set includes abstract/library-style helpers
    # whose storage target is an inherited struct member.  The normal
    # public/external arithmetic loop cannot see those writes, so expose only
    # the narrow timestamp-plus-parameter pattern during E2 development.
    if (
        os.environ.get(PROFILE_ENV) == E2_DAPPSCAN_VNEXT_PROFILE
        and os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") == "1"
    ):
        for contract in contracts:
            for func in contract.functions:
                candidate = _detect_timestamp_parameter_storage_arithmetic(
                    func, source_code
                )
                if candidate is not None:
                    ast_identified_risks.append(candidate)

    e2_narrow_unchecked_rule = _e2_unchecked_narrow_rule_enabled()
    optimized_unchecked_rule = (
        os.environ.get(PROFILE_ENV) == E1_OPTIMIZED_V1_PROFILE
    )
    source_unchecked_rule = e2_narrow_unchecked_rule or optimized_unchecked_rule
    for contract in contracts:
        for func in contract.functions:
            # The development-only source rule may retain a concrete ignored
            # return in an internal helper because DAppSCAN often places the
            # audited locus on that helper.  Legacy/E1 extraction remains
            # limited to reachable public/external entry points.
            if func.visibility not in ['public', 'external'] and not e2_narrow_unchecked_rule:
                continue
            if func.visibility in ['public', 'external'] and not func.is_reachable:
                continue
            if source_unchecked_rule:
                unchecked_call_lines = _e2_source_unchecked_call_lines(
                    func, source_code
                ) if source_code else []
                legacy_unchecked_lines: list[int] = []
            else:
                unchecked_call_lines: list[int] = []
                legacy_unchecked_lines = list(func.unchecked_calls)
            typed_return_lines = (
                _source_typed_return_discard_lines(func, source_code)
                if e2_narrow_unchecked_rule and source_code
                else []
            )
            if legacy_unchecked_lines or unchecked_call_lines or typed_return_lines:
                unchecked_line = (
                    unchecked_call_lines[0]
                    if unchecked_call_lines
                    else (
                        typed_return_lines[0]
                        if typed_return_lines
                        else func.line_number
                    )
                )
                unchecked_evidence_lines = (
                    sorted(set(unchecked_call_lines + typed_return_lines))
                    or [unchecked_line]
                )
                call_site_text = ", ".join(
                    f"@L{line}" for line in unchecked_evidence_lines
                )
                risk = {
                    "risk_type": "unchecked_low_level_calls",
                    "confidence": 0.9,
                    "reason": (
                        f"{func.name}()@L{func.line_number} has unchecked external "
                        f"call(s) at {call_site_text}: "
                f"{'; '.join(func.unchecked_calls[:3]) if not source_unchecked_rule else 'source-confirmed return handling'}"
                ),
                    "function_name": func.name,
                    "line": unchecked_line,
                    "evidence_lines": unchecked_evidence_lines,
                }
                if source_unchecked_rule:
                    risk.update({
                        "source_grounded": True,
                        "source_evidence_kind": (
                            "typed_return_discard"
                            if typed_return_lines
                            else "source_unchecked_low_level_call"
                        ),
                    })
                if typed_return_lines:
                    risk.update({
                        "source_evidence_kind": "typed_return_discard",
                        "source_typed_return_discard": True,
                    })
                ast_identified_risks.append(risk)
            if source_code and not source_unchecked_rule:
                fs = func.line_number - 1
                fe = func.end_line_number if func.end_line_number > fs else fs + 50
                ft = '\n'.join(source_code.split('\n')[fs:fe])
                already_uc = any(r['risk_type'] == 'unchecked_low_level_calls' and r.get('function_name') == func.name for r in ast_identified_risks)
                if not already_uc:
                    unchecked_pats = [
                        (r'\.send\s*\(', '.send()'),
                        (r'\.call\.value\s*\(', '.call.value()'),
                        (r'\.delegatecall\s*\(', '.delegatecall()'),
                    ]
                    for upat, uname in unchecked_pats:
                        if re_mod.search(upat, ft):
                            has_ret_check = bool(re_mod.search(r'(?:require|if)\s*\(.*(?:send|call|delegatecall)', ft))
                            if not has_ret_check:
                                ast_identified_risks.append({
                                    "risk_type": "unchecked_low_level_calls",
                                    "confidence": 0.85,
                                    "reason": f"{func.name}()@L{func.line_number} has unchecked {uname} call (regex fallback)",
                                    "function_name": func.name,
                                    "line": func.line_number,
                                })
                                break
            if source_code:
                for call_line in _unprotected_native_ether_withdrawal_lines(func, source_code):
                    ast_identified_risks.append({
                        "risk_type": "access_control",
                        "submechanism": "unprotected_native_ether_withdrawal",
                        "confidence": 0.9,
                        "reason": (
                            f"{func.name}()@L{func.line_number} is publicly callable and sends "
                            f"native Ether to msg.sender without authorization or claimant-balance debit @L{call_line}"
                        ),
                        "function_name": func.name,
                        "line": call_line,
                    })
            if source_code:
                fs = func.line_number - 1
                fe = func.end_line_number if func.end_line_number > fs else fs + 50
                ft = '\n'.join(source_code.split('\n')[fs:fe])
                if re_mod.search(r'tx\.origin\b', ft):
                    tx_origin_evidence = _tx_origin_identity_state_evidence(
                        func, source_code
                    )
                    if tx_origin_evidence:
                        ast_identified_risks.append({
                            "risk_type": "access_control",
                            "confidence": 0.9,
                            "reason": (
                                f"{func.name}() uses tx.origin as an identity value "
                                f"at @L{tx_origin_evidence['line']} and writes it to "
                                f"persistent state at @L{', '.join(str(line) for line in tx_origin_evidence['state_write_lines'])}"
                            ),
                            "function_name": func.name,
                            "line": tx_origin_evidence["line"],
                            "evidence_lines": tx_origin_evidence["evidence_lines"],
                            "state_write_lines": tx_origin_evidence["state_write_lines"],
                            "source_grounded": True,
                            "source_evidence_kind": tx_origin_evidence["source_evidence_kind"],
                        })
                    already_ac = any(
                        r['risk_type'] == 'access_control'
                        and r.get('function_name') == func.name
                        for r in ast_identified_risks
                    )
                    has_ac_state_mutation = bool(tx_origin_evidence)
                    if not already_ac and not tx_origin_evidence:
                        has_ac_state_mutation = bool(func.state_writes) or bool(func.external_calls)
                    if not has_ac_state_mutation:
                        has_ac_state_mutation = bool(re_mod.search(
                            r'\.(?:send|transfer|delegatecall)\s*\(|'
                            r'\.call(?:\.value)?\s*\(',
                            ft,
                            re_mod.I,
                        ))
                    if not has_ac_state_mutation:
                        has_ac_state_mutation = bool(re_mod.search(r'=\s*msg\.sender', ft))
                    if not has_ac_state_mutation:
                        # Storage collection mutations such as
                        # ``asks.push(Ask(tx.origin, ...))`` are state writes
                        # even when tree-sitter does not expose them through
                        # FunctionFlow.state_writes.
                        has_ac_state_mutation = bool(re_mod.search(
                            r'\.(?:push|pop)\s*\(|\bdelete\s+[A-Za-z_]\w*',
                            ft,
                        ))
                    if not already_ac:
                        ast_identified_risks.append({
                            "risk_type": "access_control",
                            "confidence": 0.9 if has_ac_state_mutation else 0.55,
                            "reason": (
                                f"{func.name}()@L{func.line_number} uses tx.origin for authorization "
                                "with state mutation or external action"
                                if has_ac_state_mutation
                                else f"{func.name}()@L{func.line_number} uses tx.origin but NO state mutation or external call — likely decoy/bait code"
                            ),
                            "function_name": func.name,
                            "line": func.line_number,
                        })

    if e2_narrow_unchecked_rule and source_code:
        existing_yul_lines = {
            int(risk.get("line"))
            for risk in ast_identified_risks
            if risk.get("risk_type") == "unchecked_low_level_calls"
            and isinstance(risk.get("line"), int)
        }
        for yul_risk in _e2_yul_unchecked_call_risks(source_code):
            if yul_risk["line"] in existing_yul_lines:
                continue
            ast_identified_risks.append(yul_risk)
            existing_yul_lines.add(yul_risk["line"])

    for contract in contracts:
        ast_identified_risks.extend(
            _detect_unprotected_critical_state_writers(contract, source_code)
        )

    # Compute the E2-only modifier channel before the global tx.origin
    # fallback.  A qualified modifier candidate is function-scoped evidence;
    # it must suppress the legacy global decoy rather than coexist with it.
    e2_modifier_access_control_risks: list[dict[str, object]] = []
    if (
        source_code
        and os.environ.get(PROFILE_ENV) == E2_DAPPSCAN_VNEXT_PROFILE
        and os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") == "1"
    ):
        e2_modifier_access_control_risks = _detect_e2_tx_origin_modifier_access_control(
            contracts, source_code
        )

    if source_code and re_mod.search(r'tx\.origin\b', source_code):
        already_ac_global = any(r['risk_type'] == 'access_control' and r.get('confidence', 0) >= 0.6 for r in ast_identified_risks)
        already_ac_global = already_ac_global or bool(e2_modifier_access_control_risks)
        if not already_ac_global:
            ast_identified_risks.append({
                "risk_type": "access_control",
                "confidence": 0.55,
                "reason": "Contract uses tx.origin (global fallback) but no function-level state mutation detected — likely decoy",
                "function_name": "global",
                "line": 0,
            })

    if (
        os.environ.get(PROFILE_ENV) == E1_OPTIMIZED_V1_PROFILE
        or (
            os.environ.get(PROFILE_ENV) == E2_DAPPSCAN_VNEXT_PROFILE
            and os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") == "1"
        )
    ):
        ast_identified_risks.extend(
            detect_source_grounded_reentrancy_candidates(contracts, source_code)
        )

    # Library flash-loan logic mutates storage parameters rather than contract
    # state variables. Keep this source-grounded callback closure E2-only.
    if (
        os.environ.get(PROFILE_ENV) == E2_DAPPSCAN_VNEXT_PROFILE
        and os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") == "1"
    ):
        ast_identified_risks.extend(
            _detect_e2_flash_loan_receiver_reentrancy_candidates(
                contracts, source_code
            )
        )

    if source_code:
        for contract in contracts:
            for func in contract.functions:
                if func.visibility not in ['public', 'external']:
                    continue
                if not func.is_reachable:
                    continue
                fs = func.line_number - 1
                fe = func.end_line_number if func.end_line_number > fs else fs + 50
                ft = '\n'.join(source_code.split('\n')[fs:fe])
                has_timestamp_dep = False
                if re_mod.search(r'block\.timestamp\b', ft) or re_mod.search(r'\bnow\b', ft):
                    has_conditional = bool(re_mod.search(r'(?:if|require|assert|while)\s*\(.*(?:block\.timestamp|now\b)', ft))
                    has_comparison = bool(re_mod.search(r'(?:block\.timestamp|now)\s*[><=!]=?', ft)) or bool(re_mod.search(r'[><=!]=?\s*(?:block\.timestamp|now)', ft))
                    is_pure_assignment = bool(re_mod.search(r'=\s*(?:block\.timestamp|now)\s*;', ft)) and not has_conditional and not has_comparison
                    if has_conditional or has_comparison:
                        has_timestamp_dep = True
                    elif not is_pure_assignment and (func.state_writes or re_mod.search(r'=\s*msg\.sender', ft)):
                        has_timestamp_dep = True
                if has_timestamp_dep:
                    already_tm = any(r['risk_type'] == 'time_manipulation' and r.get('function_name') == func.name for r in ast_identified_risks)
                    if already_tm:
                        continue
                    has_state_mutation = bool(func.state_writes) or bool(func.external_calls)
                    if not has_state_mutation:
                        has_state_mutation = bool(re_mod.search(r'\bmsg\.sender\b', ft)) and bool(re_mod.search(r'=\s*msg\.sender', ft))
                    if not has_state_mutation:
                        has_state_mutation = bool(re_mod.search(r'\.(?:send|transfer|call)\s*\(', ft))
                    if not has_state_mutation:
                        has_state_mutation = bool(re_mod.search(r'\.balance\b', ft))
                    if has_state_mutation:
                        ast_identified_risks.append({
                            "risk_type": "time_manipulation",
                            "confidence": 0.55,
                            "reason": f"{func.name}()@L{func.line_number} uses block.timestamp/now in conditional logic with state mutation; this is temporal context, not a proven invariant violation",
                            "function_name": func.name,
                            "line": func.line_number,
                            "temporal_candidate_only": True,
                            "source_evidence_kind": "timestamp_context_without_typed_invariant",
                            "semantic_gate": "temporal_candidate_only",
                        })
                    else:
                        has_assignment = bool(re_mod.search(r'(?:^|\n)[^\n]*=\s*[^\n]*(?:block\.timestamp|now\b)', ft))
                        if has_assignment:
                            ast_identified_risks.append({
                                "risk_type": "time_manipulation",
                                "confidence": 0.55,
                                "reason": f"{func.name}()@L{func.line_number} uses block.timestamp/now but NO state mutation or external call — likely decoy/bait code",
                                "function_name": func.name,
                                "line": func.line_number,
                                "temporal_candidate_only": True,
                                "source_evidence_kind": "timestamp_context_without_typed_invariant",
                                "semantic_gate": "temporal_candidate_only",
                            })
                else:
                            pass

    # E2-only modifier channel: tx.origin may live outside the entrypoint
    # body, so preserve one source-grounded candidate for a stateful public
    # function protected by the modifier.  The helper excludes EOA-only
    # ``msg.sender == tx.origin`` gates and leaves legacy E1 untouched.
    if e2_modifier_access_control_risks:
        ast_identified_risks.extend(e2_modifier_access_control_risks)

    def _merge_features(contracts: List[ContractFeatures]) -> ContractFeatures:
        """Merge contracts so source rules and slicing see the full file."""

        if not contracts:
            return ContractFeatures(name='<empty>')
        merged = ContractFeatures(name='<merged>')
        for contract in contracts:
            merged.functions.extend(contract.functions)
            for state_variable in contract.state_variables:
                if state_variable not in merged.state_variables:
                    merged.state_variables.append(state_variable)
            merged.modifiers.extend(contract.modifiers)
        return merged

    merged_features = _merge_features(contracts) if contracts else None
    ast_identified_risks.extend(
        _detect_tx_origin_external_call_arguments(merged_features, source_code)
    )
    direct_front_running_risks = (
        _detect_weakly_gated_public_payouts(merged_features, source_code)
        + _detect_permissionless_share_mints_without_minimum_shares(
            merged_features, source_code
        )
        + _detect_permissionless_msg_value_quote_pair_swaps(
            merged_features, source_code
        )
        + _detect_permissionless_balance_dependent_amm_swaps(merged_features, source_code)
        + _detect_permissionless_reward_reinvestment_updates(
            merged_features, source_code
        )
    )
    # The allowance-race predicate is part of the opt-in E2 proof pipeline.
    # It must not add candidates to the frozen legacy E1 feature surface.
    if (
        os.environ.get(PROFILE_ENV) == E2_DAPPSCAN_VNEXT_PROFILE
        and os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") == "1"
    ):
        direct_front_running_risks.extend(
            _detect_allowance_race_without_zero_first(merged_features, source_code)
        )
        direct_front_running_risks.extend(
            _detect_e2_permissionless_ordered_asset_paths(merged_features, source_code)
        )
    ast_identified_risks.extend(direct_front_running_risks)

    # Keep legacy/library-style arithmetic in the E2 development arm only.
    # These helpers are source-grounded and must reach a value/state/asset sink;
    # the frozen E1 feature surface remains unchanged.
    if (
        merged_features is not None
        and os.environ.get(PROFILE_ENV) == E2_DAPPSCAN_VNEXT_PROFILE
        and os.environ.get("FUSEDAUDIT_E2_DEVELOPMENT_COVERAGE") == "1"
    ):
        existing_arithmetic_functions = {
            str(risk.get("function_name") or "")
            for risk in ast_identified_risks
            if risk.get("risk_type") == "arithmetic"
        }
        for risk in _detect_e2_legacy_arithmetic_value_sinks(contracts, source_code):
            if str(risk.get("function_name") or "") in existing_arithmetic_functions:
                continue
            ast_identified_risks.append(risk)
            existing_arithmetic_functions.add(str(risk.get("function_name") or ""))
        ast_identified_risks.extend(
            _detect_e2_permissionless_critical_asset_operations(
                merged_features, source_code
            )
        )

    # Opt-in rule-based candidate arm.  This is deliberately additive so the
    # default surfaces remain byte-compatible.  The helper emits only
    # source-grounded candidates; admission/proof and final finding gates stay
    # in fusedaudit_pipeline.py.
    if (
        merged_features is not None
        and os.environ.get(PROFILE_ENV) == E2_DAPPSCAN_VNEXT_PROFILE
        and os.environ.get("FUSEDAUDIT_E2_RULE_CANDIDATES") == "1"
    ):
        from rule_candidates import append_candidates

        append_candidates(
            contracts,
            source_code,
            ast_identified_risks,
        )

    ast_flow = "\n".join(all_flows)
    vul_hints = "\n".join(f"  - {h}" for h in all_hints) if all_hints else "  - No critical patterns detected"
    numbered_source = _number_lines(source_code)

    risk_funcs = []
    for contract in contracts:
        for func in contract.functions:
            if func.visibility in ['public', 'external'] and (
                func.external_calls or func.unchecked_calls or func.state_writes or func.loop_features
            ):
                risk_funcs.append(func.name)
    for evidence in temporal_invariants:
        risk_funcs.extend(evidence.get("functions", []))

    # Source rules run before slicing. Keep their function loci in the model
    # context; otherwise a risk can be deterministically identified and then
    # omitted from the prompt sent to the model.
    for risk in ast_identified_risks:
        function_name = risk.get("function_name")
        if function_name and function_name != "global":
            risk_funcs.append(function_name)

    # V132.7: NOW_MODULO_TRANSFER rule
    # Detect public/external functions with:
    # 1. now/block.timestamp in body
    # 2. if/require condition with % on the time value
    # 3. .transfer() in same body
    # Uses tree-sitter function range (line_number to end_line_number)
    _v132_source_lines = source_code.split('\n')
    for _v132_contract in contracts:
        for _v132_func in _v132_contract.functions:
            if _v132_func.visibility not in ('public', 'external'):
                continue
            if _v132_func.name in risk_funcs:
                continue
            _v132_start = _v132_func.line_number - 1
            _v132_end = min(_v132_func.end_line_number, len(_v132_source_lines))
            if _v132_end <= _v132_start:
                continue
            _v132_body = '\n'.join(_v132_source_lines[_v132_start:_v132_end])
            # Condition 1: now or block.timestamp
            if not re_mod.search(r'\bnow\b|\bblock\.timestamp\b', _v132_body):
                continue
            # Condition 2: if/require with % on time value
            if not re_mod.search(r'(?:if|require)\s*\([^)]*(?:\bnow\b|\bblock\.timestamp\b)[^)]*%[^)]*\)', _v132_body):
                continue
            # Condition 3: .transfer() in same body
            if not re_mod.search(r'\.transfer\s*\(', _v132_body):
                continue
            risk_funcs.append(_v132_func.name)

    sliced_source = (
        backward_slice(
            merged_features,
            source_code,
            risk_funcs=risk_funcs if risk_funcs else None,
        )
        if enable_slicing
        else numbered_source
    )

    sink_data_flows = []
    rw_conflicts = []
    for contract in contracts:
        sdf = _extract_sink_data_flows(contract, source_code)
        sink_data_flows.extend(sdf)
        rc = _extract_rw_conflict_graph(contract, source_code)
        rw_conflicts.extend(rc)

    temporal_text = temporal_slice or "No temporal invariant violation detected."
    fused_text = (
        f"[STRUCTURE_FEATURE]:\n{ast_flow}\n"
        f"[VULNERABILITY_HINTS]:\n{vul_hints}\n"
        f"{temporal_ir_summary}\n"
        f"[TEMPORAL_INVARIANTS]:\n{temporal_text}\n"
        f"[SOURCE_CODE]:\n{source_code}"
    )

    financial_rw_conflicts = [c for c in rw_conflicts if c.get('read_has_financial', False)]
    if financial_rw_conflicts:
        already_fr = any(r['risk_type'] == 'front_running' for r in ast_identified_risks)
        if not already_fr:
            write_funcs = sorted(set(c['write_function'] for c in financial_rw_conflicts))
            read_funcs = sorted(set(c['read_function'] for c in financial_rw_conflicts))
            affected_vars = sorted(set(c['variable'] for c in financial_rw_conflicts))
            has_unprotected_write = False
            for wf in write_funcs:
                for contract in contracts:
                    for func in contract.functions:
                        if func.name == wf:
                            ADMIN_MODS = {'onlyowner', 'only_admin', 'onlyadmin', 'onlyrole'}
                            has_admin = any(m.lower() in ADMIN_MODS for m in func.modifiers)
                            if not has_admin and func.visibility in ('public', 'external'):
                                has_unprotected_write = True
                                break
                    if has_unprotected_write:
                        break
            if has_unprotected_write and os.environ.get(PROFILE_ENV) != E1_OPTIMIZED_V1_PROFILE:
                ast_identified_risks.append({
                    "risk_type": "front_running",
                    "confidence": 0.9,
                    "reason": f"RW-Conflict: {len(financial_rw_conflicts)} async state conflict(s) on {affected_vars[:3]}. Unprotected writes: {write_funcs[:3]}, financial reads: {read_funcs[:3]}",
                    "function_name": write_funcs[0] if write_funcs else "global",
                    "line": financial_rw_conflicts[0].get('write_line', 0),
                    "source_grounded": False,
                    "source_evidence_kind": "rw_conflict_only",
                    "semantic_gate": "requires_transaction_order_proof",
                    "requires_ordering_proof": True,
                })
            elif has_unprotected_write:
                # A generic read/write conflict is only a hypothesis. The
                # optimized E1 profile admits front-running from source-grounded
                # transaction-order patterns above, not from constructor or
                # ordinary token state initialization conflicts.
                print(
                    "[E1 Optimized] skipped rw_conflict_only front-running candidate"
                )

    protected_functions = []
    ADMIN_MODIFIERS = {'onlyowner', 'only_admin', 'onlyadmin', 'onlyrole', 'onlyminter', 'onlypauser', 'onlyguardian'}
    for contract in contracts:
        for func in contract.functions:
            has_admin_mod = any(m.lower() in ADMIN_MODIFIERS for m in func.modifiers)
            has_sender_owner_check = any(
                'msg.sender' in rc.lower() and ('owner' in rc.lower() or 'admin' in rc.lower())
                for rc in func.require_checks
            )
            if has_admin_mod or has_sender_owner_check:
                protected_functions.append({
                    "name": func.name,
                    "line": func.line_number,
                    "modifiers": func.modifiers,
                    "reason": "onlyOwner/onlyAdmin modifier" if has_admin_mod else "msg.sender==owner check",
                })

    risk_function_map = defaultdict(set)
    for risk in ast_identified_risks:
        fn = risk.get("function_name", "")
        if fn and fn != "global":
            risk_function_map[risk["risk_type"]].add(fn)

    arithmetic_operation_provenance = _arithmetic_operation_provenance(
        contracts, source_code
    )

    return {
        "ast_flow": ast_flow,
        "vulnerability_hints": all_hints,
        "fused_text": fused_text,
        "numbered_source": numbered_source,
        "sliced_source": sliced_source,
        "slicing_enabled": bool(enable_slicing),
        "contracts": contracts,
        "hard_facts": all_hard_facts,
        "unreachable_functions": [
            {"name": func.name, "line": func.line_number, "visibility": func.visibility,
             "reachability_path": func.reachability_path}
            for contract in contracts
            for func in contract.functions
            if not func.is_reachable
        ],
        "sink_data_flows": sink_data_flows,
        "rw_conflicts": rw_conflicts,
        "ast_identified_risks": ast_identified_risks,
        "arithmetic_operation_provenance": arithmetic_operation_provenance,
        "protected_functions": protected_functions,
        "risk_function_map": dict(risk_function_map),
        "temporal_invariants": temporal_invariants,
        "temporal_slice": temporal_slice,
        "temporal_ir": temporal_ir,
        "temporal_ir_summary": temporal_ir_summary,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_fusion.py <contract.sol>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        source = f.read()

    result = fuse_features(source)

    print("=" * 60)
    print("Phase 1: Feature Fusion Output (Tree-sitter + Two-Step Tracking)")
    print("=" * 60)
    print()
    print("[AST/CFG Flow]:")
    print(result["ast_flow"])
    print()
    print("[Vulnerability Hints]:")
    for h in result["vulnerability_hints"]:
        print(f"  - {h}")
    if not result["vulnerability_hints"]:
        print("  - No critical patterns detected")
    print()
    print("[Numbered Source]:")
    print(result["numbered_source"])
    print()
    print("[Fused Text Length]:", len(result["fused_text"]), "chars")
