"""
数据集意图解析模块

本模块从原始用户需求中锁定数据列，并生成不可越权的数据映射。
主要功能：
- is_identifier_name: 判断列名是否为标识符
- resolve_requested_columns: 从用户文本中解析需要的列
- eligible_profile_columns: 获取符合条件的数据列
- enforce_locked_mapping: 强制执行锁定列的数据映射
- mapping_columns: 生成数据映射

设计原则：
- 显式点名的列拥有最高优先级
- 标识符列（如索引、编号）不用于分析
"""

from __future__ import annotations

import re
from typing import Any, Iterable


_ORDINALS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_COLUMN_KEYS = {"column", "columns", "value_column", "group_column", "row_column", "column_column"}


def is_identifier_name(name: str) -> bool:
    """判断列名是否为标识符（如索引、编号等）"""
    normalized = re.sub(r"[\s_.:-]+", "", str(name).casefold())
    return bool(
        re.fullmatch(r"unnamed\d*", normalized)
        or normalized in {"index", "row", "rowid", "id", "序号", "编号", "行号", "索引"}
        or normalized.endswith("rowid")
    )


def resolve_requested_columns(user_text: str, profile: dict[str, Any]) -> list[str]:
    """按用户文本中的出现顺序解析真实列名；显式点名校拥有最高优先级。"""
    text = user_text.casefold()
    columns = [str(item["name"]) for item in profile.get("columns", []) if item.get("name") is not None]
    found: list[tuple[int, int, str]] = []

    for column in sorted(columns, key=len, reverse=True):
        start = text.find(column.casefold())
        if start >= 0:
            found.append((start, -len(column), column))

    for match in re.finditer(r"第\s*([一二三四五六七八九十]|\d+)\s*列", user_text):
        token = match.group(1)
        position = int(token) if token.isdigit() else _ORDINALS.get(token, 1)
        if 1 <= position <= len(columns):
            found.append((match.start(), -len(columns[position - 1]), columns[position - 1]))

    valid_columns = [c for c in columns if not is_identifier_name(c)]
    total_valid = len(valid_columns)

    for match in re.finditer(r"前(\d+)个变量", user_text):
        n = int(match.group(1))
        for i, col in enumerate(valid_columns[:n]):
            found.append((match.start(), -i, col))

    for match in re.finditer(r"后(\d+)个变量", user_text):
        n = int(match.group(1))
        start_idx = max(0, total_valid - n)
        for i, col in enumerate(valid_columns[start_idx:]):
            found.append((match.start(), -(start_idx + i), col))

    for match in re.finditer(r"第(\d+)[-到至]第(\d+)个变量", user_text):
        start_pos = int(match.group(1))
        end_pos = int(match.group(2))
        for i in range(max(0, start_pos - 1), min(end_pos, total_valid)):
            found.append((match.start(), -i, valid_columns[i]))

    for match in re.finditer(r"第(\d+)[、,，](.*?)个变量", user_text):
        first_pos = int(match.group(1))
        rest_text = match.group(2)
        positions = [first_pos]
        for num_match in re.finditer(r"第(\d+)个", "第" + rest_text + "个"):
            positions.append(int(num_match.group(1)))
        for pos in positions:
            if 1 <= pos <= total_valid:
                found.append((match.start(), -pos, valid_columns[pos - 1]))

    for match in re.finditer(r"中间(\d+)个变量", user_text):
        n = int(match.group(1))
        if total_valid >= n:
            start_idx = (total_valid - n) // 2
            for i, col in enumerate(valid_columns[start_idx:start_idx + n]):
                found.append((match.start(), -i, col))

    for match in re.finditer(r"除了前(\d+)个以外的变量", user_text):
        n = int(match.group(1))
        for i, col in enumerate(valid_columns[n:]):
            found.append((match.start(), -i, col))

    for match in re.finditer(r"除了后(\d+)个以外的变量", user_text):
        n = int(match.group(1))
        for i, col in enumerate(valid_columns[:-n] if n < total_valid else valid_columns):
            found.append((match.start(), -i, col))

    ordered: list[str] = []
    for _, _, column in sorted(found):
        if column not in ordered:
            ordered.append(column)
    return ordered


def eligible_profile_columns(profile: dict[str, Any]) -> list[str]:
    return [
        str(item["name"])
        for item in profile.get("columns", [])
        if item.get("name") is not None
        and not item.get("identifier_candidate", False)
        and not is_identifier_name(str(item["name"]))
    ]


def mapping_columns(mapping: Any) -> list[str]:
    values: list[str] = []
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            if key in _COLUMN_KEYS:
                if isinstance(value, str):
                    values.append(value)
                elif isinstance(value, list):
                    values.extend(str(item) for item in value)
            elif isinstance(value, str) and key != "kind":
                # 兼容 LLM 常见简写：{"group_a": "Sepal.Length"}。
                values.append(value)
            else:
                values.extend(mapping_columns(value))
    elif isinstance(mapping, list):
        for item in mapping:
            values.extend(mapping_columns(item))
    return list(dict.fromkeys(values))


def enforce_locked_mapping(
    inputs: Iterable[str],
    locked_columns: list[str],
    proposed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把统计输入角色确定性地绑定到用户点名列，覆盖越权的 LLM 映射。"""
    roles = tuple(inputs)
    mapping = dict(proposed or {})
    locked = list(dict.fromkeys(locked_columns))
    if not locked:
        return mapping

    if roles == ("group_a", "group_b") or roles == ("before", "after"):
        if len(locked) >= 2:
            return {
                roles[0]: {"kind": "column", "column": locked[0]},
                roles[1]: {"kind": "column", "column": locked[1]},
            }
    if roles == ("values",):
        return {"values": {"kind": "column", "column": locked[0]}}
    if roles == ("groups",):
        return {"groups": {"kind": "columns", "columns": locked}}
    if roles == ("x",):
        return {"x": {"kind": "matrix", "columns": locked}}
    if roles == ("table",) and len(locked) >= 2:
        return {"table": {"kind": "contingency", "row_column": locked[0], "column_column": locked[1]}}
    if roles == ("x", "y") and len(locked) >= 2:
        proposed_y = mapping_columns(mapping.get("y", {}))
        y_column = proposed_y[0] if proposed_y and proposed_y[0] in locked else locked[-1]
        x_columns = [column for column in locked if column != y_column]
        return {
            "x": {"kind": "matrix", "columns": x_columns},
            "y": {"kind": "column", "column": y_column},
        }
    if roles == ("time", "event", "group") and len(locked) >= 3:
        return {role: {"kind": "column", "column": column} for role, column in zip(roles, locked, strict=False)}
    if roles == ("time", "event", "x") and len(locked) >= 3:
        return {
            "time": {"kind": "column", "column": locked[0]},
            "event": {"kind": "column", "column": locked[1]},
            "x": {"kind": "matrix", "columns": locked[2:]},
        }

    # 保留只引用锁定列的复杂映射；删除任何未获用户授权的列。
    safe: dict[str, Any] = {}
    for role, spec in mapping.items():
        referenced = mapping_columns(spec)
        if referenced and set(referenced).issubset(locked):
            safe[role] = spec
    for role, column in zip((role for role in roles if role not in safe), locked, strict=False):
        safe[role] = {"kind": "column", "column": column}
    return safe