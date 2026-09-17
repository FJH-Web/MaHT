"""
方差齐性检验方法实现模块

本模块实现方差齐性假设检验方法：
- run_f_variance: 两样本方差 F 检验
- run_bartlett: Bartlett 方差齐性检验
- run_levene: Levene 方差齐性检验
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from statagent.protocol.statistical_object import MethodResult
from statagent.statistical_library.common import numeric


def _groups(data: dict[str, Any], minimum: int = 2) -> list[np.ndarray]:
    """从数据中提取并验证分组数据"""
    groups = [np.asarray(group, dtype=float) for group in data["groups"]]
    groups = [group[np.isfinite(group)] for group in groups]
    if len(groups) < 2 or any(len(group) < minimum for group in groups):
        raise ValueError("方差检验至少需要两个组，且每组至少有两个有限观测")
    return groups


def run_f_variance(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    """两样本方差 F 检验"""
    a = numeric(data, "group_a", minimum=2)
    b = numeric(data, "group_b", minimum=2)
    variance_a, variance_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    if variance_b == 0:
        raise ValueError("B 组样本方差为 0，无法计算方差比")
    statistic = variance_a / variance_b
    cdf = float(stats.f.cdf(statistic, len(a) - 1, len(b) - 1))
    alternative = parameters.get("alternative", "two-sided")
    p_value = min(1.0, 2.0 * min(cdf, 1.0 - cdf)) if alternative == "two-sided" else (1.0 - cdf if alternative == "greater" else cdf)
    return MethodResult("VAR001", "run_f_variance", {"f": statistic, "variance_a": variance_a, "variance_b": variance_b, "df_a": len(a) - 1, "df_b": len(b) - 1}, p_value=float(p_value), effect_size={"name": "variance_ratio", "value": statistic})


def run_bartlett(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    groups = _groups(data)
    result = stats.bartlett(*groups)
    return MethodResult("VAR002", "run_bartlett", {"chi2": float(result.statistic), "groups": len(groups), "n": sum(map(len, groups))}, p_value=float(result.pvalue), assumption_checks=[{"name": "normality_required", "passed": None, "reason": "Bartlett 检验对非正态分布敏感"}])


def run_levene(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    groups = _groups(data)
    center = str(parameters.get("center", "median"))
    if center not in {"mean", "median", "trimmed"}:
        raise ValueError("center 必须为 mean、median 或 trimmed")
    result = stats.levene(*groups, center=center, proportiontocut=float(parameters.get("proportiontocut", 0.05)))
    return MethodResult("VAR003", "run_levene", {"w": float(result.statistic), "groups": len(groups), "n": sum(map(len, groups)), "center": center}, p_value=float(result.pvalue))