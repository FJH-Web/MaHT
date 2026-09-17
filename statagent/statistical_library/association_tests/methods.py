"""
相关性与随机性假设检验方法实现模块

本模块实现相关性与随机性假设检验方法：
- run_correlation_test: 双变量相关性检验
- run_correlation_matrix_tests: 多变量两两相关性检验
- run_randomness_runs_test: 随机性游程检验
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from statagent.protocol.statistical_object import MethodResult
from statagent.statistical_library.common import matrix, numeric, require_same_length


def _correlation(x: np.ndarray, y: np.ndarray, method: str, alternative: str):
    """根据方法计算相关系数和 p 值"""
    if method == "pearson":
        return stats.pearsonr(x, y, alternative=alternative)
    if method == "spearman":
        return stats.spearmanr(x, y, alternative=alternative)
    if method == "kendall":
        return stats.kendalltau(x, y, alternative=alternative)
    raise ValueError("method 必须为 pearson、spearman 或 kendall")


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def run_correlation_test(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    x, y = numeric(data, "variable_x", minimum=3), numeric(data, "variable_y", minimum=3)
    require_same_length(x, y)
    method = str(parameters.get("method", "pearson"))
    result = _correlation(x, y, method, parameters.get("alternative", "two-sided"))
    return MethodResult("COR001", "run_correlation_test", {"correlation": float(result.statistic), "method": method, "n": len(x)}, p_value=float(result.pvalue), effect_size={"name": f"{method}_correlation", "value": float(result.statistic)})


def run_correlation_matrix_tests(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    x = matrix(data, "x")
    if x.shape[1] < 2:
        raise ValueError("相关矩阵检验至少需要两个变量")
    method = str(parameters.get("method", "pearson"))
    names = list(parameters.get("variable_names", [f"变量{i + 1}" for i in range(x.shape[1])]))
    comparisons, p_values = [], []
    for left in range(x.shape[1]):
        for right in range(left + 1, x.shape[1]):
            result = _correlation(x[:, left], x[:, right], method, "two-sided")
            comparisons.append({"variable_a": names[left], "variable_b": names[right], "correlation": float(result.statistic), "raw_p_value": float(result.pvalue)})
            p_values.append(float(result.pvalue))
    adjusted = _holm(p_values)
    for comparison, value in zip(comparisons, adjusted, strict=True):
        comparison["adjusted_p_value"] = value
    return MethodResult("COR002", "run_correlation_matrix_tests", {"comparisons": comparisons, "method": method, "adjustment": "Holm", "variables": x.shape[1], "n": x.shape[0]}, p_value=float(min(adjusted)))


def run_randomness_runs_test(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    values = numeric(data, "values", minimum=3)
    cutoff_param = parameters.get("cutoff")
    if cutoff_param is not None:
        cutoff = float(cutoff_param)
    else:
        cutoff = float(np.percentile(values, 50))
    labels = values >= cutoff
    n_above = int(labels.sum())
    n_below = int((~labels).sum())
    if n_above == 0 or n_below == 0:
        cutoff = float(np.percentile(values, 40))
        labels = values >= cutoff
        n_above = int(labels.sum())
        n_below = int((~labels).sum())
    if n_above == 0 or n_below == 0:
        raise ValueError("截断值两侧都必须存在观测，请提供自定义截断值")
    runs = int(1 + np.sum(labels[1:] != labels[:-1]))
    n1, n2 = int(labels.sum()), int((~labels).sum())
    expected = 1.0 + 2.0 * n1 * n2 / (n1 + n2)
    variance = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    z = (runs - expected) / np.sqrt(variance) if variance > 0 else 0.0
    return MethodResult("RUN001", "run_randomness_runs_test", {"runs": runs, "expected_runs": expected, "variance": variance, "z": float(z), "cutoff": cutoff, "above_or_equal": n1, "below": n2}, p_value=float(2.0 * stats.norm.sf(abs(z))))