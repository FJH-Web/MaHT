from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from statagent.protocol.statistical_object import MethodResult
from statagent.statistical_library.common import numeric


def run_mann_whitney(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    a, b = numeric(data, "group_a"), numeric(data, "group_b")
    result = stats.mannwhitneyu(a, b, alternative=parameters.get("alternative", "two-sided"))
    rank_biserial = 1.0 - 2.0 * float(result.statistic) / (len(a) * len(b))
    return MethodResult("NONPAR001", "run_mann_whitney", {"u": float(result.statistic)}, p_value=float(result.pvalue), effect_size={"name": "rank_biserial", "value": rank_biserial})


def run_wilcoxon(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    before, after = numeric(data, "before"), numeric(data, "after")
    if len(before) != len(after):
        raise ValueError("before and after must have equal lengths")
    result = stats.wilcoxon(before, after, alternative=parameters.get("alternative", "two-sided"))
    from ..common import normality_check, missing_check, outlier_check_iqr
    difference = after - before
    n_nonzero = np.sum(difference != 0)
    rank_biserial = 1 - (2 * result.statistic) / (n_nonzero * (n_nonzero + 1)) if n_nonzero > 0 else 0.0
    return MethodResult("NONPAR002", "run_wilcoxon", {"w": float(result.statistic), "n": len(before), "rank_biserial": float(rank_biserial)}, p_value=float(result.pvalue), effect_size={"name": "Rank-biserial r", "value": float(rank_biserial)}, assumption_checks=[normality_check(difference), missing_check({"before": before, "after": after}), outlier_check_iqr(difference)])


def run_kruskal_wallis(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    groups = [np.asarray(group, dtype=float) for group in data["groups"]]
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis needs at least two groups")
    result = stats.kruskal(*groups)
    n = sum(map(len, groups))
    epsilon_squared = (result.statistic - len(groups) + 1) / (n - len(groups)) if n > len(groups) else 0.0
    return MethodResult("NONPAR003", "run_kruskal_wallis", {"h": float(result.statistic), "groups": len(groups), "n": n}, p_value=float(result.pvalue), effect_size={"name": "epsilon_squared", "value": float(epsilon_squared)})


def run_wilcoxon_one_sample(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    values = numeric(data, "values")
    null_median = float(parameters.get("null_median", 0.0))
    result = stats.wilcoxon(values - null_median, alternative=parameters.get("alternative", "two-sided"), zero_method=parameters.get("zero_method", "wilcox"))
    return MethodResult("NONPAR004", "run_wilcoxon_one_sample", {"w": float(result.statistic), "sample_median": float(np.median(values)), "null_median": null_median, "n": len(values)}, p_value=float(result.pvalue))


def _trimmed_range(values: np.ndarray, fraction: float) -> float:
    ordered = np.sort(values)
    trim = min(int(np.floor(len(ordered) * fraction)), max(0, (len(ordered) - 2) // 2))
    trimmed = ordered[trim: len(ordered) - trim] if trim else ordered
    return float(trimmed[-1] - trimmed[0])


def run_moses_extreme_reactions(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    control = numeric(data, "group_a", minimum=3)
    treatment = numeric(data, "group_b", minimum=3)
    trim_fraction = float(parameters.get("trim_fraction", 0.05))
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction 必须位于 [0, 0.5) 区间")
    observed = _trimmed_range(treatment, trim_fraction) - _trimmed_range(control, trim_fraction)
    combined = np.concatenate([control, treatment])
    iterations = int(parameters.get("permutation_iterations", 1999))
    rng = np.random.default_rng(int(parameters.get("random_state", 2026)))
    exceed = 0
    for _ in range(iterations):
        permuted = rng.permutation(combined)
        statistic = _trimmed_range(permuted[len(control):], trim_fraction) - _trimmed_range(permuted[:len(control)], trim_fraction)
        exceed += statistic >= observed
    p_value = (exceed + 1) / (iterations + 1)
    return MethodResult("NONPAR005", "run_moses_extreme_reactions", {"range_difference": observed, "control_trimmed_range": _trimmed_range(control, trim_fraction), "treatment_trimmed_range": _trimmed_range(treatment, trim_fraction), "trim_fraction": trim_fraction, "permutation_iterations": iterations}, p_value=float(p_value), warnings=["采用可复现的置换版本检验实验组极端反应范围是否大于对照组。"])


def _runs_statistic(labels: np.ndarray) -> tuple[int, float, float, float]:
    n1, n2 = int(np.sum(labels == 0)), int(np.sum(labels == 1))
    if n1 == 0 or n2 == 0:
        raise ValueError("游程检验需要两个非空类别")
    runs = int(1 + np.sum(labels[1:] != labels[:-1]))
    mean = 1.0 + 2.0 * n1 * n2 / (n1 + n2)
    variance = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2) / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    z = (runs - mean) / np.sqrt(variance) if variance > 0 else 0.0
    return runs, mean, variance, float(z)


def run_wald_wolfowitz(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    a, b = numeric(data, "group_a"), numeric(data, "group_b")
    values = np.concatenate([a, b])
    labels = np.concatenate([np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
    order = np.argsort(values, kind="mergesort")
    runs, mean, variance, z = _runs_statistic(labels[order])
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    warnings = ["数据存在并列值；游程顺序使用稳定排序，精确 P 值可能受并列处理影响。"] if len(np.unique(values)) < len(values) else []
    return MethodResult("NONPAR006", "run_wald_wolfowitz", {"runs": runs, "expected_runs": mean, "variance": variance, "z": z, "n_a": len(a), "n_b": len(b)}, p_value=p_value, warnings=warnings)


def run_friedman(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    groups = [np.asarray(group, dtype=float) for group in data["groups"]]
    if len(groups) < 3 or len({len(group) for group in groups}) != 1 or len(groups[0]) < 2:
        raise ValueError("Friedman 检验至少需要三个等长的配对/区组样本")
    result = stats.friedmanchisquare(*groups)
    n, k = len(groups[0]), len(groups)
    kendalls_w = float(result.statistic / (n * (k - 1)))
    return MethodResult("NONPAR007", "run_friedman", {"chi2": float(result.statistic), "blocks": n, "groups": k}, p_value=float(result.pvalue), effect_size={"name": "Kendall's W", "value": kendalls_w})