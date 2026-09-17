from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from statagent.protocol.statistical_object import MethodResult
from statagent.statistical_library.common import normality_check, numeric


def _p_from_z(statistic: float, alternative: str) -> float:
    if alternative == "greater":
        return float(stats.norm.sf(statistic))
    if alternative == "less":
        return float(stats.norm.cdf(statistic))
    return float(2.0 * stats.norm.sf(abs(statistic)))


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def run_ttest_one_sample(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    values = numeric(data, "values", minimum=2)
    null_mean = float(parameters.get("population_mean", 0.0))
    result = stats.ttest_1samp(values, popmean=null_mean, alternative=parameters.get("alternative", "two-sided"))
    sd = float(values.std(ddof=1))
    effect = (float(values.mean()) - null_mean) / sd if sd else 0.0
    return MethodResult("PAR000", "run_ttest_one_sample", {"t": float(result.statistic), "sample_mean": float(values.mean()), "null_mean": null_mean, "n": len(values)}, p_value=float(result.pvalue), effect_size={"name": "Cohen's d", "value": effect}, assumption_checks=[normality_check(values)])


def run_ttest_independent(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    a = numeric(data, "group_a", minimum=2)
    b = numeric(data, "group_b", minimum=2)
    equal_var = bool(parameters.get("equal_var", False))
    statistic, p_value = stats.ttest_ind(a, b, equal_var=equal_var)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    d = (a.mean() - b.mean()) / pooled if pooled else 0.0
    levene = stats.levene(a, b)
    mean_diff = float(a.mean() - b.mean())
    ci_level = parameters.get("confidence_level", 0.95)
    alpha = 1 - ci_level
    if equal_var:
        se = float(pooled * np.sqrt(1/len(a) + 1/len(b)))
        df = len(a) + len(b) - 2
    else:
        se = float(np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b)))
        num = (a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))**2
        den = (a.var(ddof=1)/len(a))**2/(len(a)-1) + (b.var(ddof=1)/len(b))**2/(len(b)-1)
        df = num / den if den else 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se
    normality_a = normality_check(a)
    normality_b = normality_check(b)
    return MethodResult(
        "PAR001", "run_ttest_independent",
        {"t": float(statistic), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "n_a": len(a), "n_b": len(b)},
        p_value=float(p_value), effect_size={"name": "Cohen's d", "value": float(d)},
        confidence_interval={"level": ci_level, "lower": float(ci_lower), "upper": float(ci_upper), "parameter": "mean_difference"},
        assumption_checks=[normality_a, normality_b, {"name": "Levene", "p_value": float(levene.pvalue), "passed": bool(levene.pvalue >= 0.05), "type": "variance"}],
    )


def run_ttest_paired(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    before = numeric(data, "before", minimum=2)
    after = numeric(data, "after", minimum=2)
    if len(before) != len(after):
        raise ValueError("before and after must have equal lengths")
    statistic, p_value = stats.ttest_rel(before, after)
    difference = after - before
    d = difference.mean() / difference.std(ddof=1) if difference.std(ddof=1) else 0.0
    ci_level = parameters.get("confidence_level", 0.95)
    alpha = 1 - ci_level
    se = difference.std(ddof=1) / np.sqrt(len(difference))
    df = len(difference) - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = difference.mean() - t_crit * se
    ci_upper = difference.mean() + t_crit * se
    normality = normality_check(difference)
    normality["type"] = "normality"
    return MethodResult(
        "PAR002", "run_ttest_paired", {"t": float(statistic), "mean_difference": float(difference.mean()), "n": len(difference)},
        p_value=float(p_value), effect_size={"name": "Cohen's dz", "value": float(d)},
        confidence_interval={"level": ci_level, "lower": float(ci_lower), "upper": float(ci_upper), "parameter": "mean_difference"},
        assumption_checks=[normality],
    )


def run_anova_oneway(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    groups = [np.asarray(group, dtype=float) for group in data["groups"]]
    if len(groups) < 2 or any(len(group) < 2 for group in groups):
        raise ValueError("ANOVA needs at least two groups with two observations each")
    statistic, p_value = stats.f_oneway(*groups)
    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    ss_total = ((all_values - grand_mean) ** 2).sum()
    eta_squared = ss_between / ss_total if ss_total else 0.0
    levene = stats.levene(*groups)
    normality_checks = []
    for i, g in enumerate(groups):
        nc = normality_check(g)
        nc["type"] = "normality"
        normality_checks.append(nc)
    return MethodResult(
        "PAR003", "run_anova_oneway", {"f": float(statistic), "groups": len(groups), "n": len(all_values), "eta_squared": float(eta_squared)},
        p_value=float(p_value), effect_size={"name": "eta_squared", "value": float(eta_squared)},
        assumption_checks=[*normality_checks, {"name": "Levene", "p_value": float(levene.pvalue), "passed": bool(levene.pvalue >= 0.05), "type": "variance"}],
    )


def run_ztest_one_sample(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    values = numeric(data, "values", minimum=2)
    if "population_sd" not in parameters:
        raise ValueError("单样本 Z 检验必须提供已知总体标准差 population_sd")
    population_sd = float(parameters["population_sd"])
    if population_sd <= 0:
        raise ValueError("population_sd 必须大于 0")
    null_mean = float(parameters.get("population_mean", 0.0))
    standard_error = population_sd / np.sqrt(len(values))
    statistic = (float(values.mean()) - null_mean) / standard_error
    return MethodResult("PAR004", "run_ztest_one_sample", {"z": statistic, "sample_mean": float(values.mean()), "null_mean": null_mean, "population_sd": population_sd, "standard_error": standard_error, "n": len(values)}, p_value=_p_from_z(statistic, parameters.get("alternative", "two-sided")))


def run_ztest_two_sample(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    a, b = numeric(data, "group_a", minimum=2), numeric(data, "group_b", minimum=2)
    if "population_sd_a" not in parameters or "population_sd_b" not in parameters:
        raise ValueError("双样本 Z 检验必须提供 population_sd_a 和 population_sd_b")
    sd_a, sd_b = float(parameters["population_sd_a"]), float(parameters["population_sd_b"])
    if sd_a <= 0 or sd_b <= 0:
        raise ValueError("已知总体标准差必须大于 0")
    null_difference = float(parameters.get("null_difference", 0.0))
    standard_error = np.sqrt(sd_a**2 / len(a) + sd_b**2 / len(b))
    observed_difference = float(a.mean() - b.mean())
    statistic = (observed_difference - null_difference) / standard_error
    return MethodResult("PAR005", "run_ztest_two_sample", {"z": statistic, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "observed_difference": observed_difference, "null_difference": null_difference, "standard_error": float(standard_error), "n_a": len(a), "n_b": len(b)}, p_value=_p_from_z(statistic, parameters.get("alternative", "two-sided")))


def run_pairwise_ttests(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    groups = [np.asarray(group, dtype=float) for group in data["groups"]]
    if len(groups) < 3 or any(len(group) < 2 for group in groups):
        raise ValueError("均值多重比较至少需要三个组，且每组至少两个观测")
    labels = list(parameters.get("group_labels", [f"组{i + 1}" for i in range(len(groups))]))
    comparisons, raw_p = [], []
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            result = stats.ttest_ind(groups[left], groups[right], equal_var=bool(parameters.get("equal_var", False)))
            comparisons.append({"group_a": labels[left], "group_b": labels[right], "t": float(result.statistic), "mean_difference": float(groups[left].mean() - groups[right].mean()), "raw_p_value": float(result.pvalue)})
            raw_p.append(float(result.pvalue))
    adjusted = _holm(raw_p)
    for comparison, value in zip(comparisons, adjusted, strict=True):
        comparison["adjusted_p_value"] = value
    return MethodResult("PAR006", "run_pairwise_ttests", {"comparisons": comparisons, "adjustment": "Holm", "groups": len(groups)}, p_value=float(min(adjusted)))


def run_dunnett(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    groups = [np.asarray(group, dtype=float) for group in data["groups"]]
    if len(groups) < 2 or any(len(group) < 2 for group in groups):
        raise ValueError("Dunnett 检验至少需要一个对照组和一个实验组")
    control_index = int(parameters.get("control_index", 0))
    if not 0 <= control_index < len(groups):
        raise ValueError("control_index 超出组索引范围")
    control = groups[control_index]
    treatments = [group for index, group in enumerate(groups) if index != control_index]
    seed = int(parameters.get("random_state", 2026))
    try:
        result = stats.dunnett(*treatments, control=control, alternative=parameters.get("alternative", "two-sided"), rng=np.random.default_rng(seed))
    except TypeError:
        result = stats.dunnett(*treatments, control=control, alternative=parameters.get("alternative", "two-sided"), random_state=seed)
    statistics = np.atleast_1d(result.statistic).astype(float)
    p_values = np.atleast_1d(result.pvalue).astype(float)
    treatment_indices = [index for index in range(len(groups)) if index != control_index]
    comparisons = [{"control_index": control_index, "treatment_index": index, "statistic": float(statistic), "adjusted_p_value": float(p)} for index, statistic, p in zip(treatment_indices, statistics, p_values, strict=True)]
    return MethodResult("PAR007", "run_dunnett", {"comparisons": comparisons, "control_index": control_index, "groups": len(groups)}, p_value=float(p_values.min()))


def run_tukey_hsd(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    groups = [np.asarray(group, dtype=float) for group in data["groups"]]
    if len(groups) < 2 or any(len(group) < 2 for group in groups):
        raise ValueError("Tukey HSD 检验至少需要两个组")
    result = stats.tukey_hsd(*groups)
    ci = result.confidence_interval(confidence_level=float(parameters.get("confidence_level", 0.95)))
    comparisons = []
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            comparisons.append({"group_a_index": left, "group_b_index": right, "mean_difference": float(result.statistic[left, right]), "adjusted_p_value": float(result.pvalue[left, right]), "ci": [float(ci.low[left, right]), float(ci.high[left, right])]})
    return MethodResult("PAR008", "run_tukey_hsd", {"comparisons": comparisons, "groups": len(groups)}, p_value=float(min(item["adjusted_p_value"] for item in comparisons)))