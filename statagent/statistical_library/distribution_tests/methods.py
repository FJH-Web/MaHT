"""
分布假设检验方法实现模块

本模块实现分布假设检验方法：
- run_shapiro_wilk: Shapiro-Wilk 正态性检验
- run_energy_normality: Energy 多元正态性检验
- run_henze_zirkler: Henze-Zirkler 多元正态性检验
- run_royston: Royston 多元正态性检验
- run_kolmogorov_smirnov: Kolmogorov-Smirnov 分布检验
- run_chi_square_gof: 卡方分布拟合优度检验
- run_anderson_darling: Anderson-Darling 正态性检验
- run_binomial_test: 二项分布精确检验
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform

from statagent.protocol.statistical_object import MethodResult
from statagent.statistical_library.common import matrix, numeric


def _bootstrap_p_value(observed: float, generator: Callable[[], float], iterations: int) -> float:
    """计算 Bootstrap p 值"""
    simulated = np.asarray([generator() for _ in range(iterations)], dtype=float)
    return float((1 + np.sum(simulated >= observed)) / (iterations + 1))


def _standardize(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= x.shape[1]:
        raise ValueError("多元正态性检验要求样本量大于变量数")
    centered = x - x.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(np.atleast_2d(covariance))
    if np.any(eigenvalues <= np.finfo(float).eps):
        raise ValueError("协方差矩阵奇异，无法进行多元正态性检验")
    return centered @ (eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T)


def _energy_statistic(x: np.ndarray, reference: np.ndarray) -> float:
    cross = 2.0 * float(cdist(x, reference).mean())
    within_x = float(squareform(pdist(x)).mean())
    within_ref = float(squareform(pdist(reference)).mean())
    return len(x) * (cross - within_x - within_ref)


def _hz_statistic(x: np.ndarray) -> float:
    z = _standardize(x)
    n, p = z.shape
    beta = (1.0 / np.sqrt(2.0)) * (((2.0 * p + 1.0) * n / 4.0) ** (1.0 / (p + 4.0)))
    pairwise_squared = squareform(pdist(z, metric="sqeuclidean"))
    radius_squared = np.sum(z * z, axis=1)
    first = np.exp(-(beta**2) * pairwise_squared / 2.0).sum() / n
    second = 2.0 * (1.0 + beta**2) ** (-p / 2.0) * np.exp(
        -(beta**2) * radius_squared / (2.0 * (1.0 + beta**2))
    ).sum()
    third = n * (1.0 + 2.0 * beta**2) ** (-p / 2.0)
    return float(first - second + third)


def _royston_statistic(x: np.ndarray) -> tuple[float, list[float]]:
    p_values = [float(stats.shapiro(x[:, index]).pvalue) for index in range(x.shape[1])]
    clipped = np.clip(p_values, np.finfo(float).tiny, 1.0)
    return float(-2.0 * np.log(clipped).sum()), p_values


def run_shapiro_wilk(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    values = numeric(data, "values", minimum=3)
    result = stats.shapiro(values[:5000])
    warnings = ["Shapiro–Wilk 在样本量超过 5000 时仅使用前 5000 个有限观测。"] if len(values) > 5000 else []
    return MethodResult("DIST001", "run_shapiro_wilk", {"w": float(result.statistic), "n": len(values)}, p_value=float(result.pvalue), warnings=warnings)


def run_energy_normality(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    x = matrix(data, "x")
    z = _standardize(x)
    iterations = int(parameters.get("bootstrap_iterations", 199))
    rng = np.random.default_rng(int(parameters.get("random_state", 2026)))
    reference = rng.normal(size=(max(500, 5 * len(z)), z.shape[1]))
    observed = _energy_statistic(z, reference)
    p_value = _bootstrap_p_value(observed, lambda: _energy_statistic(_standardize(rng.normal(size=z.shape)), reference), iterations)
    return MethodResult("DIST002", "run_energy_normality", {"energy": observed, "n": len(z), "dimensions": z.shape[1], "bootstrap_iterations": iterations}, p_value=p_value, warnings=["P 值采用固定随机种子的参数自助法估计。"])


def run_henze_zirkler(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    x = matrix(data, "x")
    observed = _hz_statistic(x)
    iterations = int(parameters.get("bootstrap_iterations", 199))
    rng = np.random.default_rng(int(parameters.get("random_state", 2026)))
    p_value = _bootstrap_p_value(observed, lambda: _hz_statistic(rng.normal(size=x.shape)), iterations)
    return MethodResult("DIST003", "run_henze_zirkler", {"hz": observed, "n": x.shape[0], "dimensions": x.shape[1], "bootstrap_iterations": iterations}, p_value=p_value, warnings=["Henze–Zirkler 统计量按原始定义计算；P 值采用参数自助法校准。"])


def run_royston(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    x = matrix(data, "x")
    if not 3 <= len(x) <= 5000:
        raise ValueError("Royston 检验要求样本量在 3 到 5000 之间")
    observed, marginal = _royston_statistic(x)
    iterations = int(parameters.get("bootstrap_iterations", 199))
    rng = np.random.default_rng(int(parameters.get("random_state", 2026)))
    correlation = np.corrcoef(x, rowvar=False)
    p_value = _bootstrap_p_value(observed, lambda: _royston_statistic(rng.multivariate_normal(np.zeros(x.shape[1]), correlation, size=len(x)))[0], iterations)
    return MethodResult("DIST004", "run_royston", {"h": observed, "marginal_shapiro_p_values": marginal, "n": len(x), "dimensions": x.shape[1], "bootstrap_iterations": iterations}, p_value=p_value, warnings=["该实现使用 Royston 的边际 Shapiro–Wilk 组合思想，并以样本相关结构进行参数自助校准。"])


def run_kolmogorov_smirnov(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    values = numeric(data, "values", minimum=2)
    distributions = {"norm": stats.norm, "uniform": stats.uniform, "expon": stats.expon, "logistic": stats.logistic}
    distribution = str(parameters.get("distribution", "norm"))
    if distribution not in distributions:
        raise ValueError(f"不支持的连续分布：{distribution}")
    args = tuple(parameters.get("args", ()))
    result = stats.kstest(values, distributions[distribution].cdf, args=args, alternative=parameters.get("alternative", "two-sided"))
    return MethodResult("DIST005", "run_kolmogorov_smirnov", {"d": float(result.statistic), "n": len(values), "distribution": distribution, "distribution_args": list(args)}, p_value=float(result.pvalue), warnings=["分布参数必须事先给定；若由同一数据估计参数，应改用相应的校正拟合优度检验。"])


def run_anderson_darling(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    """Anderson-Darling 正态性检验"""
    values = numeric(data, "values", minimum=5)
    if len(values) > 5000:
        values = values[:5000]
        warnings = ["Anderson–Darling 在样本量超过 5000 时仅使用前 5000 个有限观测。"]
    else:
        warnings = []
    result = stats.anderson(values, dist="norm")
    statistic = float(result.statistic)
    critical_values = result.critical_values.tolist()
    significance_levels = result.significance_level.tolist()
    p_value = _anderson_p_value(statistic, critical_values, significance_levels)
    return MethodResult("DIST008", "run_anderson_darling", {"a2": statistic, "n": len(values), "critical_values": critical_values, "significance_levels": significance_levels}, p_value=p_value, warnings=warnings)


def _anderson_p_value(statistic: float, critical_values: list[float], significance_levels: list[float]) -> float:
    """根据 Anderson-Darling 统计量计算 p 值"""
    for cv, sl in zip(critical_values, significance_levels):
        if statistic <= cv:
            return float(sl / 100)
    return 0.001


def run_chi_square_gof(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    values = np.asarray(data["values"], dtype=object)
    values = values[pd_not_missing(values)]
    if not len(values):
        raise ValueError("拟合优度检验至少需要一个非缺失观测")
    levels, observed = np.unique(values.astype(str), return_counts=True)
    probabilities = parameters.get("expected_probabilities")
    if probabilities is None:
        expected = np.repeat(len(values) / len(levels), len(levels))
    elif isinstance(probabilities, dict):
        expected = np.asarray([float(probabilities[str(level)]) * len(values) for level in levels])
    else:
        probabilities = np.asarray(probabilities, dtype=float)
        if len(probabilities) != len(levels) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("expected_probabilities 必须与类别数一致且概率和为 1")
        expected = probabilities * len(values)
    result = stats.chisquare(observed, expected, ddof=int(parameters.get("ddof", 0)))
    return MethodResult("DIST006", "run_chi_square_gof", {"chi2": float(result.statistic), "levels": levels.tolist(), "observed": observed.tolist(), "expected": expected.tolist()}, p_value=float(result.pvalue), assumption_checks=[{"name": "expected_cell_count", "minimum": float(expected.min()), "passed": bool((expected >= 5).all())}])


def pd_not_missing(values: np.ndarray) -> np.ndarray:
    return np.asarray([item is not None and str(item).lower() != "nan" for item in values], dtype=bool)


def run_binomial_test(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    values = np.asarray(data["values"], dtype=object)
    values = values[pd_not_missing(values)]
    success_value = str(parameters.get("success_value", "1"))
    successes = int(np.sum(values.astype(str) == success_value))
    probability = float(parameters.get("probability", 0.5))
    result = stats.binomtest(successes, len(values), p=probability, alternative=parameters.get("alternative", "two-sided"))
    ci = result.proportion_ci(confidence_level=float(parameters.get("confidence_level", 0.95)))
    return MethodResult("DIST007", "run_binomial_test", {"successes": successes, "trials": len(values), "observed_proportion": successes / len(values), "null_probability": probability}, p_value=float(result.pvalue), confidence_interval={"low": float(ci.low), "high": float(ci.high)})