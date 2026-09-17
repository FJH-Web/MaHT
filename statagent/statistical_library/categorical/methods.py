from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from scipy import stats

from statagent.protocol.statistical_object import MethodResult


def run_chi_square(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    table = np.asarray(data["table"], dtype=float)
    if table.ndim != 2 or (table < 0).any():
        raise ValueError("table must be a non-negative two-dimensional contingency table")
    statistic, p_value, dof, expected = stats.chi2_contingency(table, correction=bool(parameters.get("correction", True)))
    n = table.sum()
    phi = np.sqrt(statistic / (n * max(1, min(table.shape) - 1))) if n else 0.0
    return MethodResult("CAT001", "run_chi_square", {"chi2": float(statistic), "dof": int(dof), "expected": expected.tolist()}, p_value=float(p_value), effect_size={"name": "Cramer's V", "value": float(phi)}, assumption_checks=[{"name": "expected_cell_count", "minimum": float(expected.min()), "passed": bool((expected >= 5).all())}])


def run_fisher_exact(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    table = np.asarray(data["table"], dtype=int)
    if table.shape != (2, 2) or (table < 0).any():
        raise ValueError("Fisher exact test requires a non-negative 2x2 table")
    odds_ratio, p_value = stats.fisher_exact(table, alternative=parameters.get("alternative", "two-sided"))
    return MethodResult("CAT002", "run_fisher_exact", {"odds_ratio": float(odds_ratio)}, p_value=float(p_value), effect_size={"name": "odds_ratio", "value": float(odds_ratio)})


def run_cochran_mantel_haenszel(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    from statsmodels.stats.contingency_tables import StratifiedTable

    row, column, strata = np.asarray(data["row"]), np.asarray(data["column"]), np.asarray(data["strata"])
    if not (len(row) == len(column) == len(strata)):
        raise ValueError("row、column 与 strata 必须具有相同长度")
    row_levels, column_levels = np.unique(row), np.unique(column)
    if len(row_levels) != 2 or len(column_levels) != 2:
        raise ValueError("Cochran–Mantel–Haenszel 基线实现要求两个二分类变量")
    tables = []
    stratum_levels = np.unique(strata)
    for level in stratum_levels:
        mask = strata == level
        table = np.zeros((2, 2), dtype=float)
        for i, row_level in enumerate(row_levels):
            for j, column_level in enumerate(column_levels):
                table[i, j] = np.sum(mask & (row == row_level) & (column == column_level))
        tables.append(table)
    model = StratifiedTable(np.stack(tables, axis=2), shift_zeros=bool(parameters.get("shift_zeros", False)))
    result = model.test_null_odds(correction=bool(parameters.get("correction", False)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        homogeneity = model.test_equal_odds(adjust=bool(parameters.get("adjust_homogeneity", False)))
    homogeneity_p = float(homogeneity.pvalue)
    homogeneity_passed = bool(homogeneity_p >= 0.05) if np.isfinite(homogeneity_p) else None
    return MethodResult("CAT003", "run_cochran_mantel_haenszel", {"chi2": float(result.statistic), "strata": len(tables), "pooled_odds_ratio": float(model.oddsratio_pooled), "homogeneity_chi2": float(homogeneity.statistic), "homogeneity_p_value": homogeneity_p, "tables": [table.tolist() for table in tables]}, p_value=float(result.pvalue), effect_size={"name": "common_odds_ratio", "value": float(model.oddsratio_pooled)}, assumption_checks=[{"name": "homogeneous_odds_ratios", "p_value": homogeneity_p, "passed": homogeneity_passed}])
