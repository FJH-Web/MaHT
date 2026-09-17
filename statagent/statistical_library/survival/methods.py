"""
生存分析假设检验方法实现模块

本模块实现生存分析假设检验方法：
- run_logrank: Log-rank 生存曲线检验
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from statagent.protocol.statistical_object import MethodResult
from statagent.statistical_library.common import numeric, require_same_length


def run_logrank(data: dict[str, Any], parameters: dict[str, Any]) -> MethodResult:
    """Log-rank 生存曲线检验"""
    time, event = numeric(data, "time"), np.asarray(data["event"], dtype=int)
    group = np.asarray(data["group"])
    require_same_length(time, event, group)
    if not set(np.unique(event)).issubset({0, 1}):
        raise ValueError("event 必须仅包含 0（删失）与 1（事件）")
    levels = np.unique(group)
    if len(levels) != 2:
        raise ValueError("当前 Log-rank 实现要求恰好两个组")
    observed = expected = variance = 0.0
    for current in np.unique(time[event == 1]):
        at_risk = time >= current
        event_now = (time == current) & (event == 1)
        n = at_risk.sum()
        n1 = (at_risk & (group == levels[0])).sum()
        d = event_now.sum()
        d1 = (event_now & (group == levels[0])).sum()
        if n <= 1:
            continue
        observed += d1
        expected += d * n1 / n
        variance += n1 * (n - n1) * d * (n - d) / (n * n * (n - 1))
    statistic = (observed - expected) ** 2 / variance if variance > 0 else 0.0
    return MethodResult("SURV002", "run_logrank", {"chi2": float(statistic), "observed": float(observed), "expected": float(expected), "groups": levels.tolist()}, p_value=float(stats.chi2.sf(statistic, 1)))