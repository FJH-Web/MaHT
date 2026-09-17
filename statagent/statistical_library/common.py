"""
统计库通用工具模块

本模块提供统计方法中常用的工具函数，包括：
- numeric: 验证和提取数值数据
- matrix: 验证和提取矩阵数据
- json_value: 将 numpy 类型转换为 JSON 兼容类型
- normality_check: 正态性检验
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def numeric(data: dict[str, Any], key: str, *, minimum: int = 1) -> np.ndarray:
    """从数据字典中提取并验证数值数组
    
    参数：
        data: 数据字典
        key: 键名
        minimum: 最少需要的数据点数
        
    返回：
        numpy 数组
        
    抛出：
        ValueError: 数据无效时
    """
    values = np.asarray(data[key], dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{key} must be one-dimensional")
    values = values[np.isfinite(values)]
    if len(values) < minimum:
        raise ValueError(f"{key} requires at least {minimum} finite observations")
    return values


def matrix(data: dict[str, Any], key: str) -> np.ndarray:
    """从数据字典中提取并验证二维矩阵"""
    values = np.asarray(data[key], dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{key} must be a finite two-dimensional matrix")
    return values


def json_value(value: Any) -> Any:
    """将 numpy 类型转换为 JSON 兼容的 Python 类型"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def normality_check(values: np.ndarray) -> dict[str, Any]:
    """使用 Shapiro-Wilk 检验检查数据正态性

    参数：
        values: 数值数组

    返回：
        包含检验结果的字典
    """
    from scipy import stats

    if len(values) < 3:
        return {"name": "Shapiro-Wilk", "status": "not_evaluated", "reason": "n < 3", "type": "normality"}
    sample = values[:5000]
    statistic, p_value = stats.shapiro(sample)
    return {
        "name": "Shapiro-Wilk",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "passed": bool(p_value >= 0.05),
        "type": "normality",
    }


def independence_check(values: np.ndarray) -> dict[str, Any]:
    """使用 Durbin-Watson 检验检查观测的独立性（自相关）

    参数：
        values: 数值数组

    返回：
        包含检验结果的字典
    """
    from scipy import stats
    import numpy as np

    if len(values) < 4:
        return {"name": "Durbin-Watson", "status": "not_evaluated", "reason": "n < 4", "type": "independence"}

    diff = np.diff(values)
    dw = np.sum(diff[:-1]**2) / np.sum(diff**2) if np.sum(diff**2) > 0 else 2.0
    passed = bool(1.5 < dw < 2.5)
    return {
        "name": "Durbin-Watson",
        "statistic": float(dw),
        "passed": passed,
        "type": "independence",
    }


def missing_check(data: dict[str, Any]) -> dict[str, Any]:
    """检查数据中的缺失值

    参数：
        data: 数据字典

    返回：
        包含缺失值检查结果的字典
    """
    import numpy as np

    total_cells = 0
    missing_cells = 0

    for key, values in data.items():
        if isinstance(values, (list, np.ndarray)):
            arr = np.asarray(values)
            total_cells += arr.size
            missing_cells += np.isnan(arr).sum() if arr.dtype.kind == 'f' else 0

    missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
    passed = bool(missing_ratio < 0.05)

    return {
        "name": "Missing Values",
        "total": int(total_cells),
        "missing": int(missing_cells),
        "ratio": float(missing_ratio),
        "passed": passed,
        "type": "missing",
    }


def outlier_check_iqr(values: np.ndarray) -> dict[str, Any]:
    """使用 IQR 方法检查异常值

    参数：
        values: 数值数组

    返回：
        包含异常值检查结果的字典
    """
    import numpy as np

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = (values < lower) | (values > upper)
    outlier_count = outliers.sum()
    outlier_ratio = outlier_count / len(values) if len(values) > 0 else 0

    return {
        "name": "Outlier (IQR)",
        "outlier_count": int(outlier_count),
        "total": int(len(values)),
        "ratio": float(outlier_ratio),
        "passed": bool(outlier_ratio < 0.05),
        "type": "outlier",
    }


def influential_points_check(values: np.ndarray) -> dict[str, Any]:
    """使用 Z-score 方法检查强影响点

    参数：
        values: 数值数组

    返回：
        包含强影响点检查结果的字典
    """
    import numpy as np

    z_scores = np.abs((values - values.mean()) / values.std(ddof=1)) if values.std(ddof=1) > 0 else np.zeros_like(values)
    influential = z_scores > 3
    influential_count = influential.sum()
    influential_ratio = influential_count / len(values) if len(values) > 0 else 0

    return {
        "name": "Influential Points (|Z|>3)",
        "influential_count": int(influential_count),
        "total": int(len(values)),
        "ratio": float(influential_ratio),
        "passed": bool(influential_ratio < 0.02),
        "type": "influential",
    }


def require_same_length(*arrays: Iterable[Any]) -> None:
    lengths = {len(array) for array in arrays}  # type: ignore[arg-type]
    if len(lengths) != 1:
        raise ValueError("inputs must have equal lengths")