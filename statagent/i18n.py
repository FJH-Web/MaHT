"""
StatAgent 中文显示层

本模块提供中文本地化支持，包括：
- 方法名称中文化：将统计方法的技术名称转换为中文描述
- 决策结果中文化：将统计决策结果转换为中文表达
- 字段名称中文化：将统计结果字段名转换为中文标签

内部稳定标识符保持不变，仅用于显示层翻译。
"""

from __future__ import annotations

from typing import Any, Mapping


METHOD_NAMES_ZH = {
    "DIST001": "Shapiro–Wilk 正态性检验", "DIST002": "Energy 多元正态性检验",
    "DIST003": "Henze–Zirkler 多元正态性检验", "DIST004": "Royston 多元正态性检验",
    "DIST005": "Kolmogorov–Smirnov 分布检验", "DIST006": "卡方分布拟合优度检验", "DIST007": "二项分布精确检验",
    "VAR001": "两样本方差 F 检验", "VAR002": "Bartlett 方差齐性检验", "VAR003": "Levene 方差齐性检验",
    "PAR000": "单样本 t 检验",
    "PAR001": "独立样本 t 检验",
    "PAR002": "配对样本 t 检验",
    "PAR003": "单因素方差分析",
    "PAR004": "单样本 Z 检验", "PAR005": "双样本 Z 检验", "PAR006": "均值的成对多重 t 比较",
    "PAR007": "Dunnett t 检验", "PAR008": "Tukey HSD 检验",
    "NONPAR001": "Mann–Whitney U / Wilcoxon 秩和检验",
    "NONPAR002": "Wilcoxon 符号秩检验",
    "NONPAR003": "Kruskal–Wallis 检验",
    "NONPAR004": "单样本 Wilcoxon 符号秩检验", "NONPAR005": "Moses 极限反应检验",
    "NONPAR006": "Wald–Wolfowitz 两样本游程检验", "NONPAR007": "Friedman 秩和检验",
    "CAT001": "卡方独立性检验",
    "CAT002": "Fisher 精确检验",
    "CAT003": "Cochran–Mantel–Haenszel 检验",
    "COR001": "双变量相关性检验", "COR002": "多变量两两相关性检验", "RUN001": "随机性游程检验",
    "SURV002": "Log-rank 检验",
}

DECISION_ZH = {
    "pending": "待决策",
    "support": "✓ 显著（支持研究假设）",
    "reject": "✗ 不显著（不支持研究假设）",
    "inconclusive": "⚠ 证据不足",
}

FIELD_NAMES_ZH = {
    "mean": "均值", "median": "中位数", "sd": "标准差", "n": "样本量", "ddof": "自由度修正",
    "count": "频数", "proportion": "构成比", "t": "t 统计量", "f": "F 统计量", "u": "U 统计量",
    "w": "W 统计量", "h": "H 统计量", "chi2": "卡方统计量", "dof": "自由度", "p_value": "P 值",
    "mean_a": "A 组均值", "mean_b": "B 组均值", "n_a": "A 组样本量", "n_b": "B 组样本量",
    "mean_difference": "平均差值", "groups": "组数", "expected": "期望频数", "odds_ratio": "优势比",
    "curve": "曲线数据", "median_survival": "中位生存时间", "events": "事件数", "observed": "观察值",
    "coefficients": "系数估计", "log_likelihood": "对数似然", "hazard_ratio": "风险比", "coef": "回归系数",
    "estimate": "估计值", "std_error": "标准误", "ci": "置信区间", "ci_odds_ratio": "优势比置信区间",
    "r_squared": "决定系数 R²", "adjusted_r_squared": "调整 R²", "pseudo_r_squared": "伪 R²",
    "components": "主成分载荷", "scores": "主成分得分", "explained_variance": "解释方差",
    "explained_variance_ratio": "解释方差比例", "singular_values": "奇异值", "feature_importance": "变量重要性",
    "predictions": "预测值", "accuracy": "准确率", "aic": "AIC", "bic": "BIC", "metric": "评价指标",
    "method_card_version": "方法卡版本", "elapsed_seconds": "运行耗时（秒）", "parameters": "运行参数",
    "w": "W 统计量", "energy": "能量统计量", "hz": "Henze–Zirkler 统计量", "h": "H 统计量", "d": "D 统计量",
    "successes": "成功次数", "trials": "试验次数", "observed_proportion": "观察比例", "null_probability": "原假设概率",
    "population_sd": "已知总体标准差", "population_sd_a": "A 组总体标准差", "population_sd_b": "B 组总体标准差",
    "null_difference": "原假设均值差", "standard_error": "标准误", "comparisons": "多重比较结果",
    "raw_p_value": "原始 P 值", "adjusted_p_value": "校正后 P 值", "adjustment": "多重性校正",
    "range_difference": "截尾范围差", "control_trimmed_range": "对照组截尾范围", "treatment_trimmed_range": "实验组截尾范围",
    "runs": "游程数", "expected_runs": "期望游程数", "variance": "方差", "z": "Z 统计量",
    "correlation": "相关系数", "method": "相关方法", "strata": "分层数", "pooled_odds_ratio": "合并优势比",
    "homogeneity_chi2": "齐性检验卡方统计量", "homogeneity_p_value": "齐性检验 P 值", "blocks": "区组数",
    "kendalls_w": "Kendall W", "dimensions": "变量维度", "bootstrap_iterations": "自助法次数",
    "group_a": "A 组", "group_b": "B 组", "before": "干预前", "after": "干预后", "group": "分组变量",
    "values": "分析变量", "table": "列联表", "time": "时间变量", "event": "事件变量", "x": "自变量矩阵", "y": "因变量",
    "kind": "映射类型", "column": "数据列", "columns": "数据列集合", "value_column": "数值列", "group_column": "分组列",
    "row_column": "行变量", "column_column": "列变量", "equal_var": "是否假定方差齐性", "task": "任务类型",
    "rows": "样本行数", "missing_total": "缺失值总数", "dtype": "数据类型", "non_missing": "非缺失数",
    "missing": "缺失数", "unique": "唯一值数", "levels": "类别水平", "min": "最小值", "max": "最大值",
    "name": "名称", "passed": "是否通过", "statistic": "统计量", "status": "状态", "reason": "原因",
    "warnings": "警告", "diagnostics": "诊断信息", "random_state": "随机种子", "feature_names": "变量名称",
}

VALUE_NAMES_ZH = {
    "numeric": "数值型", "categorical": "分类型", "column": "单列", "columns": "多列",
    "matrix": "矩阵", "grouped": "按组拆分", "contingency": "列联表", "classification": "分类",
    "regression": "回归", "not_evaluated": "未评估", "two-sided": "双侧", "greater": "大于", "less": "小于",
}


def method_name_zh(method: Mapping[str, Any] | None) -> str:
    if not method:
        return "未选择方法"
    return METHOD_NAMES_ZH.get(str(method.get("method_id", "")), str(method.get("name", "未知方法")))


def decision_zh(value: str) -> str:
    return DECISION_ZH.get(value, value)


def localize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {FIELD_NAMES_ZH.get(str(key), str(key)): localize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [localize_value(item) for item in value]
    if isinstance(value, str):
        return VALUE_NAMES_ZH.get(value, value)
    return value