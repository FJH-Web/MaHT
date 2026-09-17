"""
非参数检验方法模块

本模块包含非参数假设检验方法：
- run_mann_whitney: Mann-Whitney U 检验
- run_wilcoxon: 配对 Wilcoxon 符号秩检验
- run_wilcoxon_one_sample: 单样本 Wilcoxon 符号秩检验
- run_kruskal_wallis: Kruskal-Wallis 秩和检验
- run_moses_extreme_reactions: Moses 极限反应检验
- run_wald_wolfowitz: Wald-Wolfowitz 游程检验
- run_friedman: Friedman 秩和检验
"""

from statagent.statistical_library.nonparametric.methods import (
    run_friedman,
    run_kruskal_wallis,
    run_mann_whitney,
    run_moses_extreme_reactions,
    run_wald_wolfowitz,
    run_wilcoxon,
    run_wilcoxon_one_sample,
)

__all__ = [
    "run_friedman", "run_kruskal_wallis", "run_mann_whitney",
    "run_moses_extreme_reactions", "run_wald_wolfowitz", "run_wilcoxon",
    "run_wilcoxon_one_sample",
]