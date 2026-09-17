"""
参数检验方法模块

本模块包含参数假设检验方法：
- run_ttest_one_sample: 单样本 t 检验
- run_ttest_independent: 独立样本 t 检验
- run_ttest_paired: 配对样本 t 检验
- run_anova_oneway: 单因素方差分析
- run_ztest_one_sample: 单样本 Z 检验
- run_ztest_two_sample: 双样本 Z 检验
- run_pairwise_ttests: 均值的成对多重 t 比较
- run_dunnett: Dunnett t 检验
- run_tukey_hsd: Tukey HSD 检验
"""

from statagent.statistical_library.parametric.methods import (
    run_anova_oneway,
    run_dunnett,
    run_pairwise_ttests,
    run_ttest_independent,
    run_ttest_one_sample,
    run_ttest_paired,
    run_tukey_hsd,
    run_ztest_one_sample,
    run_ztest_two_sample,
)

__all__ = [
    "run_anova_oneway", "run_dunnett", "run_pairwise_ttests",
    "run_ttest_independent", "run_ttest_one_sample", "run_ttest_paired",
    "run_tukey_hsd", "run_ztest_one_sample", "run_ztest_two_sample",
]