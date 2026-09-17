"""
分布假设检验方法模块

本模块包含分布假设检验方法：
- run_shapiro_wilk: Shapiro-Wilk 正态性检验
- run_energy_normality: Energy 多元正态性检验
- run_henze_zirkler: Henze-Zirkler 多元正态性检验
- run_royston: Royston 多元正态性检验
- run_kolmogorov_smirnov: Kolmogorov-Smirnov 分布检验
- run_chi_square_gof: 卡方分布拟合优度检验
- run_anderson_darling: Anderson-Darling 正态性检验
- run_binomial_test: 二项分布精确检验
"""

from statagent.statistical_library.distribution_tests.methods import (
    run_anderson_darling,
    run_binomial_test,
    run_chi_square_gof,
    run_energy_normality,
    run_henze_zirkler,
    run_kolmogorov_smirnov,
    run_royston,
    run_shapiro_wilk,
)

__all__ = [
    "run_anderson_darling", "run_binomial_test", "run_chi_square_gof", "run_energy_normality",
    "run_henze_zirkler", "run_kolmogorov_smirnov", "run_royston",
    "run_shapiro_wilk",
]