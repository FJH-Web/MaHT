"""
列联表检验方法模块

本模块包含列联表检验方法：
- run_chi_square: 卡方独立性检验
- run_fisher_exact: Fisher 精确检验
- run_cochran_mantel_haenszel: Cochran-Mantel-Haenszel 检验
"""

from statagent.statistical_library.categorical.methods import (
    run_chi_square,
    run_cochran_mantel_haenszel,
    run_fisher_exact,
)

__all__ = ["run_chi_square", "run_cochran_mantel_haenszel", "run_fisher_exact"]