"""
相关性与随机性检验方法模块

本模块包含相关性与随机性检验方法：
- run_correlation_test: 双变量相关性检验
- run_correlation_matrix_tests: 多变量两两相关性检验
- run_randomness_runs_test: 随机性游程检验
"""

from statagent.statistical_library.association_tests.methods import (
    run_correlation_matrix_tests,
    run_correlation_test,
    run_randomness_runs_test,
)

__all__ = ["run_correlation_matrix_tests", "run_correlation_test", "run_randomness_runs_test"]