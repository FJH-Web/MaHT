"""
兼容性模块：用于向后兼容的 HypothesisState 协议模型

本模块重新导出 protocol.statistical_object 中的核心类，
以保持与旧版本代码的兼容性。

导出的类：
- Evidence: 证据类
- HypothesisState: 假设状态类
- HypothesisStatus: 假设状态枚举
"""

from statagent.protocol.statistical_object import Evidence, HypothesisState, HypothesisStatus

__all__ = ["Evidence", "HypothesisState", "HypothesisStatus"]