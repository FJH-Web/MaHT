"""
假设管理模块

本模块提供假设图和假设状态管理功能。
核心类：
- HypothesisState: 假设状态数据结构
- HypothesisStatus: 假设状态枚举
- HypothesisGraph: 假设依赖图

设计原则：
- 假设之间可能存在依赖关系，使用图结构管理
- 假设状态包括待检验、检验中、已通过、已拒绝等
"""

from statagent.hypothesis.graph import HypothesisGraph
from statagent.protocol.statistical_object import HypothesisState, HypothesisStatus

__all__ = ["HypothesisGraph", "HypothesisState", "HypothesisStatus"]