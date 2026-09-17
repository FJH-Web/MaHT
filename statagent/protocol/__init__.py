"""
统计协议模块

本模块定义统计推断的数据模型和决策协议。
核心类：
- StatisticalObject: 统计对象，包含假设、方法和结果
- HypothesisState: 假设状态
- HypothesisStatus: 假设状态枚举
- Evidence: 证据，包含统计推断结果
- Decision: 统计决策枚举（接受/拒绝/需要更多数据）
- DecisionPolicy: 决策策略抽象基类
- ThresholdDecisionPolicy: 阈值决策策略
- InterpretationBundle: 解释包，包含结果的中文解读
- MethodResult: 方法执行结果
- VisualizationArtifact: 可视化产物

设计原则：
- 这些数据结构在 SDP 中作为 Agent 之间的受限通信介质
- 使用 frozen dataclass 确保不可变性
"""

from statagent.protocol.decision_protocol import DecisionPolicy, ThresholdDecisionPolicy
from statagent.protocol.statistical_object import (
    Decision,
    Evidence,
    HypothesisState,
    HypothesisStatus,
    MethodResult,
    InterpretationBundle,
    StatisticalObject,
    VisualizationArtifact,
)

__all__ = [
    "Decision",
    "DecisionPolicy",
    "Evidence",
    "HypothesisState",
    "HypothesisStatus",
    "MethodResult",
    "InterpretationBundle",
    "StatisticalObject",
    "VisualizationArtifact",
    "ThresholdDecisionPolicy",
]