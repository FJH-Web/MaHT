"""
决策 Agent 模块

DecisionAgent 负责根据统计结果做出决策。
主要功能：
- 应用决策策略将统计结果转换为决策
- 更新假设状态

决策结果包括：支持、拒绝、证据不足、待决策。
"""

from __future__ import annotations

from dataclasses import dataclass

from statagent.agents.base import StatisticalAgent
from statagent.protocol.decision_protocol import DecisionPolicy
from statagent.protocol.statistical_object import Decision, HypothesisStatus, StatisticalObject


@dataclass(slots=True)
class DecisionAgent(StatisticalAgent):
    """决策 Agent：根据统计结果做出决策"""
    
    policy: DecisionPolicy

    def process(self, message: StatisticalObject) -> StatisticalObject:
        """应用决策策略并更新假设状态"""
        message = self.policy.decide(message)
        status_map = {
            Decision.SUPPORT: HypothesisStatus.SUPPORTED,
            Decision.REJECT: HypothesisStatus.REJECTED,
            Decision.INCONCLUSIVE: HypothesisStatus.INCONCLUSIVE,
            Decision.PENDING: HypothesisStatus.TESTING,
        }
        message.hypothesis.status = status_map[message.decision]
        return message