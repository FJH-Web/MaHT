"""
假设管理器模块

HypothesisManager 负责管理假设图中的假设生命周期。
主要功能：
- 将新假设添加到假设图
- 更新假设状态
- 同步假设的后验概率和证据
"""

from __future__ import annotations

from dataclasses import dataclass

from statagent.agents.base import StatisticalAgent
from statagent.hypothesis.graph import HypothesisGraph
from statagent.protocol.statistical_object import HypothesisStatus, StatisticalObject


@dataclass(slots=True)
class HypothesisManager(StatisticalAgent):
    """假设管理器：管理假设图中的假设生命周期"""
    
    graph: HypothesisGraph

    def process(self, message: StatisticalObject) -> StatisticalObject:
        """处理假设消息，更新假设图"""
        try:
            current = self.graph.get(message.hypothesis.id)
        except KeyError:
            self.graph.add_node(message.hypothesis)
        else:
            message.hypothesis = current
        if message.hypothesis.status == HypothesisStatus.PROPOSED:
            message.hypothesis.status = HypothesisStatus.READY
        return message

    def synchronize(self, message: StatisticalObject) -> None:
        """同步假设状态、证据到假设图"""
        hypothesis = message.hypothesis
        self.graph.update_status(hypothesis.id, hypothesis.status)
        for evidence in message.evidence:
            if evidence.timestamp not in {item.timestamp for item in self.graph.get(hypothesis.id).evidence}:
                self.graph.update_evidence(hypothesis.id, evidence)