"""
主动假设调度器模块 (AHS)

本模块实现主动假设调度功能，根据效用函数选择下一个待检验的假设。

理论映射说明：
- AHS 最大化 U(H_i) = EIG(H_i) - Cost(H_i)
- EIG(H_i) 是期望信息增益，基于假设的后验概率和熵计算
- Cost(H_i) 是检验成本

设计原则：
- 抽象的调度器边界允许后续替换为 MARL（多智能体强化学习）实现
- 支持将 LLM 返回的中文成本标签转换为数值成本
"""

from __future__ import annotations

import heapq
import math
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from statagent.protocol.statistical_object import HypothesisState


@dataclass(frozen=True, slots=True)
class UtilityEstimate:
    """效用估计：信息增益和成本"""
    
    information_gain: float
    cost: float

    @property
    def utility(self) -> float:
        """效用 = 信息增益 - 成本"""
        return self.information_gain - self.cost


class Scheduler(ABC):
    """调度器抽象基类"""
    
    @abstractmethod
    def schedule(self, hypothesis: HypothesisState, estimate: UtilityEstimate | None = None) -> None:
        """将假设加入调度队列"""
        ...

    @abstractmethod
    def next_hypothesis(self) -> HypothesisState | None:
        """获取下一个待检验的假设"""
        ...

    @abstractmethod
    def update(self, hypothesis: HypothesisState) -> None:
        """更新假设状态"""
        ...


@dataclass(slots=True)
class ActiveHypothesisScheduler(Scheduler):
    """主动假设调度器：基于效用的假设调度"""
    
    _queue: list[tuple[float, int, str]] = field(default_factory=list)
    _states: dict[str, HypothesisState] = field(default_factory=dict)
    _counter: int = 0

    def expected_information_gain(self, hypothesis: HypothesisState) -> float:
        """计算期望信息增益
        
        基于假设的证据数量和不确定性进行简化计算
        """
        evidence_discount = 1.0 / (1.0 + len(hypothesis.evidence))
        return evidence_discount

    @staticmethod
    def _coerce_cost(value: object) -> float:
        """将 LLM 返回的成本标签转换为数值成本
        
        DeepSeek 可能返回中文标签如"低"/"中"/"高"，
        调度是控制平面操作，需要将标签安全转换为数值。
        """
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, numbers.Real):
            return float(value) if math.isfinite(float(value)) else 0.0
        text = str(value).strip().casefold()
        labels = {
            "低": 0.1,
            "很低": 0.05,
            "较低": 0.1,
            "low": 0.1,
            "中": 0.5,
            "中等": 0.5,
            "一般": 0.5,
            "适中": 0.5,
            "medium": 0.5,
            "moderate": 0.5,
            "高": 1.0,
            "较高": 1.0,
            "很高": 1.5,
            "high": 1.0,
        }
        if text in labels:
            return labels[text]
        try:
            parsed = float(text)
        except (TypeError, ValueError):
            return 0.0
        return parsed if math.isfinite(parsed) else 0.0

    def schedule(self, hypothesis: HypothesisState, estimate: UtilityEstimate | None = None) -> None:
        cost = self._coerce_cost(hypothesis.metadata.get("cost", 0.0))
        estimate = estimate or UtilityEstimate(self.expected_information_gain(hypothesis), cost)
        hypothesis.priority = estimate.utility
        self._states[hypothesis.id] = hypothesis
        self._counter += 1
        heapq.heappush(self._queue, (-estimate.utility, self._counter, hypothesis.id))

    def next_hypothesis(self) -> HypothesisState | None:
        while self._queue:
            neg_utility, _, hypothesis_id = heapq.heappop(self._queue)
            state = self._states.get(hypothesis_id)
            if state is not None and math.isclose(state.priority, -neg_utility):
                del self._states[hypothesis_id]
                return state
        return None

    def update(self, hypothesis: HypothesisState) -> None:
        self.schedule(hypothesis)