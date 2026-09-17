"""
统计决策策略模块

本模块定义统计决策策略的抽象接口和默认实现。
策略是可替换的，修改决策逻辑不需要改变 Agent 代码。

核心类：
- DecisionPolicy: 抽象决策策略基类
- ThresholdDecisionPolicy: 基于p值阈值的默认决策策略
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from statagent.protocol.statistical_object import Decision, StatisticalObject


class DecisionPolicy(ABC):
    """决策策略抽象基类"""
    
    @abstractmethod
    def decide(self, message: StatisticalObject) -> StatisticalObject:
        """根据统计结果返回更新后的SDP消息，不重新计算统计量"""


@dataclass(slots=True)
class ThresholdDecisionPolicy(DecisionPolicy):
    """基于阈值的决策策略：将频率学派结果转换为决策
    
    这是默认的决策适配器，将统计检验结果（p值）转换为决策。
    注意：这并非通用的科学规则，只是一个工程实现。
    
    属性：
        alpha: 显著性水平，默认0.05
    """

    alpha: float = 0.05

    def decide(self, message: StatisticalObject) -> StatisticalObject:
        result = message.result
        if result is None or result.p_value is None:
            message.decision = Decision.INCONCLUSIVE
            return message
        
        method_code = ''
        if message.method:
            if hasattr(message.method, 'code'):
                method_code = message.method.code
            elif isinstance(message.method, dict):
                method_code = message.method.get('code', '')
        
        normality_tests = ['DIST001', 'DIST002', 'DIST003', 'DIST004', 'shapiro', 'royston']
        is_normality = any(nt in str(method_code).upper() for nt in normality_tests)
        
        if is_normality:
            message.decision = Decision.SUPPORT if result.p_value < self.alpha else Decision.REJECT
        else:
            message.decision = Decision.SUPPORT if result.p_value < self.alpha else Decision.REJECT
        return message