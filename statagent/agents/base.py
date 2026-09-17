"""
Agent 基类接口模块

本模块定义所有 Agent 的抽象基类 StatisticalAgent。
所有 Agent 都必须遵守统计决策协议（SDP）的约束。

设计原则：
- Agent 内部只能交换 StatisticalObject
- 禁止使用自由文本作为内部消息类型
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from statagent.protocol.statistical_object import StatisticalObject


class StatisticalAgent(ABC):
    """统计 Agent 抽象基类"""
    
    @abstractmethod
    def process(self, message: StatisticalObject) -> StatisticalObject:
        """处理一个 SDP 消息并返回更新后的消息
        
        注意：自由文本不是内部消息类型，所有通信必须使用 StatisticalObject。
        """