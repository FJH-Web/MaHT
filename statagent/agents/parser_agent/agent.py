"""
解析 Agent 模块

Parser Agent 是唯一接受用户自然语言的组件。
它负责将用户的文本输入解析为结构化的假设对象。

核心功能：
- 解析用户输入的自然语言请求
- 生成结构化的 HypothesisState 对象列表
- 为每个假设分配唯一的 trace_id 用于追踪
"""

from __future__ import annotations

import uuid
import inspect
from dataclasses import dataclass

from statagent.agents.parser_agent.deepseek import HypothesisParserBackend
from statagent.protocol.statistical_object import StatisticalObject


@dataclass(slots=True)
class ParserAgent:
    """解析 Agent：唯一接受用户自然语言输入的组件"""
    
    backend: HypothesisParserBackend

    def parse(self, user_text: str, context: dict | None = None) -> list[StatisticalObject]:
        """解析用户输入为结构化假设列表
        
        参数：
            user_text: 用户输入的自然语言文本
            context: 可选的上下文信息
            
        返回：
            StatisticalObject 列表，每个包含一个假设
        """
        accepts_context = len(inspect.signature(self.backend.parse).parameters) >= 2
        hypotheses = self.backend.parse(user_text, context) if context is not None and accepts_context else self.backend.parse(user_text)
        trace_id = uuid.uuid4().hex
        if context:
            for hypothesis in hypotheses:
                hypothesis.metadata["analysis_context"] = dict(context)
        return [StatisticalObject(hypothesis=item, trace_id=trace_id) for item in hypotheses]