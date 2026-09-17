"""
推断 Agent 模块

InferenceAgent 负责执行统计推断。
主要功能：
- 调用本地注册的统计方法
- 生成统计结果
- 收集证据

注意：InferenceAgent 需要结构化数据输入，不能处理原始消息。
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from typing import Any

from statagent.agents.base import StatisticalAgent
from statagent.api.dispatcher import StatisticalDispatcher
from statagent.protocol.statistical_object import Evidence, HypothesisStatus, StatisticalObject
from statagent.retrieval.method_card import MethodCard


@dataclass(slots=True)
class InferenceAgent(StatisticalAgent):
    """推断 Agent：执行统计推断"""
    
    dispatcher: StatisticalDispatcher

    def process_with_data(
        self,
        message: StatisticalObject,
        data: dict[str, Any],
        parameters: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> StatisticalObject:
        """使用提供的数据执行统计推断
        
        参数：
            message: 包含假设和方法信息的统计对象
            data: 结构化输入数据
            parameters: 可选的统计方法参数
            diagnostics: 可选的诊断信息（如缺失值处理）
            
        返回：
            包含执行结果的统计对象
        """
        if message.method is None:
            raise ValueError("Inference requires a retrieved Method Card")
        message.hypothesis.status = HypothesisStatus.TESTING
        card_fields = {item.name for item in fields(MethodCard)}
        card_data = {key: value for key, value in message.method.items() if key in card_fields}
        card = MethodCard.from_dict(card_data)
        message.result = self.dispatcher.execute(card, data, parameters, diagnostics)
        evidence = Evidence(
            source=f"local_api:{card.api}",
            summary=f"{card.name} completed using registered method {card.method_id}",
            strength=1.0 - message.result.p_value if message.result.p_value is not None else 0.5,
            payload=message.result.to_dict(),
        )
        message.evidence.append(evidence)
        return message

    def process(self, message: StatisticalObject) -> StatisticalObject:
        """ InferenceAgent 需要结构化数据，请调用 process_with_data 方法"""
        raise TypeError("InferenceAgent requires structured data; call process_with_data")