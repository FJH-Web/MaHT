"""
StatAgent: 以假设为中心的统计推断框架

本模块是 StatAgent 的顶层入口，导出核心组件供外部调用。
主要包含：证据循环、假设状态、统计对象等核心概念的接口。

设计原则：
- LLM 负责结构化理解、候选方法匹配、结果解释，但不生成或执行统计代码
- 内部只交换 StatisticalObject，业务逻辑不依赖自由文本
- 执行前必须检索 Method Card，Dispatcher 只允许运行注册表中的本地函数
"""

from statagent.app.pipeline import EvidenceLoop, StatAgentConfig
from statagent.protocol.statistical_object import HypothesisState, StatisticalObject

__all__ = ["EvidenceLoop", "HypothesisState", "StatAgentConfig", "StatisticalObject"]
__version__ = "0.1.0"