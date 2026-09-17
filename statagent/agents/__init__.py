"""
Agent 模块：六类 SDP Agent 的实现

本模块包含统计推断流程中六类核心 Agent：
- ParserAgent: 解析用户自然语言输入
- HypothesisManager: 管理假设生命周期
- RetrievalAgent: 检索匹配统计方法
- InferenceAgent: 执行统计推断
- DecisionAgent: 做出统计决策
- ReportAgent: 生成分析报告

每个 Agent 都继承自 StatisticalAgent 基类，
Agent 之间只能通过 StatisticalObject 进行通信。
"""

from statagent.agents.decision_agent.agent import DecisionAgent
from statagent.agents.hypothesis_manager.agent import HypothesisManager
from statagent.agents.inference_agent.agent import InferenceAgent
from statagent.agents.parser_agent.agent import ParserAgent
from statagent.agents.report_agent.agent import ReportAgent
from statagent.agents.retrieval_agent.agent import RetrievalAgent

__all__ = ["DecisionAgent", "HypothesisManager", "InferenceAgent", "ParserAgent", "ReportAgent", "RetrievalAgent"]