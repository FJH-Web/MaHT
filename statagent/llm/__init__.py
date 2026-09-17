"""
LLM 接口模块

本模块提供 LLM 客户端和语义匹配功能。
核心类：
- DeepSeekClient: DeepSeek API 客户端
- LLMError: LLM 调用错误
- SemanticMethodMatcher: 语义方法匹配器
- MethodMatch: 方法匹配结果

设计原则：
- 通过语义匹配理解用户的统计推断意图
- 将自然语言请求转换为方法卡片选择
"""

from statagent.llm.deepseek import DeepSeekClient, LLMError
from statagent.llm.semantic import MethodMatch, SemanticMethodMatcher

__all__ = ["DeepSeekClient", "LLMError", "MethodMatch", "SemanticMethodMatcher"]