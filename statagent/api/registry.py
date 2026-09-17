"""
统计方法注册表模块

本模块提供动态注册表，将 Method Card 与预部署的统计 API 进行关联。
核心功能：
- 注册新的统计方法卡片和对应函数
- 根据方法ID获取方法卡片
- 根据API名称获取执行函数

设计原则：
- 方法ID和API名称必须唯一
- 新方法只需注册新的 Card 与预部署函数，无需修改路由器
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from statagent.protocol.statistical_object import MethodResult
from statagent.retrieval.method_card import MethodCard

# 统计函数的类型定义：接收 (输入数据, 参数) 返回 MethodResult
StatisticalFunction = Callable[[dict[str, Any], dict[str, Any]], MethodResult]


@dataclass(slots=True)
class StatisticalMethodRegistry:
    """统计方法注册表：将方法卡片与本地函数进行关联"""
    
    _cards: dict[str, MethodCard] = field(default_factory=dict)
    _functions: dict[str, StatisticalFunction] = field(default_factory=dict)

    def register(self, card: MethodCard, function: StatisticalFunction) -> None:
        """注册新的方法卡片和对应函数"""
        if card.method_id in self._cards:
            raise KeyError(f"Duplicate method id: {card.method_id}")
        if card.api in self._functions:
            raise KeyError(f"Duplicate API id: {card.api}")
        self._cards[card.method_id] = card
        self._functions[card.api] = function

    def card(self, method_id: str) -> MethodCard:
        """根据方法ID获取方法卡片"""
        return self._cards[method_id]

    def function(self, api: str) -> StatisticalFunction:
        """根据API名称获取统计函数"""
        try:
            return self._functions[api]
        except KeyError as exc:
            raise KeyError(f"Unregistered statistical API: {api}") from exc

    def cards(self) -> list[MethodCard]:
        """获取所有已注册的方法卡片"""
        return list(self._cards.values())