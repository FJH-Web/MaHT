"""
向量存储模块

本模块提供向量存储的抽象接口和确定性内存实现。
主要功能：
- VectorStore: 向量存储的抽象接口
- InMemoryVectorStore: 内存中的向量存储实现

设计原则：
- 支持元数据过滤
- 使用余弦相似度进行搜索
- 确定性实现确保可复现性
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class VectorStore(ABC):
    """向量存储抽象基类"""
    
    @abstractmethod
    def upsert(self, item_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """插入或更新向量条目"""
        ...

    @abstractmethod
    def search(
        self, vector: list[float], limit: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """搜索最相似的向量条目"""
        ...


@dataclass(slots=True)
class InMemoryVectorStore(VectorStore):
    """内存向量存储：确定性实现"""
    
    _items: dict[str, tuple[list[float], dict[str, Any]]] = field(default_factory=dict)

    def upsert(self, item_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """插入或更新向量条目"""
        self._items[item_id] = (list(vector), dict(metadata))

    def search(
        self, vector: list[float], limit: int = 5, metadata_filter: dict[str, Any] | None = None
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """搜索最相似的向量条目
        
        参数：
            vector: 查询向量
            limit: 返回结果数量上限
            metadata_filter: 元数据过滤条件
            
        返回：
            列表，每个元素为 (item_id, 相似度分数, 元数据)
        """
        candidates: list[tuple[str, float, dict[str, Any]]] = []
        for item_id, (candidate, metadata) in self._items.items():
            if metadata_filter and any(metadata.get(k) != v for k, v in metadata_filter.items()):
                continue
            score = self._cosine(vector, candidate)
            candidates.append((item_id, score, dict(metadata)))
        candidates.sort(key=lambda row: (-row[1], row[0]))
        return candidates[:limit]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        """计算余弦相似度"""
        if len(left) != len(right):
            raise ValueError("vector dimensions do not match")
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0