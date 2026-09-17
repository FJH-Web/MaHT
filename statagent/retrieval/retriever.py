"""
层次化统计检索模块 (HSR)

本模块实现层次化统计检索功能。

理论映射说明：
- HSR 首先将查询约束到统计分类学的某个分支
- 然后对方法卡片叶子节点进行排序
- 检索的是元数据，不包含源代码

主要功能：
- 方法卡片索引
- 基于类别和语义相似度的检索
"""

from __future__ import annotations

from dataclasses import dataclass

from statagent.protocol.statistical_object import HypothesisState
from statagent.retrieval.embedding import EmbeddingProvider
from statagent.retrieval.method_card import MethodCard
from statagent.retrieval.vector_store import VectorStore


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """检索结果：包含方法卡片、得分和排名"""
    
    card: MethodCard
    score: float
    rank: int


class HierarchicalRetriever:
    """层次化检索器"""
    
    def __init__(self, embedding: EmbeddingProvider, store: VectorStore) -> None:
        self.embedding = embedding
        self.store = store
        self._cards: dict[str, MethodCard] = {}

    def index(self, cards: list[MethodCard]) -> None:
        """索引方法卡片"""
        vectors = self.embedding.embed([card.search_text for card in cards])
        for card, vector in zip(cards, vectors, strict=True):
            self._cards[card.method_id] = card
            self.store.upsert(
                card.method_id,
                vector,
                {"root": card.category[0], "category": "/".join(card.category)},
            )

    def retrieve(
        self,
        hypothesis: HypothesisState,
        *,
        limit: int = 3,
        category: str | None = None,
    ) -> list[RetrievalResult]:
        """检索匹配假设的方法卡片"""
        query_text = " ".join(
            [hypothesis.description, *map(str, hypothesis.metadata.get("retrieval_terms", []))]
        )
        query_vector = self.embedding.embed([query_text])[0]
        metadata_filter = {"root": category} if category else None
        matches = self.store.search(query_vector, limit=limit, metadata_filter=metadata_filter)
        return [
            RetrievalResult(card=self._cards[item_id], score=score, rank=rank)
            for rank, (item_id, score, _) in enumerate(matches, start=1)
        ]