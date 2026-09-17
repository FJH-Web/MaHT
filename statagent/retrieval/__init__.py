"""
检索模块

本模块提供统计方法的检索功能。
核心组件：
- EmbeddingProvider: 嵌入向量生成器抽象接口
- HashingEmbedding: 基于哈希的确定性嵌入实现
- VectorStore: 向量存储抽象接口
- InMemoryVectorStore: 内存向量存储实现
- HierarchicalRetriever: 层次化检索器
- MethodCard: 方法卡片元数据
- RetrievalResult: 检索结果
"""

from statagent.retrieval.embedding import EmbeddingProvider, HashingEmbedding
from statagent.retrieval.method_card import MethodCard
from statagent.retrieval.retriever import HierarchicalRetriever, RetrievalResult
from statagent.retrieval.vector_store import InMemoryVectorStore

__all__ = [
    "EmbeddingProvider",
    "HashingEmbedding",
    "HierarchicalRetriever",
    "InMemoryVectorStore",
    "MethodCard",
    "RetrievalResult",
]