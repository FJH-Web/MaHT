"""
嵌入向量生成模块

本模块提供可替换的本地嵌入向量接口，无需网络访问。
主要功能：
- EmbeddingProvider: 嵌入向量生成的抽象接口
- HashingEmbedding: 基于哈希的确定性嵌入实现

设计原则：
- 使用哈希方式生成确定性嵌入向量，确保可复现性
- 不依赖外部API，保护数据隐私
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """嵌入向量生成器抽象基类"""
    
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本映射到数值向量"""
        ...


class HashingEmbedding(EmbeddingProvider):
    """基于哈希的确定性嵌入向量生成器
    
    这是一个无依赖的基线嵌入实现，确保可复现性。
    使用 BLAKE2 哈希函数将文本 tokens 映射到固定维度的向量。
    """
    
    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量生成嵌入向量"""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        """为单个文本生成嵌入向量"""
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w\-]+", text.casefold())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            index = number % self.dimensions
            sign = 1.0 if (number >> 8) % 2 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]