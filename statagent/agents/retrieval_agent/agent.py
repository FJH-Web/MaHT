"""
检索 Agent 模块

RetrievalAgent 负责检索匹配假设的统计方法。
主要功能：
- 从方法卡片目录中检索相关方法
- 支持基于分类的过滤
- 使用语义匹配器选择最佳方法
"""

from __future__ import annotations

from dataclasses import dataclass

from statagent.agents.base import StatisticalAgent
from statagent.protocol.statistical_object import StatisticalObject
from statagent.retrieval.retriever import HierarchicalRetriever
from statagent.llm.semantic import SemanticMethodMatcher


class MethodNotFoundError(LookupError):
    """方法未找到异常"""
    pass


@dataclass(slots=True)
class RetrievalAgent(StatisticalAgent):
    """检索 Agent：为假设匹配统计方法"""
    
    retriever: HierarchicalRetriever
    matcher: SemanticMethodMatcher | None = None

    def process(self, message: StatisticalObject) -> StatisticalObject:
        """根据假设检索匹配的统计方法"""
        category = message.hypothesis.metadata.get("category")
        category_aliases = {
            "Parametric": "参数检验", "Nonparametric": "非参数检验", "Categorical": "列联表检验",
            "Survival": "生存假设检验", "Distribution": "分布假设检验", "Variance": "方差齐性检验",
            "Correlation": "相关性检验", "Randomness": "随机性检验",
        }
        category = category_aliases.get(str(category), category) if category else None
        results = self.retriever.retrieve(message.hypothesis, limit=5 if category else 10, category=category)
        if not results and category:
            results = self.retriever.retrieve(message.hypothesis, limit=10)
        if not results:
            raise MethodNotFoundError(f"No Method Card found for {message.hypothesis.id}")
        if self.matcher:
            context = message.hypothesis.metadata.get("analysis_context", {})
            match = self.matcher.select(
                message.hypothesis,
                [result.card for result in results],
                context.get("dataset_profile", {}),
                context.get("requested_columns", []),
            )
            selected = next(result for result in results if result.card.method_id == match.method_id)
            message.method = {
                **selected.card.to_dict(),
                "retrieval_score": selected.score,
                "rank": selected.rank,
                "match": match.to_dict(),
                "candidate_method_ids": [result.card.method_id for result in results],
            }
        else:
            selected = results[0]
            message.method = {**selected.card.to_dict(), "retrieval_score": selected.score, "rank": selected.rank}
        return message