"""
语义方法匹配模块

本模块使用 LLM 从候选方法卡片中选择最合适的统计方法。
特点：
- 匹配约束为已检索的方法卡片候选集
- 不允许 LLM 选择未在候选集中的方法
- 提供数据映射和参数建议
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from statagent.hypothesis.state import HypothesisState
from statagent.llm.deepseek import DeepSeekClient, LLMError
from statagent.retrieval.method_card import MethodCard
from statagent.datasets.intent import eligible_profile_columns, enforce_locked_mapping, mapping_columns


@dataclass(slots=True)
class MethodMatch:
    """方法匹配结果"""
    
    method_id: str
    confidence: float
    rationale: str
    data_mapping: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    locked_columns: list[str] = field(default_factory=list)
    mapping_audit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "data_mapping": self.data_mapping,
            "parameters": self.parameters,
            "locked_columns": self.locked_columns,
            "mapping_audit": self.mapping_audit,
        }


@dataclass(slots=True)
class SemanticMethodMatcher:
    """语义方法匹配器"""
    
    client: DeepSeekClient | None = None
    allow_fallback: bool = True

    def select(
        self,
        hypothesis: HypothesisState,
        candidates: list[MethodCard],
        dataset_profile: dict[str, Any] | None = None,
        requested_columns: list[str] | None = None,
    ) -> MethodMatch:
        """从候选方法卡片中选择最合适的方法"""
        if not candidates:
            raise ValueError("Method matching requires retrieved candidates")
        profile = dataset_profile or {}
        locked = list(requested_columns or [])
        if self.client and self.client.configured:
            try:
                payload = self.client.complete_json(
                    self._system_prompt(),
                    json_prompt(hypothesis, candidates, profile, locked),
                )
                match = self._validate(payload, candidates)
                return self._finalize(match, candidates, locked)
            except (LLMError, ValueError, TypeError):
                if not self.allow_fallback:
                    raise
        return self._fallback(candidates, hypothesis, profile, locked)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是 StatAgent 的统计方法匹配决策模块。只能从已召回的 Method Card 中选择一个 method_id，"
            "绝对不得发明方法、API、代码、统计量或结果。请结合中文研究需求、研究设计与数据集结构，"
            "选择最有统计学依据的方法。返回 JSON：method_id、confidence、rationale、data_mapping、parameters。"
            "rationale 必须使用简体中文，清楚说明选择依据、备选方法为何不优以及关键假设。"
            "data_mapping 把每个输入角色映射到真实列名或结构化分组规则。若数据不足，仍选择最佳候选，"
            "但必须在中文 rationale 中说明缺失信息。original_request 是最高优先级研究约束；"
            "locked_columns 是用户在原始需求中明确点名的变量。只要 locked_columns 非空，data_mapping "
            "只能引用这些列，不得加入索引列、第一列或任何未点名变量，也不得为了迁就错误映射而改变研究问题。"
        )

    @staticmethod
    def _validate(payload: dict[str, Any], candidates: list[MethodCard]) -> MethodMatch:
        allowed = {card.method_id for card in candidates}
        method_id = str(payload.get("method_id", ""))
        if method_id not in allowed:
            raise ValueError("LLM selected a method outside retrieved candidates")
        return MethodMatch(
            method_id=method_id,
            confidence=min(max(float(payload.get("confidence", 0.5)), 0.0), 1.0),
            rationale=str(payload.get("rationale", "")),
            data_mapping=dict(payload.get("data_mapping", {})),
            parameters=dict(payload.get("parameters", {})),
        )

    @staticmethod
    def _fallback(
        candidates: list[MethodCard],
        hypothesis: HypothesisState,
        profile: dict[str, Any],
        locked_columns: list[str] | None = None,
    ) -> MethodMatch:
        query = " ".join([hypothesis.description, *map(str, hypothesis.metadata.get("retrieval_terms", []))]).casefold()

        def score(card: MethodCard) -> float:
            searchable = card.search_text.casefold()
            phrases = [phrase.casefold() for phrase in (*card.tags, card.name, card.description)]
            value = sum(2.0 for phrase in phrases if phrase and phrase in query)
            query_tokens = set(re.findall(r"[a-z0-9][a-z0-9–-]*|[\u4e00-\u9fff]", query))
            value += 0.15 * len(query_tokens.intersection(set(re.findall(r"[a-z0-9][a-z0-9–-]*|[\u4e00-\u9fff]", searchable))))
            if any(token in query for token in ("two groups", "two independent", "两组", "两个独立")) and card.inputs == ("group_a", "group_b"):
                value += 5.0
            if any(token in query for token in ("paired", "配对", "重复测量")) and card.inputs == ("before", "after"):
                value += 5.0
            if any(token in query for token in ("multiple groups", "多组", "三组")) and card.inputs == ("groups",):
                value += 3.0
            return value

        card = max(candidates, key=score)
        locked = list(locked_columns or [])
        columns = locked or eligible_profile_columns(profile)
        mapping = enforce_locked_mapping(card.inputs, columns, {})
        audit = "已严格按用户点名变量锁定映射。" if locked else "用户未明确点名变量，已排除索引/编号列后自动映射。"
        return MethodMatch(
            card.method_id,
            0.35,
            "当前使用本地回退：在已召回 Method Card 中按研究设计、变量角色和统计术语匹配；配置可用的 DeepSeek 接口后将启用更细粒度中文语义匹配。",
            mapping,
            {},
            locked,
            audit,
        )

    @staticmethod
    def _finalize(match: MethodMatch, candidates: list[MethodCard], locked: list[str]) -> MethodMatch:
        card = next(card for card in candidates if card.method_id == match.method_id)
        original_columns = mapping_columns(match.data_mapping)
        match.data_mapping = enforce_locked_mapping(card.inputs, locked, match.data_mapping)
        match.locked_columns = locked
        final_columns = mapping_columns(match.data_mapping)
        if locked:
            replaced = sorted(set(original_columns) - set(locked))
            match.mapping_audit = (
                f"已锁定用户指定变量：{', '.join(locked)}。"
                + (f"已移除大模型误映射变量：{', '.join(replaced)}。" if replaced else "映射审计通过，未发现越权变量。")
                + f"最终参与分析的变量：{', '.join(final_columns)}。"
            )
        else:
            match.mapping_audit = "用户需求中未识别到明确列名，采用经标识列过滤后的自动映射。"
        return match


def json_prompt(
    hypothesis: HypothesisState,
    candidates: list[MethodCard],
    profile: dict[str, Any],
    locked_columns: list[str] | None = None,
) -> str:
    import json

    return json.dumps(
        {
            "hypothesis": hypothesis.to_dict(),
            "dataset_profile": profile,
            "original_request": hypothesis.metadata.get("analysis_context", {}).get("original_request", hypothesis.description),
            "locked_columns": list(locked_columns or []),
            "retrieved_method_cards": [card.to_dict() for card in candidates],
        },
        ensure_ascii=False,
    )