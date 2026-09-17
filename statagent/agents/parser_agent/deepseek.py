"""
DeepSeek 解析器后端模块

本模块提供 DeepSeek API 的适配器，用于将自然语言解析为 HypothesisState 对象。

设计原则：
- DeepSeek 仅用于语义解析和协调
- 系统消息明确禁止生成统计代码
- 请求严格的 JSON 模式输出
- 源代码中不存储 API 密钥

核心类：
- HypothesisParserBackend: 解析器后端抽象基类
- DeepSeekHypothesisParser: DeepSeek 实现
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from statagent.protocol.statistical_object import HypothesisState
from statagent.llm.deepseek import DeepSeekClient, LLMError


class HypothesisParserBackend(ABC):
    """解析器后端抽象基类"""
    
    @abstractmethod
    def parse(self, user_text: str, context: dict[str, Any] | None = None) -> list[HypothesisState]:
        """将用户文本解析为假设状态列表"""
        ...


class ParserBackendError(RuntimeError):
    """解析器后端错误异常"""
    pass


@dataclass(slots=True)
class DeepSeekHypothesisParser(HypothesisParserBackend):
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    timeout_seconds: float = 45.0
    allow_local_fallback: bool = True

    def __post_init__(self) -> None:
        self.api_key = (self.api_key or os.getenv("DEEPSEEK_API_KEY") or "").strip() or None
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", self.base_url).rstrip("/")
        self.model = os.getenv("DEEPSEEK_MODEL", self.model)

    def parse(self, user_text: str, context: dict[str, Any] | None = None) -> list[HypothesisState]:
        if not self.api_key:
            if self.allow_local_fallback:
                return self._local_parse(user_text, context or {})
            raise ParserBackendError("DEEPSEEK_API_KEY is not configured")
        try:
            client = DeepSeekClient(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                timeout_seconds=self.timeout_seconds,
            )
            parsed = client.complete_json(
                self._system_instruction(),
                json.dumps({"request": user_text, "context": context or {}}, ensure_ascii=False),
            )
            return self._validate(parsed)
        except (LLMError, KeyError, IndexError, TypeError, ValueError) as exc:
            if self.allow_local_fallback:
                return self._local_parse(user_text, context or {})
            raise ParserBackendError(f"DeepSeek hypothesis parsing failed: {exc}") from exc

    @staticmethod
    def _system_instruction() -> str:
        return (
            "你是统计推断系统边界处的中文语义解析器。禁止生成 Python、R、SQL、伪代码或任何可执行分析。"
            "请把用户需求转换为可证伪的统计假设，并且所有面向用户的描述必须使用简体中文。"
            "只返回一个 JSON 对象，顶层键为 hypotheses。每个元素必须包含 id、description、dependencies、"
            "priority 和 metadata。metadata 必须包含 retrieval_terms、variables、"
            "outcome_type、design 和 cost。retrieval_terms 可同时包含中英文统计术语以提高检索率。"
            "识别变量和研究设计时必须参考给定的数据集概况。依赖关系必须无环且前置假设在先。"
            "不得选择 API，不得编造统计结果。"
            "【变量解析规则】当用户在 variables 中指定变量时，必须根据 dataset_profile 中的 columns 顺序解析为具体列名。"
            "支持的描述方式："
            "1. 直接使用列名：如「身高」「age」"
            "2. 位置描述：「前N个变量」「后N个变量」「第N个变量」「第N-M个变量」「中间N个变量」"
            "3. 排除描述：「除了前N个以外的变量」「除了后N个以外的变量」"
            "4. 组合描述：「第1、3、5个变量」"
            "解析时请参考 dataset_profile 中 columns 的顺序，从1开始计数，排除 identifier_candidate 为 true 的列。"
        )

    @staticmethod
    def _local_parse(user_text: str, context: dict[str, Any]) -> list[HypothesisState]:
        """Deterministic offline boundary parser for smoke tests and demos without a key."""
        text = user_text.casefold()
        if any(token in text for token in ("正态", "normality", "shapiro", "anderson", "分布检验", "拟合优度")):
            terms, category = ["normality test", "Shapiro-Wilk", "Anderson-Darling", "distribution test"], "分布假设检验"
        elif any(token in text for token in ("方差齐", "variance", "levene", "bartlett")):
            terms, category = ["variance homogeneity", "Levene", "Bartlett"], "方差齐性检验"
        elif any(token in text for token in ("anova", "三组", "多组", "multiple groups")):
            terms, category = ["one-way anova", "multiple groups", "mean"], "参数检验"
        elif any(token in text for token in ("非参数", "wilcoxon", "kruskal", "friedman", "秩和")):
            terms, category = ["nonparametric test", "rank test"], "非参数检验"
        elif any(token in text for token in ("生存", "survival", "log-rank", "logrank")):
            terms, category = ["log-rank test", "survival comparison", "censoring"], "生存假设检验"
        elif any(token in text for token in ("卡方", "chi-square", "categorical", "分类")):
            terms, category = ["chi-square", "contingency", "categorical"], "列联表检验"
        elif any(token in text for token in ("相关", "correlation", "pearson", "spearman")):
            terms, category = ["correlation test", "Pearson", "Spearman"], "相关性检验"
        elif any(token in text for token in ("随机性", "游程", "runs test", "randomness")):
            terms, category = ["runs test", "randomness"], "随机性检验"
        elif any(token in text for token in ("两组", "two groups", "均值", "平均", "比较", "差异", "mean", "t-test", "t 检验")):
            terms, category = ["independent t-test", "two groups", "mean"], "参数检验"
        else:
            terms, category = ["hypothesis test", "significance test"], None
        return [
            HypothesisState(
                id="H001",
                description=user_text.strip() or "Describe the supplied dataset",
                metadata={"retrieval_terms": terms, **({"category": category} if category else {}), "analysis_context": context, "llm_parser": "local-fallback"},
            )
        ]

    @staticmethod
    def _validate(payload: dict[str, Any]) -> list[HypothesisState]:
        raw_hypotheses = payload.get("hypotheses")
        if not isinstance(raw_hypotheses, list) or not raw_hypotheses:
            raise ValueError("response must contain a non-empty hypotheses list")
        hypotheses: list[HypothesisState] = []
        seen: set[str] = set()
        for index, item in enumerate(raw_hypotheses, start=1):
            if not isinstance(item, dict):
                raise ValueError("each hypothesis must be an object")
            hypothesis_id = str(item.get("id") or f"H{index:03d}")
            dependencies = [str(value) for value in item.get("dependencies", [])]
            if any(value not in seen for value in dependencies):
                raise ValueError("dependencies must reference earlier hypotheses")
            hypothesis = HypothesisState(
                id=hypothesis_id,
                description=str(item["description"]),
                priority=float(item.get("priority", 0.0)),
                dependencies=dependencies,
                metadata=dict(item.get("metadata", {})),
            )
            hypotheses.append(hypothesis)
            seen.add(hypothesis_id)
        return hypotheses