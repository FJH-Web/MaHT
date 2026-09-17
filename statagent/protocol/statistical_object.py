"""
标准统计决策协议 (SDP) 数据模型

理论映射说明：
- 论文中的"假设状态" H_i = (d_i, s_i, c_i, p_i, E_i, q_i, D_i) 对应本文件的 HypothesisState 类
- 论文中的"统计决策协议" SDP = (H, E, M, R, P, D, C, T) 对应本文件的 StatisticalObject 类

本模块定义的核心数据结构：
- HypothesisState: 假设状态，包含假设描述、状态、置信度、后验概率等
- StatisticalObject: 统计对象，是 Agent 之间唯一允许交换的消息类型
- Evidence: 证据，记录假设的支持信息
- MethodResult: 方法执行结果，包含统计量、p值、效应量等
- VisualizationArtifact: 可视化产物，包含本地生成的图表及其语义数据
- InterpretationBundle: 结构化的LLM解释结果

注意：Agent 之间只能交换 StatisticalObject，不能使用原始提示文本。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Mapping


def utc_now() -> str:
    """获取当前 UTC 时间戳"""
    return datetime.now(UTC).isoformat()


class HypothesisStatus(str, Enum):
    """假设状态枚举，表示假设在推断过程中的当前阶段"""
    PROPOSED = "proposed"       # 已提出，等待检验
    READY = "ready"             # 准备就绪，可以检验
    TESTING = "testing"         # 正在检验中
    SUPPORTED = "supported"     # 被支持
    REJECTED = "rejected"       # 被拒绝
    INCONCLUSIVE = "inconclusive"  # 证据不足
    BLOCKED = "blocked"         # 被阻塞（依赖未满足）


class Decision(str, Enum):
    """统计决策枚举，表示假设检验的最终决策结果"""
    PENDING = "pending"         # 待决策
    SUPPORT = "support"         # 支持原假设
    REJECT = "reject"           # 拒绝原假设
    INCONCLUSIVE = "inconclusive"  # 证据不足


@dataclass(slots=True)
class Evidence:
    """证据类：记录假设的支持信息
    
    属性：
        source: 证据来源（如方法名称、文献引用等）
        summary: 证据摘要描述
        strength: 证据强度，取值范围 [0, 1]
        payload: 附加的元数据字典
        timestamp: 证据创建时间戳
    """
    source: str
    summary: str
    strength: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)


@dataclass(slots=True)
class VisualizationArtifact:
    """可视化产物类：可复现的本地图表及其语义解释数据
    
    属性：
        id: 唯一标识符
        title: 图表标题
        kind: 图表类型（如 distribution, boxplot, qq_plot 等）
        path: 本地文件路径
        caption: 图表说明文字
        semantic_payload: 用于解释图表的语义数据
        created_at: 创建时间戳
    """

    id: str
    title: str
    kind: str
    path: str
    caption: str
    semantic_payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class InterpretationBundle:
    """结构化LLM解释类：LLM对统计结果的结构化解读
    
    注意：文本解释不能改变数值结果，只能提供语义层面的解读。
    
    属性：
        summary: 总体摘要
        statistical_interpretation: 统计意义上的解释
        practical_interpretation: 实际意义的解释
        assumption_assessment: 假设满足情况的评估
        limitations: 局限性列表
        result_interpretations: 各个统计结果字段的解释
        figure_interpretations: 各个图表的解释
        conclusion: 结论
        model: 使用的模型名称
    """

    summary: str
    statistical_interpretation: str
    practical_interpretation: str
    assumption_assessment: str
    limitations: list[str] = field(default_factory=list)
    result_interpretations: dict[str, str] = field(default_factory=dict)
    figure_interpretations: dict[str, str] = field(default_factory=dict)
    conclusion: str = ""
    model: str | None = None


@dataclass(slots=True)
class HypothesisState:
    """假设状态类：论文中假设状态 H_i 的工程实现
    
    这是统计推断过程中的核心数据结构，记录假设的完整状态信息。
    
    属性：
        id: 假设唯一标识符
        description: 假设描述文本
        status: 当前状态（HypothesisStatus枚举）
        evidence: 证据列表
        priority: 优先级（由调度器使用）
        dependencies: 依赖的其他假设ID列表
        metadata: 附加元数据
    """

    id: str
    description: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    evidence: list[Evidence] = field(default_factory=list)
    priority: float = 0.0
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Hypothesis id cannot be empty")
        if not self.description.strip():
            raise ValueError("Hypothesis description cannot be empty")
        self.dependencies = list(dict.fromkeys(self.dependencies))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HypothesisState":
        values = dict(data)
        values["status"] = HypothesisStatus(values.get("status", HypothesisStatus.PROPOSED))
        values["evidence"] = [
            item if isinstance(item, Evidence) else Evidence(**item)
            for item in values.get("evidence", [])
        ]
        return cls(**values)


@dataclass(slots=True)
class MethodResult:
    """方法执行结果类：记录统计方法执行后的完整结果
    
    属性：
        method_id: 方法唯一标识符
        api: 调用的API名称
        estimates: 参数估计值字典
        p_value: p值（如果适用）
        confidence_interval: 置信区间
        effect_size: 效应量
        assumption_checks: 假设检验结果列表
        residuals: 残差值（如果适用）
        diagnostics: 诊断信息
        warnings: 警告信息
        visualizations: 生成的可视化产物列表
        reproducibility: 可复现性信息（如随机种子等）
    """
    method_id: str
    api: str
    estimates: dict[str, Any] = field(default_factory=dict)
    p_value: float | None = None
    confidence_interval: dict[str, Any] | None = None
    effect_size: dict[str, Any] | None = None
    assumption_checks: list[dict[str, Any]] = field(default_factory=list)
    residuals: list[float] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    visualizations: list[VisualizationArtifact] = field(default_factory=list)
    reproducibility: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MethodResult":
        values = dict(data)
        values["visualizations"] = [
            item if isinstance(item, VisualizationArtifact) else VisualizationArtifact(**item)
            for item in values.get("visualizations", [])
        ]
        return cls(**values)


@dataclass(slots=True)
class StatisticalObject:
    """统计对象类：Agent之间唯一允许交换的消息类型
    
    这是论文中统计决策协议(SDP)的工程实现，包含假设、证据、方法、结果、决策等完整信息。
    Agent之间只能交换此对象，不能使用原始提示文本。
    
    属性：
        hypothesis: 假设状态
        evidence: 证据列表
        method: 使用的统计方法信息
        result: 方法执行结果
        decision: 决策结果（Decision枚举）
        interpretation: LLM解释结果
        timestamp: 时间戳
        trace_id: 追踪ID，用于关联整个推断过程
    """

    hypothesis: HypothesisState
    evidence: list[Evidence] = field(default_factory=list)
    method: dict[str, Any] | None = None
    result: MethodResult | None = None
    decision: Decision = Decision.PENDING
    interpretation: InterpretationBundle | None = None
    timestamp: str = field(default_factory=utc_now)
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StatisticalObject":
        values = dict(data)
        values["hypothesis"] = HypothesisState.from_dict(values["hypothesis"])
        values["evidence"] = [Evidence(**item) for item in values.get("evidence", [])]
        if values.get("result"):
            values["result"] = MethodResult.from_dict(values["result"])
        if values.get("interpretation"):
            values["interpretation"] = InterpretationBundle(**values["interpretation"])
        values["decision"] = Decision(values.get("decision", Decision.PENDING))
        return cls(**values)