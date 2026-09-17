"""
证据循环编排模块

Evidence Loop 是有限证据循环的编排器，这是一个决策循环，而非代码循环。

核心流程：
1. Parser 解析用户输入为假设
2. Hypothesis Manager 管理假设生命周期
3. Retrieval Agent 检索匹配方法
4. Inference Agent 执行统计推断
5. Interpretation Agent 解释结果和图形
6. Decision Agent 做出统计决策
7. 存储到 Statistical Memory
8. Report Agent 生成报告

设计原则：
- 每个假设最多执行一次
- 统计量与图片始终由本地注册算法生成
- LLM 负责结构化理解和方法匹配，不生成统计代码
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from statagent.agents.pipeline_executor.agent import PipelineExecutor
from statagent.agents.decision_agent.agent import DecisionAgent
from statagent.agents.hypothesis_manager.agent import HypothesisManager
from statagent.agents.inference_agent.agent import InferenceAgent
from statagent.agents.interpretation_agent.agent import InterpretationAgent
from statagent.agents.parser_agent.agent import ParserAgent
from statagent.agents.parser_agent.deepseek import DeepSeekHypothesisParser
from statagent.agents.report_agent.agent import ReportAgent
from statagent.agents.retrieval_agent.agent import RetrievalAgent
from statagent.api.dispatcher import StatisticalDispatcher
from statagent.hypothesis.graph import HypothesisGraph
from statagent.llm.deepseek import DeepSeekClient
from statagent.llm.semantic import SemanticMethodMatcher
from statagent.memory.statistical_memory import MemoryRecord, StatisticalMemory
from statagent.protocol.decision_protocol import ThresholdDecisionPolicy
from statagent.protocol.statistical_object import StatisticalObject
from statagent.retrieval.embedding import HashingEmbedding
from statagent.retrieval.retriever import HierarchicalRetriever
from statagent.retrieval.vector_store import InMemoryVectorStore
from statagent.scheduler.active_scheduler import ActiveHypothesisScheduler
from statagent.statistical_library import build_default_registry
from statagent.report.composer import PaperReportComposer
from statagent.visualization.engine import VisualizationEngine
from statagent.datasets.intent import resolve_requested_columns

logger = logging.getLogger("statagent")


@dataclass(frozen=True, slots=True)
class ExecutionInput:
    """执行输入：包含数据和参数"""
    
    data: dict[str, Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


DataProvider = Callable[[StatisticalObject], ExecutionInput]


@dataclass(slots=True)
class StatAgentConfig:
    alpha: float = 0.05
    random_seed: int = 2026
    memory_path: Path | None = None
    retrieval_dimensions: int = 256
    log_level: int = logging.INFO
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    artifact_dir: Path = Path("artifacts")
    cleanup_on_startup: bool = False
    use_temp_dir: bool = False

    def __post_init__(self) -> None:
        self.memory_path = Path(self.memory_path) if self.memory_path is not None else None
        self.artifact_dir = Path(self.artifact_dir)
        if self.use_temp_dir:
            import tempfile
            self.artifact_dir = Path(tempfile.gettempdir()) / "statagent_artifacts"
        if self.cleanup_on_startup and self.artifact_dir.exists():
            import shutil
            shutil.rmtree(self.artifact_dir, ignore_errors=True)


@dataclass(slots=True)
class EvidenceLoop:
    parser: ParserAgent
    hypothesis_manager: HypothesisManager
    retrieval_agent: RetrievalAgent
    inference_agent: InferenceAgent
    decision_agent: DecisionAgent
    report_agent: ReportAgent
    scheduler: ActiveHypothesisScheduler
    visualization_engine: VisualizationEngine
    interpretation_agent: InterpretationAgent
    pipeline_executor: PipelineExecutor | None = None
    _registry: Any = field(default=None, repr=False)

    def run(
        self,
        user_text: str,
        data_provider: DataProvider,
        context: dict[str, Any] | None = None,
        exploratory_data: Any | None = None,
    ) -> str:
        """Execute each parsed hypothesis at most once and return a memory-only report."""
        analysis_context = dict(context or {})
        analysis_context["original_request"] = user_text
        dataset_profile = analysis_context.get("dataset_profile", {})
        analysis_context["requested_columns"] = resolve_requested_columns(user_text, dataset_profile)
        messages = self.parser.parse(user_text, analysis_context)
        for message in messages:
            self.hypothesis_manager.process(message)

        by_id = {message.hypothesis.id: message for message in messages}
        confirmed_columns = analysis_context.get("confirmed_columns", [])
        requested_columns = analysis_context.get("requested_columns", [])
        target_columns = confirmed_columns if confirmed_columns else requested_columns
        exploratory_artifacts = (
            self.visualization_engine.render_exploratory(exploratory_data, target_columns)
            if exploratory_data is not None
            else []
        )
        queued: set[str] = set()
        completed: set[str] = set()
        self._schedule_ready(queued, completed)

        while len(completed) < len(messages):
            hypothesis = self.scheduler.next_hypothesis()
            if hypothesis is None:
                blocked = sorted(set(by_id) - completed)
                raise RuntimeError(f"No schedulable hypothesis; blocked: {blocked}")
            queued.discard(hypothesis.id)
            message = by_id[hypothesis.id]
            message = self.retrieval_agent.process(message)
            execution_input = data_provider(message)
            
            if self.pipeline_executor is not None and message.method is not None:
                method_id = message.method.get("method_id", "")
                final_method_id, summary, assumption_results, alternative_used = self.pipeline_executor.check_and_prepare_method(
                    message, execution_input.data, execution_input.diagnostics
                )
                
                if alternative_used:
                    new_card = self._registry.card(final_method_id)
                    card_fields = {item.name for item in new_card.__dataclass_fields__.values()} if hasattr(new_card, '__dataclass_fields__') else set()
                    message.method["method_id"] = final_method_id
                    message.method["name"] = new_card.name
                    message.method["api"] = new_card.api
                    message.method["inputs"] = new_card.inputs
                    message.method["outputs"] = new_card.outputs
                    message.hypothesis.metadata["pipeline_assumption_results"] = [
                        {"type": r.assumption_type, "test_method": r.test_method_name, "p_value": r.p_value, "passed": r.passed, "conclusion": r.conclusion}
                        for r in assumption_results
                    ]
                    message.hypothesis.metadata["pipeline_summary"] = summary
                    message.hypothesis.metadata["original_method_id"] = method_id
            
            message = self.inference_agent.process_with_data(
                message, execution_input.data, execution_input.parameters, execution_input.diagnostics
            )
            self.visualization_engine.render(message, execution_input.data)
            if exploratory_artifacts and message.result:
                message.result.visualizations = [*exploratory_artifacts, *message.result.visualizations]
                exploratory_artifacts = []
            record_id = f"hypothesis:{hypothesis.id}"
            self.report_agent.memory.update(MemoryRecord(record_id=record_id, message=message))
            message = self.decision_agent.process(message)
            self.hypothesis_manager.synchronize(message)
            message = self.interpretation_agent.process(message)
            self.report_agent.memory.update(MemoryRecord(record_id=record_id, message=message))
            by_id[hypothesis.id] = message
            completed.add(hypothesis.id)
            self._schedule_ready(queued, completed)
        return self.report_agent.generate()

    def _schedule_ready(self, queued: set[str], completed: set[str]) -> None:
        for hypothesis in self.hypothesis_manager.graph.ready_nodes():
            if hypothesis.id not in queued and hypothesis.id not in completed:
                self.scheduler.schedule(hypothesis)
                queued.add(hypothesis.id)


def build_default_app(config: StatAgentConfig | None = None) -> EvidenceLoop:
    config = config or StatAgentConfig()
    logging.basicConfig(level=config.log_level)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    registry = build_default_registry()
    dispatcher = StatisticalDispatcher(registry)
    pipeline_executor = PipelineExecutor(dispatcher, registry)
    client = DeepSeekClient(api_key=config.deepseek_api_key, base_url=config.deepseek_base_url, model=config.deepseek_model)
    retriever = HierarchicalRetriever(
        HashingEmbedding(config.retrieval_dimensions), InMemoryVectorStore()
    )
    retriever.index(registry.cards())
    memory = StatisticalMemory(config.memory_path)
    return EvidenceLoop(
        parser=ParserAgent(DeepSeekHypothesisParser(api_key=config.deepseek_api_key, base_url=config.deepseek_base_url, model=config.deepseek_model)),
        hypothesis_manager=HypothesisManager(HypothesisGraph()),
        retrieval_agent=RetrievalAgent(retriever, SemanticMethodMatcher(client)),
        inference_agent=InferenceAgent(dispatcher),
        decision_agent=DecisionAgent(ThresholdDecisionPolicy(alpha=config.alpha)),
        report_agent=ReportAgent(memory, PaperReportComposer(client)),
        scheduler=ActiveHypothesisScheduler(),
        visualization_engine=VisualizationEngine(config.artifact_dir / "figures"),
        interpretation_agent=InterpretationAgent(client),
        pipeline_executor=pipeline_executor,
        _registry=registry,
    )