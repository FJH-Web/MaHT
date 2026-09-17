from __future__ import annotations

from pathlib import Path

from statagent.agents.parser_agent.deepseek import HypothesisParserBackend
from statagent.app.pipeline import ExecutionInput, StatAgentConfig, build_default_app
from statagent.memory.statistical_memory import MemoryRecord, StatisticalMemory
from statagent.protocol.statistical_object import HypothesisState, StatisticalObject
from statagent.scheduler.active_scheduler import ActiveHypothesisScheduler, UtilityEstimate


class FakeParser(HypothesisParserBackend):
    def parse(self, user_text: str) -> list[HypothesisState]:
        return [
            HypothesisState(
                id="H1",
                description="The means of two independent groups differ",
                metadata={
                    "category": "Parametric",
                    "retrieval_terms": ["independent t-test", "two groups", "mean"],
                    "cost": 0.1,
                },
            )
        ]


def test_memory_history_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = StatisticalMemory(path)
    message = StatisticalObject(HypothesisState(id="H1", description="A"))
    memory.update(MemoryRecord("r1", message))
    message.confidence = 0.9
    memory.update(MemoryRecord("r1", message))
    assert memory.history("r1")[0].message.confidence == 0.5
    assert StatisticalMemory(path).recall("r1").revision == 2


def test_scheduler_uses_information_gain_minus_cost() -> None:
    scheduler = ActiveHypothesisScheduler()
    low = HypothesisState(id="low", description="low")
    high = HypothesisState(id="high", description="high")
    scheduler.schedule(low, UtilityEstimate(0.5, 0.4))
    scheduler.schedule(high, UtilityEstimate(0.8, 0.1))
    assert scheduler.next_hypothesis().id == "high"  # type: ignore[union-attr]


def test_scheduler_accepts_chinese_cost_labels_from_parser() -> None:
    scheduler = ActiveHypothesisScheduler()
    hypothesis = HypothesisState(id="cn-cost", description="中文成本", metadata={"cost": "低"})
    scheduler.schedule(hypothesis)
    assert hypothesis.priority >= 0.0
    assert scheduler.next_hypothesis() is hypothesis


def test_finite_end_to_end_evidence_loop(tmp_path: Path) -> None:
    app = build_default_app(StatAgentConfig(memory_path=tmp_path / "memory.json"))
    app.parser.backend = FakeParser()
    report = app.run(
        "Compare two groups",
        lambda message: ExecutionInput(
            data={"group_a": [1, 2, 3, 4], "group_b": [8, 9, 10, 11]}
        ),
    )
    assert "独立样本 t 检验" in report
    assert "支持研究假设" in report
    assert len(app.report_agent.memory.history("hypothesis:H1")) == 2
