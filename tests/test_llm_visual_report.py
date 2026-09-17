from __future__ import annotations

from pathlib import Path

import pandas as pd

from statagent.agents.interpretation_agent.agent import InterpretationAgent
from statagent.datasets import DataFrameAdapter, profile_dataframe, resolve_requested_columns
from statagent.llm.semantic import SemanticMethodMatcher
from statagent.memory.statistical_memory import MemoryRecord, StatisticalMemory
from statagent.protocol.statistical_object import HypothesisState, MethodResult, StatisticalObject, VisualizationArtifact
from statagent.retrieval.method_card import MethodCard
from statagent.visualization import VisualizationEngine
from statagent.visualization import ExploratoryVisualizationEngine


def test_dataframe_profile_and_binding_for_local_api() -> None:
    frame = pd.DataFrame({"control": [1.0, 2.0, 3.0], "treatment": [4.0, 5.0, 6.0]})
    profile = profile_dataframe(frame)
    assert profile["rows"] == 3
    card = MethodCard(
        method_id="PAR001", name="Independent t-test", category=("Parametric",),
        description="compare", inputs=("group_a", "group_b"), outputs=("p_value",), api="run_ttest_independent",
    )
    message = StatisticalObject(HypothesisState("H1", "means differ"), method={**card.to_dict(), "match": {"data_mapping": {"group_a": "control", "group_b": "treatment"}}})
    execution = DataFrameAdapter(frame).execution_input(message)
    assert execution.data == {"group_a": [1.0, 2.0, 3.0], "group_b": [4.0, 5.0, 6.0]}


def test_visualization_and_html_report_are_generated(tmp_path: Path) -> None:
    card = MethodCard(
        method_id="PAR001", name="Independent t-test", category=("Parametric",),
        description="compare", inputs=("group_a", "group_b"), outputs=("p_value",), api="run_ttest_independent",
        visualizations=("two_group_distribution", "two_group_boxplot"),
    )
    message = StatisticalObject(
        HypothesisState("H1", "means differ"),
        method=card.to_dict(),
        result=MethodResult("PAR001", "run_ttest_independent", {"t": 2.0}, p_value=0.03),
    )
    artifacts = VisualizationEngine(tmp_path / "figures").render(message, {"group_a": [1, 2, 3], "group_b": [4, 5, 6]})
    assert len(artifacts) == 2
    memory = StatisticalMemory()
    memory.update(MemoryRecord("H1", message))
    message = InterpretationAgent(client=None).process(message)
    memory.update(MemoryRecord("H1", message))
    from statagent.report.composer import PaperReportComposer
    html = PaperReportComposer().render_html(memory.search(), "Test paper")
    assert "Test paper" in html
    assert "data:image/png;base64" in html
    assert "Statistical Memory" in html
    assert "figure-grid" in html
    assert all(len(text) >= 100 for text in message.interpretation.figure_interpretations.values())


def test_matcher_fallback_is_constrained_to_candidates() -> None:
    cards = [
        MethodCard("A", "A", ("Parametric",), "A", ("values",), ("mean",), "run_a"),
        MethodCard("B", "B", ("Nonparametric",), "B", ("values",), ("median",), "run_b"),
    ]
    match = SemanticMethodMatcher(client=None).select(HypothesisState("H", "compare"), cards)
    assert match.method_id in {"A", "B"}


def test_chinese_exploratory_visualization_suite(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "年龄": [31, 42, 53, 64, 75, 48],
            "评分": [4.2, 5.1, 6.0, 7.5, 8.1, 5.8],
            "组别": ["治疗", "治疗", "对照", "对照", "治疗", "对照"],
        }
    )
    artifacts = ExploratoryVisualizationEngine(tmp_path / "eda").render(frame)
    assert len(artifacts) >= 10
    assert any("分布" in artifact.title for artifact in artifacts)
    assert any("缺失" in artifact.title for artifact in artifacts)
    assert all(Path(artifact.path).exists() for artifact in artifacts)


def test_explicit_columns_override_index_column_mapping() -> None:
    frame = pd.DataFrame({
        "Unnamed: 0": [1, 2, 3],
        "Sepal.Length": [5.1, 4.9, 4.7],
        "Sepal.Width": [3.5, 3.0, 3.2],
    })
    profile = profile_dataframe(frame)
    request = "请分析 Sepal.Length 和 Sepal.Width 之间的差异"
    locked = resolve_requested_columns(request, profile)
    assert locked == ["Sepal.Length", "Sepal.Width"]

    card = MethodCard(
        method_id="PAR001", name="Independent t-test", category=("Parametric",),
        description="compare", inputs=("group_a", "group_b"), outputs=("p_value",), api="run_ttest_independent",
    )
    match = SemanticMethodMatcher(client=None).select(
        HypothesisState("H1", request), [card], profile, locked
    )
    hypothesis = HypothesisState("H1", request, metadata={"analysis_context": {"original_request": request, "requested_columns": locked}})
    message = StatisticalObject(hypothesis, method={**card.to_dict(), "match": match.to_dict()})
    execution = DataFrameAdapter(frame).execution_input(message)
    assert execution.data["group_a"] == frame["Sepal.Length"].tolist()
    assert execution.data["group_b"] == frame["Sepal.Width"].tolist()
    assert "Unnamed: 0" not in str(match.data_mapping)


def test_llm_wrong_mapping_is_corrected_to_locked_columns() -> None:
    class WrongMappingClient:
        configured = True
        model = "test"

        def complete_json(self, *args, **kwargs):
            return {
                "method_id": "PAR001", "confidence": 0.9, "rationale": "测试",
                "data_mapping": {"group_a": "Unnamed: 0", "group_b": "Sepal.Length"},
                "parameters": {},
            }

    card = MethodCard(
        method_id="PAR001", name="Independent t-test", category=("Parametric",),
        description="compare", inputs=("group_a", "group_b"), outputs=("p_value",), api="run_ttest_independent",
    )
    match = SemanticMethodMatcher(client=WrongMappingClient()).select(
        HypothesisState("H1", "比较两个变量"), [card], {}, ["Sepal.Length", "Sepal.Width"]
    )
    assert match.data_mapping["group_a"]["column"] == "Sepal.Length"
    assert match.data_mapping["group_b"]["column"] == "Sepal.Width"
    assert "Unnamed: 0" in match.mapping_audit


def test_figure_interpretation_accepts_list_response(tmp_path: Path) -> None:
    class ListResponseClient:
        configured = True
        model = "test"

        def __init__(self):
            self.calls = 0

        def complete_json(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {
                    "summary": "摘要", "statistical_interpretation": "统计解释",
                    "practical_interpretation": "实际解释", "assumption_assessment": "假设检查",
                    "limitations": [], "result_interpretations": {},
                    "figure_interpretations": {"fig-1": "简短解释"}, "conclusion": "结论",
                }
            return {
                "figure_interpretations": [
                    {"id": "fig-1", "interpretation": "这是完整的图形解释。" * 15}
                ]
            }

    figure = VisualizationArtifact(
        id="fig-1", title="测试图", kind="scatter", path=str(tmp_path / "missing.png"),
        caption="测试图说明", semantic_payload={"横轴": "变量甲", "纵轴": "变量乙"},
    )
    message = StatisticalObject(
        HypothesisState("H1", "分析变量甲和变量乙"),
        method={"method_id": "TEST", "match": {}},
        result=MethodResult("TEST", "run_test", {"estimate": 1.0}, visualizations=[figure]),
    )
    interpreted = InterpretationAgent(client=ListResponseClient()).process(message)
    assert len(interpreted.interpretation.figure_interpretations["fig-1"]) >= 100
