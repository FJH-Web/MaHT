from __future__ import annotations

import pytest

from statagent.hypothesis.graph import HypothesisGraph
from statagent.protocol.statistical_object import Evidence, HypothesisState, HypothesisStatus, StatisticalObject


def test_protocol_round_trip() -> None:
    message = StatisticalObject(
        hypothesis=HypothesisState(id="H1", description="Group means differ"),
        evidence=[Evidence(source="test", summary="observed")],
    )
    restored = StatisticalObject.from_dict(message.to_dict())
    assert restored.hypothesis.id == "H1"
    assert restored.evidence[0].source == "test"


def test_graph_dependency_order_updates_and_visualization() -> None:
    graph = HypothesisGraph()
    graph.add_node(HypothesisState(id="H1", description="Primary"))
    graph.add_node(HypothesisState(id="H2", description="Secondary", dependencies=["H1"]))
    assert [item.id for item in graph.topological_order()] == ["H1", "H2"]
    assert [item.id for item in graph.ready_nodes()] == ["H1"]
    graph.update_status("H1", HypothesisStatus.SUPPORTED)
    assert [item.id for item in graph.ready_nodes()] == ["H2"]
    graph.update_posterior("H2", 0.8)
    graph.update_evidence("H2", Evidence(source="unit", summary="evidence"))
    assert graph.get("H2").posterior == 0.8
    assert "H_H1 --> H_H2" in graph.to_mermaid()


def test_graph_rejects_unknown_dependencies() -> None:
    graph = HypothesisGraph()
    with pytest.raises(KeyError):
        graph.add_node(HypothesisState(id="H2", description="Invalid", dependencies=["H1"]))

