from __future__ import annotations

from statagent.protocol.statistical_object import HypothesisState
from statagent.retrieval.embedding import HashingEmbedding
from statagent.retrieval.retriever import HierarchicalRetriever
from statagent.retrieval.vector_store import InMemoryVectorStore
from statagent.statistical_library import build_default_registry


def test_catalog_has_all_initial_methods_and_unique_apis() -> None:
    registry = build_default_registry()
    cards = registry.cards()
    assert len(cards) == 33
    assert len({card.method_id for card in cards}) == 33
    assert len({card.api for card in cards}) == 33
    assert all(card.references and card.inputs and card.outputs and card.tags for card in cards)
    assert not any(card.method_id.startswith(prefix) for card in cards for prefix in ("DESC", "REG", "DIM", "ML"))


def test_hierarchical_retrieval_returns_method_card_not_code() -> None:
    registry = build_default_registry()
    retriever = HierarchicalRetriever(HashingEmbedding(), InMemoryVectorStore())
    retriever.index(registry.cards())
    hypothesis = HypothesisState(
        id="H1",
        description="Compare the mean of two independent groups",
        metadata={"retrieval_terms": ["independent t-test", "two groups"]},
    )
    results = retriever.retrieve(hypothesis, category="参数检验", limit=20)
    assert results
    assert all(result.card.category[0] == "参数检验" for result in results)
    assert any(result.card.api == "run_ttest_independent" for result in results)
    result = next(result for result in results if result.card.api == "run_ttest_independent")
    assert not hasattr(result.card, "source_code")
