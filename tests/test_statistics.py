from __future__ import annotations

from statagent.api.dispatcher import StatisticalDispatcher
from statagent.statistical_library import build_default_registry


def test_independent_ttest_via_registered_dispatcher() -> None:
    registry = build_default_registry()
    dispatcher = StatisticalDispatcher(registry)
    result = dispatcher.execute(
        registry.card("PAR001"),
        {"group_a": [1, 2, 3, 4, 5], "group_b": [10, 11, 12, 13, 14]},
    )
    assert result.p_value is not None and result.p_value < 0.001
    assert result.effect_size and result.effect_size["name"] == "Cohen's d"
    assert result.reproducibility["method_card_version"] == "2.0.0"


def test_distribution_test_is_reproducible() -> None:
    registry = build_default_registry()
    dispatcher = StatisticalDispatcher(registry)
    result = dispatcher.execute(registry.card("DIST001"), {"values": [1, 2, 3, 4, 5, 6]})
    assert result.estimates["n"] == 6
    assert result.p_value is not None


def test_image_catalog_methods_execute_with_unified_results() -> None:
    registry = build_default_registry()
    dispatcher = StatisticalDispatcher(registry)
    checks = [
        ("DIST003", {"x": [[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5], [7, 8], [8, 7]]}, {"bootstrap_iterations": 9}),
        ("VAR003", {"groups": [[1, 2, 3], [2, 3, 4], [1, 2, 4]]}, {}),
        ("PAR008", {"groups": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, {}),
        ("NONPAR007", {"groups": [[1, 2, 3], [2, 3, 4], [3, 4, 5]]}, {}),
        ("CAT003", {"row": ["a", "a", "b", "b", "a", "b", "a", "b"], "column": ["x", "y", "x", "y", "x", "y", "y", "x"], "strata": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"]}, {}),
        ("COR002", {"x": [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]}, {}),
    ]
    for method_id, data, parameters in checks:
        result = dispatcher.execute(registry.card(method_id), data, parameters)
        assert result.p_value is not None
        assert result.reproducibility["method_card_version"] == "2.0.0"
