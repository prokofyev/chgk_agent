"""Тесты reciprocal rank fusion."""

import pytest

from chgk_agent.search.rrf import RankedItem, ranks_of, reciprocal_rank_fusion


def test_fusion_of_single_branch_normalizes_to_one() -> None:
    fused = reciprocal_rank_fusion(
        [[RankedItem(key="a"), RankedItem(key="b"), RankedItem(key="c")]]
    )

    assert fused["a"] == pytest.approx(1.0)
    assert fused["a"] > fused["b"] > fused["c"]
    assert all(0.0 <= score <= 1.0 for score in fused.values())


def test_item_present_in_both_branches_wins() -> None:
    fused = reciprocal_rank_fusion(
        [
            [RankedItem(key="a"), RankedItem(key="b")],
            [RankedItem(key="b"), RankedItem(key="c")],
        ]
    )

    assert max(fused, key=fused.get) == "b"


def test_explicit_rank_is_respected() -> None:
    fused = reciprocal_rank_fusion(
        [[RankedItem(key="a", rank=5), RankedItem(key="b", rank=1)]]
    )

    assert fused["b"] > fused["a"]


def test_weights_shift_the_fusion() -> None:
    branches = [
        [RankedItem(key="a"), RankedItem(key="b")],
        [RankedItem(key="b"), RankedItem(key="a")],
    ]

    second_branch_heavy = reciprocal_rank_fusion(branches, weights=[0.1, 1.0])
    first_branch_heavy = reciprocal_rank_fusion(branches, weights=[1.0, 0.1])

    assert max(second_branch_heavy, key=second_branch_heavy.get) == "b"
    assert max(first_branch_heavy, key=first_branch_heavy.get) == "a"


def test_empty_branches_return_empty_mapping() -> None:
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[]]) == {}


def test_scores_stay_within_bounds() -> None:
    fused = reciprocal_rank_fusion(
        [
            [RankedItem(key=f"a{index}") for index in range(20)],
            [RankedItem(key=f"b{index}") for index in range(20)],
        ]
    )

    assert all(0.0 <= score <= 1.0 for score in fused.values())


def test_ranks_of_returns_positions() -> None:
    items = [RankedItem(key="x"), RankedItem(key="y", rank=7)]

    assert ranks_of(items) == {"x": 1, "y": 7}
