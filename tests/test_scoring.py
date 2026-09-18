"""Тесты единой оценки совпадений из двух источников."""

import pytest

from chgk_agent.search.lexical import Bm25Index, tokenize
from chgk_agent.search.scoring import (
    LEXICAL_KIND,
    NONE_KIND,
    SEMANTIC_KIND,
    SEMANTIC_LEXICAL_KIND,
    ScoredCandidate,
    build_index,
    cosine_similarity,
    describe_score_kind,
    score_candidates,
)


def _corpus() -> Bm25Index:
    """Небольшая коллекция документов для проверки BM25."""

    index = Bm25Index()
    index.add("local:1", "вопрос про галстук и моду")
    index.add("local:2", "вопрос про и в что он")
    for number in range(20):
        index.add(f"local:{number + 10}", "и в что он на и")
    return index.finalize()


def test_cosine_similarity_bounds_and_identity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_rare_term_scores_higher_than_common_term() -> None:
    """Редкий термин даёт больший вклад, чем служебное слово."""

    index = _corpus()

    rare = index.score(tokenize("галстук"), "local:1")
    common = index.score(tokenize("и"), "local:2")

    assert rare > common


def test_lexical_signal_identical_for_local_and_external_text() -> None:
    """Один и тот же текст получает один сигнал независимо от роли стороны."""

    description = "галстук"
    text = "вопрос про галстук и моду"
    base = _corpus()

    local = score_candidates(
        description,
        [ScoredCandidate(key="local:99", text=text, semantic_similarity=0.4)],
        base_index=base,
    )
    external = score_candidates(
        description,
        [ScoredCandidate(key="external:99", text=text, semantic_similarity=0.4)],
        base_index=base,
    )

    local_score = local["local:99"]
    external_score = external["external:99"]
    assert local_score.lexical_score == pytest.approx(external_score.lexical_score)
    assert local_score.score == pytest.approx(external_score.score)


def test_lexical_signal_uses_combined_collection() -> None:
    """Статистика считается по объединению локальных и внешних документов."""

    base = _corpus()
    with_card = score_candidates(
        "галстук",
        [
            ScoredCandidate(
                key="local:1", text="вопрос про галстук и моду", semantic_similarity=0.0
            ),
            ScoredCandidate(
                key="external:1", text="другой вопрос про галстук", semantic_similarity=0.0
            ),
        ],
        base_index=base,
    )

    assert with_card["local:1"].lexical_score > 0.0
    assert with_card["external:1"].lexical_score > 0.0


def test_score_stays_in_unit_range_with_weight() -> None:
    """Линейная комбинация с любым весом остаётся в диапазоне `[0, 1]`."""

    base = _corpus()
    candidates = [
        ScoredCandidate(key="local:1", text="вопрос про галстук", semantic_similarity=1.0),
        ScoredCandidate(key="local:2", text="вопрос про и", semantic_similarity=1.0),
    ]

    for weight in (0.0, 1.0, 5.0, 20.0):
        scores = score_candidates(
            "галстук и в что он",
            candidates,
            base_index=base,
            lexical_weight=weight,
            lexical_saturation=0.5,
        )
        assert all(0.0 <= item.score <= 1.0 for item in scores.values())


def test_lexical_contribution_raises_score_of_exact_match() -> None:
    """Совпадение по точному слову поднимает оценку выше похожего без него."""

    base = _corpus()
    scores = score_candidates(
        "галстук",
        [
            ScoredCandidate(
                key="local:1", text="вопрос про галстук и моду", semantic_similarity=0.4
            ),
            ScoredCandidate(
                key="local:2", text="вопрос про и в что он", semantic_similarity=0.4
            ),
        ],
        base_index=base,
        lexical_weight=1.0,
        lexical_saturation=1.0,
    )

    assert scores["local:1"].score > scores["local:2"].score
    assert scores["local:1"].score_kind == SEMANTIC_LEXICAL_KIND
    assert scores["local:2"].score_kind == SEMANTIC_KIND


def test_missing_index_degrades_to_semantic_only() -> None:
    """Без готового индекса лексика не подменяется статистикой по кандидатам."""

    scores = score_candidates(
        "галстук",
        [ScoredCandidate(key="a", text="вопрос про галстук", semantic_similarity=0.7)],
        base_index=None,
    )

    assert scores["a"].lexical_score == 0.0
    assert scores["a"].score == pytest.approx(0.7)
    assert scores["a"].score_kind == SEMANTIC_KIND


def test_build_index_extends_base_without_mutating_it() -> None:
    """Надстройка внешних карточек не меняет разделяемый кэш."""

    base = _corpus()
    size_before = base.size

    extended = build_index([("external:1", "карточка про галстук")], base=base)

    assert base.size == size_before
    assert extended.size == size_before + 1
    assert "external:1" in extended._documents


def test_describe_score_kind_names_contributions() -> None:
    assert describe_score_kind(0.5, 0.0) == SEMANTIC_KIND
    assert describe_score_kind(0.0, 0.3) == LEXICAL_KIND
    assert describe_score_kind(0.5, 0.3) == SEMANTIC_LEXICAL_KIND
    assert describe_score_kind(0.0, 0.0) == NONE_KIND
