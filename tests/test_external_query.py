"""Тесты формирования коротких запросов для внешнего источника."""

import pytest

from chgk_agent.external.query import ShortQueryBuilder, build_short_queries

LONG_DESCRIPTION = (
    "В докладе 1947 года этот человек предложил использовать две комнаты, "
    "двух не очень сильных игроков в шахматы и оператора. Назовите этого человека."
)


def test_short_description_is_used_as_is() -> None:
    plan = build_short_queries("тест Тьюринга")

    assert plan.queries == ["тест Тьюринга"]
    assert plan.truncated is False
    assert plan.primary == "тест Тьюринга"


def test_description_at_exact_limit_is_not_truncated() -> None:
    description = "а" * 50

    plan = build_short_queries(description)

    assert plan.queries == [description]
    assert plan.truncated is False


def test_long_description_produces_short_queries() -> None:
    plan = build_short_queries(LONG_DESCRIPTION)

    assert plan.truncated is True
    assert plan.queries
    assert all(len(query) <= 50 for query in plan.queries)


def test_long_description_keeps_key_terms() -> None:
    plan = build_short_queries(LONG_DESCRIPTION)

    joined = " ".join(plan.queries).casefold()
    assert "1947" in joined
    assert "шахматы" in joined


def test_proper_nouns_are_prioritized() -> None:
    plan = build_short_queries(
        "Английский учёный прошлого века Ангус Бейтмен во время экспериментов "
        "давал ИМ клички: Щетинка, Лысый, Волосатое крыло. Назовите ИХ."
    )

    assert "Бейтмен" in plan.primary


def test_every_query_respects_the_limit() -> None:
    builder = ShortQueryBuilder(max_chars=30, max_queries=5)

    plan = builder.build(LONG_DESCRIPTION)

    assert plan.queries
    assert all(len(query) <= 30 for query in plan.queries)


def test_number_of_queries_is_limited() -> None:
    builder = ShortQueryBuilder(max_chars=12, max_queries=2)

    plan = builder.build(LONG_DESCRIPTION)

    assert len(plan.queries) <= 2


def test_stopwords_are_dropped() -> None:
    plan = build_short_queries(
        "И вот этот самый человек, который был там, назвал именно это слово. Назовите его."
    )

    assert plan.queries
    primary = plan.primary.casefold()
    assert "который" not in primary


def test_empty_description_gives_no_queries() -> None:
    plan = build_short_queries("   ")

    assert plan.queries == []
    assert plan.primary == ""


def test_single_long_token_is_not_sent_over_limit() -> None:
    plan = build_short_queries("ы" * 80)

    assert all(len(query) <= 50 for query in plan.queries)


@pytest.mark.parametrize("description", [LONG_DESCRIPTION, "коротко", "а" * 200])
def test_queries_never_exceed_limit(description: str) -> None:
    plan = build_short_queries(description)

    assert all(len(query) <= 50 for query in plan.queries)
