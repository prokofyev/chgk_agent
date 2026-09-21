"""Тесты формирования коротких запросов для внешнего источника."""

import pytest

from chgk_agent.external.query import ShortQueryBuilder, build_short_queries
from chgk_agent.search.lexical import tokenize

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


def test_long_description_produces_one_term_per_query() -> None:
    """Каждый запрос несёт ровно один термин: сайт считает слова конъюнкцией."""

    plan = build_short_queries(LONG_DESCRIPTION)

    assert plan.truncated is True
    assert plan.queries
    assert all(" " not in query for query in plan.queries)
    assert all(len(query) <= 50 for query in plan.queries)


def test_rare_term_beats_long_common_word() -> None:
    """Редкое короткое слово вытесняет частое длинное."""

    description = "жираф альфа бета гамма дельта стеклоочиститель дополнительно"
    weights = {
        tokenize("жираф")[0]: 5.0,
        tokenize("альфа")[0]: 3.0,
        tokenize("бета")[0]: 3.0,
        tokenize("гамма")[0]: 3.0,
        tokenize("дельта")[0]: 3.0,
        tokenize("стеклоочиститель")[0]: 0.1,
    }

    plan = build_short_queries(
        description, max_queries=1, term_weights=weights
    )

    assert plan.queries == ["жираф"]


def test_number_of_requests_is_limited() -> None:
    """Число запросов не превышает настроенный бюджет."""

    builder = ShortQueryBuilder(max_chars=50, max_queries=2)

    plan = builder.build(LONG_DESCRIPTION)

    assert len(plan.queries) == 2
    assert builder.max_queries == 2


def test_most_informative_terms_are_chosen_first() -> None:
    """В бюджет попадают самые информативные термины."""

    description = "жираф альфа бета гамма дельта эпсилон дзета эта тета йота каппа"
    weights = {
        tokenize("жираф")[0]: 1.0,
        tokenize("альфа")[0]: 5.0,
        tokenize("бета")[0]: 4.0,
        tokenize("гамма")[0]: 0.5,
    }

    plan = build_short_queries(description, max_queries=2, term_weights=weights)

    assert plan.queries == ["альфа", "бета"]


def test_terms_outside_corpus_are_informative() -> None:
    """Без весов запрос всё равно формируется и не пуст."""

    description = "Неизвестное описание с редкими словами " * 3

    plan = build_short_queries(description)

    assert plan.queries
    assert all(query for query in plan.queries)


def test_proper_nouns_are_prioritized_without_weights() -> None:
    """Без весов имя собственное обходит более длинное знаменательное слово."""

    plan = build_short_queries(
        "Обычное длинное описание про необычный галстук и шляпу. Ангус Бейтмен.",
        max_queries=1,
    )

    assert plan.primary == "Бейтмен"


def test_stopwords_are_dropped() -> None:
    plan = build_short_queries(
        "И вот этот самый человек, который был там, назвал именно это слово. Назовите его."
    )

    joined = " ".join(plan.queries).casefold()
    assert "который" not in joined


def test_repeated_stem_is_not_duplicated() -> None:
    """Повтор основы не занимает бюджет запросов дважды."""

    description = "Галстук галстуки галстуков и шляпа " * 3
    weights = {tokenize("галстуки")[0]: 5.0, tokenize("шляпа")[0]: 4.0}

    plan = build_short_queries(description, max_queries=2, term_weights=weights)

    stems = [tokenize(query)[0] for query in plan.queries]
    assert len(stems) == len(set(stems))


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
