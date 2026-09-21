"""Тесты общей сборки лексического индекса и информативности терминов."""

from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from chgk_agent.search.corpus import ensure_corpus_index, local_key, term_informativeness
from chgk_agent.search.lexical import CorpusIndex, tokenize


@dataclass
class _Row:
    id: int
    question_text: str
    answer_text: str


class _FakeResult:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    def all(self) -> list[_Row]:
        return list(self._rows)


class _FakeSession:
    """Сессия, отдающая заранее заданный корпус и считающая обращения."""

    def __init__(self, rows: Iterable[_Row]) -> None:
        self._rows = list(rows)
        self.queries = 0

    async def execute(self, statement: object) -> _FakeResult:
        self.queries += 1
        return _FakeResult(self._rows)


async def test_index_is_built_once_and_shared_from_cache() -> None:
    """Обе стороны получают один индекс: база опрашивается единожды."""

    session = _FakeSession(
        [
            _Row(1, "Как называется галстук?", "Бабочка"),
            _Row(2, "Что носят на шее?", "Галстук"),
        ]
    )
    cache = CorpusIndex()

    first = await ensure_corpus_index(session, cache=cache)
    second = await ensure_corpus_index(session, cache=cache)

    assert first is second
    assert session.queries == 1
    assert cache.is_ready is True
    assert first.contains(local_key(1))


async def test_ready_index_does_not_touch_database() -> None:
    """Готовый индекс возвращается без обращения к базе."""

    session = _FakeSession([_Row(1, "вопрос", "ответ")])
    cache = CorpusIndex()
    cache.build([(local_key(1), "вопрос ответ")])

    index = await ensure_corpus_index(session, cache=cache)

    assert index is cache.index
    assert session.queries == 0


async def test_node_and_local_branch_share_the_same_cached_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Нода графа и локальная ветка получают один индекс из общего кэша."""

    from chgk_agent.db import repository
    from chgk_agent.search import nodes
    from chgk_agent.search.lexical import get_corpus_index

    cache = get_corpus_index()
    cache.invalidate()
    calls = 0

    async def counted_all_question_texts(session: object) -> list[tuple[int, str, str]]:
        nonlocal calls
        calls += 1
        return [(1, "Как называется галстук?", "Бабочка")]

    monkeypatch.setattr(repository, "all_question_texts", counted_all_question_texts)

    from_node = await nodes.build_corpus_index(_FakeSession([]))

    class UnusableSession:
        async def execute(self, statement: object) -> object:
            raise AssertionError("готовый индекс должен вернуться из кэша")

    from_local = await ensure_corpus_index(UnusableSession())  # type: ignore[arg-type]

    assert from_node is from_local
    assert calls == 1
    cache.invalidate()


async def test_term_informativeness_uses_corpus_stemming() -> None:
    """Информативность считается по основам, а не по словоформам."""

    session = _FakeSession(
        [_Row(1, "Как называется галстук?", "Бабочка"), _Row(2, "Вопрос", "Ответ")]
    )
    index = await ensure_corpus_index(session, cache=CorpusIndex())

    weights = term_informativeness(index, "галстуки и жираф")

    assert weights[tokenize("галстуки")[0]] == pytest.approx(
        index.idf(tokenize("галстук")[0])
    )
    assert weights[tokenize("жираф")[0]] == pytest.approx(index.idf(tokenize("жираф")[0]))
    assert weights[tokenize("жираф")[0]] > weights[tokenize("галстуки")[0]]
