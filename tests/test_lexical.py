"""Тесты лексического сигнала BM25."""

import pytest

from chgk_agent.search.lexical import Bm25Index, normalize_bm25, tokenize


def test_stemming_unifies_word_forms() -> None:
    """Разные формы одного слова дают одну основу."""

    stems = {tokenize(word)[0] for word in ("галстуки", "галстуков", "галстук")}

    assert len(stems) == 1


def test_stemming_keeps_unrelated_words_apart() -> None:
    """Стеммер не сводит разные слова к одной основе."""

    assert tokenize("учёного")[0] != tokenize("ученики")[0]


def test_tokenize_is_case_insensitive() -> None:
    assert tokenize("ГАЛСТУК") == tokenize("галстук")


def test_tokenize_ignores_punctuation() -> None:
    assert tokenize("галстук, и ещё — раз!") == ["галстук", "и", "ещ", "раз"]


def test_rare_term_scores_higher_than_common_term() -> None:
    """Редкий термин весит больше распространённого."""

    index = Bm25Index()
    index.add("a", "галстук")
    for number in range(50):
        index.add(f"c{number}", "и в что он")
    index.finalize()

    rare = index.score(tokenize("галстук"), "a")
    index.add("b", "и в что он")
    index.finalize()
    common = index.score(tokenize("и"), "b")

    assert rare > common


def test_score_is_zero_for_absent_term() -> None:
    index = Bm25Index()
    index.add("a", "мост через реку")

    assert index.score(tokenize("жираф"), "a") == 0.0


def test_score_is_zero_for_unknown_document() -> None:
    index = Bm25Index()
    index.add("a", "мост через реку")

    assert index.score(tokenize("мост"), "missing") == 0.0


def test_scores_returns_only_matching_documents() -> None:
    index = Bm25Index()
    index.add("a", "мост через реку")
    index.add("b", "жираф в зоопарке")
    index.add("c", "мост и река")

    scored = index.scores(tokenize("мост"))

    assert set(scored) == {"a", "c"}
    assert all(value > 0 for value in scored.values())


def test_more_occurrences_score_higher() -> None:
    """Повторение термина повышает оценку при прочих равных."""

    index = Bm25Index()
    index.add("once", "мост")
    index.add("twice", "мост мост")

    scored = index.scores(tokenize("мост"))

    assert scored["twice"] > scored["once"]


def test_length_normalization_penalizes_long_documents() -> None:
    """Длинный документ с одним совпадением проигрывает короткому."""

    index = Bm25Index()
    index.add("short", "мост")
    index.add("long", "мост " + " ".join(f"слово{n}" for n in range(40)))

    scored = index.scores(tokenize("мост"))

    assert scored["short"] > scored["long"]


def test_duplicate_document_is_ignored() -> None:
    index = Bm25Index()
    index.add("a", "мост")
    index.add("a", "река")

    assert index.size == 1
    assert index.score(tokenize("мост"), "a") > 0
    assert index.score(tokenize("река"), "a") == 0.0


def test_index_reports_size_and_average_length() -> None:
    index = Bm25Index()
    index.add("a", "один два")
    index.add("b", "три")

    assert index.size == 2
    assert index.average_length == pytest.approx(1.5)


def test_normalize_bm25_is_bounded_and_monotonic() -> None:
    values = [normalize_bm25(value, saturation=10.0) for value in (0.0, 1.0, 10.0, 100.0)]

    assert values[0] == 0.0
    assert values == sorted(values)
    assert all(0.0 <= value < 1.0 for value in values)


def test_normalize_bm25_handles_zero_saturation() -> None:
    assert normalize_bm25(5.0, saturation=0.0) == pytest.approx(1.0)


def test_collection_statistics_span_all_documents() -> None:
    """Редкость термина считается по всей коллекции, а не по одному документу."""

    index = Bm25Index()
    index.add("a", "галстук")
    for number in range(20):
        index.add(f"d{number}", f"слово{number}")

    scored = index.scores(tokenize("галстук"))

    assert set(scored) == {"a"}


def test_corpus_index_builds_once_and_is_reused() -> None:
    """Индекс строится один раз и переиспользуется между запросами."""

    from chgk_agent.search.lexical import CorpusIndex

    cache = CorpusIndex()
    calls = 0

    def documents():
        nonlocal calls
        calls += 1
        yield "a", "мост"
        yield "b", "река"

    first = cache.ensure(documents())
    second = cache.ensure(documents())

    assert first is second
    assert calls == 1
    assert cache.is_ready is True


def test_corpus_index_rebuilds_after_invalidation() -> None:
    """После изменения корпуса индекс перестраивается."""

    from chgk_agent.search.lexical import CorpusIndex

    cache = CorpusIndex()
    cache.ensure([("a", "мост")])

    cache.invalidate()

    assert cache.is_ready is False
    rebuilt = cache.ensure([("a", "мост"), ("b", "река")])
    assert rebuilt.size == 2


def test_corpus_index_without_documents_is_not_ready() -> None:
    from chgk_agent.search.lexical import CorpusIndex

    cache = CorpusIndex()

    assert cache.is_ready is False
    assert cache.index is None


def test_copy_with_does_not_mutate_original() -> None:
    """Карточки внешнего источника не меняют разделяемый индекс."""

    index = Bm25Index()
    index.add("a", "мост")

    clone = index.copy_with([("ext-1", "галстук")])

    assert index.score(tokenize("галстук"), "ext-1") == 0.0
    assert clone.score(tokenize("галстук"), "ext-1") > 0
    assert clone.size == index.size + 1


def test_copy_with_extends_collection_statistics() -> None:
    """Дополнительные документы участвуют в статистике коллекции."""

    index = Bm25Index()
    for number in range(10):
        index.add(f"a{number}", f"слово{number}")

    clone = index.copy_with([("ext-1", "галстук")])

    assert clone.score(tokenize("галстук"), "ext-1") > 0
    assert clone.size == 11
