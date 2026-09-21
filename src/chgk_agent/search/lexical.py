"""Лексический сигнал BM25 по тексту «вопрос + ответ».

BM25 учитывает частоту термина, длину документа и редкость термина в
коллекции. Последнее и есть причина, по которой он выбран вместо
полнотекстового ранга и покрытия терминов: редкое слово в описании должно
весить на порядок больше служебного.

Коллекция статистики общая для обоих источников поиска, поэтому индекс по
локальному корпусу живёт здесь и переиспользуется между запросами, а карточки
внешнего источника добавляются к нему как надстройка на время запроса.
"""

import math
import re
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import snowballstemmer

QUERY_KIND = "bm25"

WORD_RE = re.compile(r"[а-яёa-z0-9]+")

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


class _Stemmer:
    """Ленивая обёртка над стеммером русского языка.

    Стеммер Snowball не потокобезопасен в части инициализации таблиц, поэтому
    создаётся один раз под замком.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stemmer = None

    def stem(self, word: str) -> str:
        """Вернуть основу слова."""

        if self._stemmer is None:
            with self._lock:
                if self._stemmer is None:
                    self._stemmer = snowballstemmer.stemmer("russian")
        return self._stemmer.stemWord(word)


_STEMMER = _Stemmer()


def tokenize(text: str) -> list[str]:
    """Разбить текст на основы слов.

    Токенизация одинакова для описания, локальных вопросов и карточек внешнего
    источника: расхождение в предобработке сделало бы оценки несопоставимыми.
    """

    return [_STEMMER.stem(word) for word in WORD_RE.findall(text.casefold())]


@dataclass(slots=True)
class Bm25Index:
    """Индекс BM25 по набору документов.

    Документ — это текст «вопрос + ответ». Статистика коллекции (`N`, `df`,
    `avgdl`) считается по всем документам индекса, а не по выдаче одного
    запроса.
    """

    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    _documents: dict[str, Counter[str]] = field(default_factory=dict)
    _lengths: dict[str, int] = field(default_factory=dict)
    _df: Counter[str] = field(default_factory=Counter)
    _total_length: int = 0
    _finalized: bool = False

    @property
    def size(self) -> int:
        """Число документов в индексе."""

        return len(self._documents)

    @property
    def average_length(self) -> float:
        """Средняя длина документа в основах."""

        size = len(self._documents)
        return self._total_length / size if size else 0.0

    def add(self, key: str, text: str) -> None:
        """Добавить документ в индекс."""

        if key in self._documents:
            return
        terms = tokenize(text)
        term_frequencies = Counter(terms)
        self._documents[key] = term_frequencies
        self._lengths[key] = len(terms)
        self._total_length += len(terms)
        self._finalized = False
        for term in term_frequencies:
            self._df[term] += 1

    def extend(self, documents: Iterable[tuple[str, str]]) -> None:
        """Добавить несколько документов."""

        for key, text in documents:
            self.add(key, text)

    def finalize(self) -> "Bm25Index":
        """Отметить индекс готовым к скорингу."""

        self._finalized = True
        return self

    def copy_with(self, documents: Iterable[tuple[str, str]]) -> "Bm25Index":
        """Вернуть копию индекса с дополнительными документами.

        Карточки внешнего источника добавляются к локальному индексу на время
        запроса, не изменяя разделяемый кэш: иначе выдача одного поиска влияла
        бы на статистику следующего.
        """

        clone = Bm25Index(k1=self.k1, b=self.b)
        clone._documents = dict(self._documents)
        clone._lengths = dict(self._lengths)
        clone._df = Counter(self._df)
        clone._total_length = self._total_length
        clone._finalized = self._finalized
        clone.extend(documents)
        return clone

    def contains(self, key: str) -> bool:
        """Есть ли документ в индексе."""

        return key in self._documents

    def document_frequency(self, term: str) -> int:
        """Сколько документов коллекции содержат основу `term`."""

        return self._df.get(term, 0)

    def idf(self, term: str) -> float:
        """Информативность основы по её документной частоте в коллекции.

        Термин вне корпуса получает максимальное значение: имя собственное,
        которого нет в базе, — сильный сигнал, и приравнивать его к служебному
        слову значило бы выбрасывать лучшую часть описания.
        """

        return _idf(self.size, self.document_frequency(term))

    def score(self, query_terms: Sequence[str], key: str) -> float:
        """Оценить документ по терминам запроса."""

        if not self._finalized:
            self.finalize()

        term_frequencies = self._documents.get(key)
        if not term_frequencies:
            return 0.0

        size = len(self._documents)
        length = self._lengths[key]
        average = self.average_length or 1.0
        total = 0.0

        for term in query_terms:
            frequency = term_frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = self._df.get(term, 0)
            idf = _idf(size, document_frequency)
            if idf <= 0:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / average
            )
            total += idf * frequency * (self.k1 + 1) / denominator

        return total

    def scores(self, query_terms: Sequence[str]) -> dict[str, float]:
        """Оценить все документы индекса и вернуть ненулевые оценки."""

        scored: dict[str, float] = {}
        for key in self._documents:
            value = self.score(query_terms, key)
            if value > 0:
                scored[key] = value
        return scored


def _idf(size: int, document_frequency: int) -> float:
    """Обратная документная частота в неотрицательной форме."""

    return max(0.0, math.log(1 + (size - document_frequency + 0.5) / (document_frequency + 0.5)))


def normalize_bm25(score: float, *, saturation: float) -> float:
    """Привести BM25 к диапазону `[0, 1)` насыщением.

    Деление на максимум по выдаче не используется: максимум зависел бы от
    состава кандидатов, то есть от того, сколько результатов вернул каждый
    источник, и баланс источников стал бы произвольным.
    """

    if score <= 0:
        return 0.0
    bound = max(saturation, 1e-9)
    return score / (score + bound)


class CorpusIndex:
    """Кэш BM25-индекса по локальному корпусу с ручной инвалидацией.

    Построение индекса по всему корпусу стоит сотни миллисекунд, поэтому он
    строится один раз и переиспользуется между запросами. Импорт и
    переиндексация меняют состав вопросов и их тексты, поэтому после них индекс
    инвалидируется явно: определять устаревание запросом к базе на каждый поиск
    дороже, чем перестроить индекс после изменения.
    """

    def __init__(self, *, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self._k1 = k1
        self._b = b
        self._index: Bm25Index | None = None
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        """Построен ли индекс."""

        return self._index is not None

    @property
    def index(self) -> Bm25Index | None:
        """Текущий индекс или `None`, если он ещё не построен."""

        return self._index

    def invalidate(self) -> None:
        """Сбросить индекс: следующий запрос перестроит его."""

        with self._lock:
            self._index = None

    def build(self, documents: Iterable[tuple[str, str]]) -> Bm25Index:
        """Построить индекс по документам `ключ -> текст`."""

        index = Bm25Index(k1=self._k1, b=self._b)
        index.extend(documents)
        index.finalize()
        with self._lock:
            self._index = index
        return index

    def ensure(
        self, documents: Iterable[tuple[str, str]]
    ) -> Bm25Index:
        """Вернуть готовый индекс, построив его при необходимости."""

        current = self._index
        if current is not None:
            return current
        return self.build(documents)


_DEFAULT_CORPUS_INDEX = CorpusIndex()


def get_corpus_index() -> CorpusIndex:
    """Вернуть общий кэш индекса локального корпуса."""

    return _DEFAULT_CORPUS_INDEX


__all__ = [
    "DEFAULT_B",
    "DEFAULT_K1",
    "QUERY_KIND",
    "Bm25Index",
    "CorpusIndex",
    "get_corpus_index",
    "normalize_bm25",
    "tokenize",
]
