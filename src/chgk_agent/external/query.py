"""Формирование коротких поисковых запросов для внешнего источника.

Сайт принимает не более 50 символов и считает слова запроса конъюнкцией,
поэтому длинное описание сжимается в несколько запросов по одному самому
информативному термину. Информативность приходит извне — как редкость основы в
локальном корпусе: корпус известен до обращения к сайту и отражает предметную
область, тогда как статистика внешней выдачи появилась бы только после ответа.
"""

import re
from dataclasses import dataclass

from chgk_agent.search.lexical import tokenize

WORD_RE = re.compile(r"[\w'’-]+", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")

BOILERPLATE = frozenset(
    {
        "назовите",
        "назвать",
        "ответ",
        "ответьте",
        "вопрос",
        "человек",
        "человека",
        "люди",
        "людей",
        "герой",
        "героя",
        "персонаж",
        "персонажа",
        "этот",
        "эта",
        "это",
        "эти",
        "тот",
        "та",
        "те",
        "который",
        "которая",
        "которое",
        "которые",
        "свой",
        "свои",
        "своей",
        "его",
        "ее",
        "их",
        "им",
        "них",
        "нам",
        "вам",
    }
)

STOPWORDS = frozenset(
    {
        "а",
        "без",
        "более",
        "больше",
        "будет",
        "будто",
        "бы",
        "был",
        "была",
        "были",
        "было",
        "быть",
        "в",
        "вам",
        "вас",
        "вдруг",
        "ведь",
        "весь",
        "во",
        "вот",
        "впрочем",
        "все",
        "всегда",
        "всего",
        "всех",
        "всю",
        "вся",
        "вы",
        "где",
        "да",
        "даже",
        "два",
        "для",
        "до",
        "другой",
        "его",
        "ее",
        "ей",
        "ему",
        "если",
        "есть",
        "еще",
        "ж",
        "же",
        "за",
        "зачем",
        "здесь",
        "и",
        "из",
        "или",
        "им",
        "иногда",
        "их",
        "к",
        "как",
        "какая",
        "какой",
        "когда",
        "конечно",
        "которое",
        "который",
        "которые",
        "которая",
        "кто",
        "куда",
        "ли",
        "лучше",
        "между",
        "меня",
        "мне",
        "много",
        "мой",
        "моя",
        "мы",
        "на",
        "над",
        "надо",
        "наконец",
        "нас",
        "не",
        "него",
        "нее",
        "ней",
        "нельзя",
        "нет",
        "ни",
        "нибудь",
        "никогда",
        "ним",
        "них",
        "ничего",
        "но",
        "ну",
        "о",
        "об",
        "один",
        "он",
        "она",
        "они",
        "опять",
        "от",
        "перед",
        "по",
        "под",
        "после",
        "потом",
        "потому",
        "почти",
        "при",
        "про",
        "раз",
        "разве",
        "с",
        "сам",
        "свой",
        "свои",
        "свою",
        "себе",
        "себя",
        "сейчас",
        "со",
        "совсем",
        "так",
        "также",
        "такой",
        "там",
        "тебя",
        "тем",
        "теперь",
        "то",
        "тогда",
        "того",
        "тоже",
        "том",
        "тот",
        "три",
        "тут",
        "ты",
        "у",
        "уж",
        "уже",
        "хоть",
        "хорошо",
        "чего",
        "чем",
        "через",
        "что",
        "чтоб",
        "чтобы",
        "чуть",
        "эти",
        "этих",
        "этого",
        "этой",
        "этом",
        "этот",
        "эту",
        "это",
        "я",
    }
)


@dataclass(slots=True)
class QueryPlan:
    """План коротких запросов для внешнего источника."""

    queries: list[str]
    truncated: bool

    @property
    def primary(self) -> str:
        """Первый (основной) запрос."""

        return self.queries[0] if self.queries else ""


def _terms(text: str) -> list[tuple[str, bool]]:
    """Извлечь термины описания вместе с признаком начала предложения."""

    terms: list[tuple[str, bool]] = []
    for sentence in SENTENCE_SPLIT_RE.split(text):
        for index, match in enumerate(WORD_RE.finditer(sentence)):
            terms.append((match.group(0), index == 0))
    return terms


def _is_keyword(token: str, *, starts_sentence: bool) -> bool:
    """Похож ли термин на имя собственное, аббревиатуру или число.

    Заглавная буква в начале предложения — обычная орфография, а не признак
    имени собственного, поэтому такие слова не получают повышенный вес.
    """

    if any(character.isdigit() for character in token):
        return True
    if token.isupper() and len(token) > 1:
        return True
    return token[0].isupper() and not starts_sentence


def _meaningful_terms(
    tokens: list[tuple[str, bool]],
) -> list[tuple[int, str, str, bool]]:
    """Отобрать знаменательные термины вместе с порядком и основой.

    Стоп-слова и типовые формулировки отбрасываются до всякого отбора: IDF сам
    их не уберёт, потому что при коротком описании служебное слово может
    оказаться редким относительно длины текста. Повторные словоформы
    схлопываются по основе, чтобы один корень не занимал весь бюджет запроса.
    """

    terms: list[tuple[int, str, str, bool]] = []
    seen: set[str] = set()
    for index, (token, starts_sentence) in enumerate(tokens):
        key = token.casefold()
        if key in STOPWORDS or key in BOILERPLATE or len(token) < 3:
            continue
        stem = tokenize(token)[0] if tokenize(token) else key
        if stem in seen:
            continue
        seen.add(stem)
        terms.append((index, token, stem, starts_sentence))
    return terms


def _rank_terms(
    terms: list[tuple[int, str, str, bool]],
    *,
    term_weights: dict[str, float] | None,
) -> list[tuple[int, str]]:
    """Упорядочить термины по информативности, затем по появлению в тексте.

    При доступных весах информативность — редкость основы в локальном корпусе;
    без них — прежнее правило: имя собственное, число или аббревиатура, затем
    длина слова. Порядок появления разрешает ничьи: при `df = 0` величина
    одинакова у всех терминов вне корпуса.
    """

    def rank(entry: tuple[int, str, str, bool]) -> tuple[float, float, int]:
        index, token, stem, starts_sentence = entry
        if term_weights is not None:
            weight = term_weights.get(stem)
            return (-(weight if weight is not None else 0.0), 0.0, index)
        priority = 2.0 if _is_keyword(token, starts_sentence=starts_sentence) else 1.0
        return (-priority, -float(len(token)), index)

    return [(index, token) for index, token, _, _ in sorted(terms, key=rank)]


class ShortQueryBuilder:
    """Превращает описание в короткие запросы по одному термину.

    Сайт считает слова запроса конъюнкцией, поэтому многотёрмовый запрос сужает
    выдачу вплоть до пустой. Каждый запрос несёт ровно один самый информативный
    термин, а число запросов ограничивается бюджетом.
    """

    def __init__(
        self,
        *,
        max_chars: int = 50,
        max_queries: int = 3,
    ) -> None:
        self._max_chars = max_chars
        self._max_queries = max(max_queries, 1)

    @property
    def max_chars(self) -> int:
        """Максимальная длина одного запроса."""

        return self._max_chars

    @property
    def max_queries(self) -> int:
        """Сколько коротких запросов разрешено на одно описание."""

        return self._max_queries

    def build(
        self,
        description: str,
        *,
        term_weights: dict[str, float] | None = None,
    ) -> QueryPlan:
        """Построить план коротких запросов по описанию."""

        text = " ".join(description.split())
        if not text:
            return QueryPlan(queries=[], truncated=False)
        if len(text) <= self._max_chars:
            return QueryPlan(queries=[text], truncated=False)

        ranked = _rank_terms(
            _meaningful_terms(_terms(text)),
            term_weights=term_weights,
        )
        queries: list[str] = []
        for _index, token in ranked:
            if len(token) > self._max_chars:
                continue
            queries.append(token)
            if len(queries) >= self._max_queries:
                break

        if not queries:
            fallback = " ".join(token for token, _ in _terms(text))[: self._max_chars].strip()
            queries = [fallback] if fallback else []

        return QueryPlan(queries=queries, truncated=True)


def build_short_queries(
    description: str,
    *,
    max_chars: int = 50,
    max_queries: int = 3,
    term_weights: dict[str, float] | None = None,
) -> QueryPlan:
    """Собрать короткие запросы для внешнего источника."""

    return ShortQueryBuilder(max_chars=max_chars, max_queries=max_queries).build(
        description, term_weights=term_weights
    )


__all__ = ["ShortQueryBuilder", "build_short_queries"]
