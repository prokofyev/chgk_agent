"""Формирование коротких поисковых запросов для внешнего источника.

Сайт принимает не более 50 символов: более длинный запрос молча
отклоняется, поэтому длинное описание превращается в один или несколько
запросов из наиболее информативных терминов.
"""

import re
from dataclasses import dataclass

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


def _rank_terms(tokens: list[tuple[str, bool]]) -> list[str]:
    """Упорядочить уникальные термины по информативности.

    Сначала идут имена собственные, числа и аббревиатуры, затем длинные
    знаменательные слова, затем всё остальное. Стоп-слова и типовые
    формулировки вопросов ЧГК отбрасываются: они не помогают поиску.
    """

    ranked: list[tuple[int, int, int, str]] = []
    seen: set[str] = set()
    for index, (token, starts_sentence) in enumerate(tokens):
        key = token.casefold()
        if key in seen or key in STOPWORDS or key in BOILERPLATE or len(token) < 3:
            continue
        seen.add(key)
        priority = 2 if _is_keyword(token, starts_sentence=starts_sentence) else 1
        ranked.append((priority, len(token), -index, token))

    ranked.sort(reverse=True)
    return [token for _, _, _, token in ranked]


def _chunk_terms(tokens: list[str], max_chars: int) -> list[str]:
    """Уложить термины в запросы длиной не более `max_chars`."""

    queries: list[str] = []
    current: list[str] = []
    length = 0

    for token in tokens:
        if len(token) > max_chars:
            continue
        extra = len(token) if not current else len(token) + 1
        if length + extra > max_chars:
            if current:
                queries.append(" ".join(current))
            current = [token]
            length = len(token)
        else:
            current.append(token)
            length += extra

    if current:
        queries.append(" ".join(current))
    return queries


class ShortQueryBuilder:
    """Превращает описание в один или несколько коротких запросов."""

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

    def build(self, description: str) -> QueryPlan:
        """Построить план коротких запросов по описанию."""

        text = " ".join(description.split())
        if not text:
            return QueryPlan(queries=[], truncated=False)
        if len(text) <= self._max_chars:
            return QueryPlan(queries=[text], truncated=False)

        queries = _chunk_terms(_rank_terms(_terms(text)), self._max_chars)
        queries = [query for query in queries if query][: self._max_queries]

        if not queries:
            fallback = " ".join(token for token, _ in _terms(text))[: self._max_chars].strip()
            queries = [fallback] if fallback else []

        return QueryPlan(queries=queries, truncated=True)


def build_short_queries(
    description: str,
    *,
    max_chars: int = 50,
    max_queries: int = 3,
) -> QueryPlan:
    """Собрать короткие запросы для внешнего источника."""

    return ShortQueryBuilder(max_chars=max_chars, max_queries=max_queries).build(description)
