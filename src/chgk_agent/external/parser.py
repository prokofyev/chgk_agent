"""Разбор HTML-выдачи и страницы вопроса gotquestions.online.

Разметка сайта неофициальная, поэтому парсер устроен консервативно: он
опирается на устойчивые признаки (ссылка `/question/<id>`, блок ответа)
и различает три разных «нет результатов», а не сводит их к пустому списку.
"""

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

QUESTION_LINK_RE = re.compile(r"^/question/(?P<id>\d+)$")
EMPTY_MARKERS = ("Ничего не найдено", "ничего не найдено", "По вашему запросу ничего не найдено")
RESULTS_MARKERS = ("Найдено вопросов", "Найдено")
ANSWER_LABELS = ("Ответ:", "Ответ :")
COMMENT_LABELS = ("Комментарий:", "Комментарии:")
SOURCE_LABELS = ("Источник:", "Источники:")
PASS_LABELS = ("Зачет:", "Зачёт:")

COUNTER_RE = re.compile(
    r"Найдено вопросов\s*(?:Over 9000!)?\s*(?P<total>[\d\s]+?)\s*"
    r"(?P<start>\d+)\s*·\s*(?P<end>\d+)"
)
QUERY_RE = re.compile(
    r"Запрос\s*:\s*(.+?)\s*"
    r"(?:На странице|Вопросы|Пакеты|Персоны|Ничего|Найдено|Показать)"
)
PAGE_SIZE_RE = re.compile(r"На странице\s*:\s*(\d+)")
SCAFFOLD_MARKERS = ("Запрос :", "На странице :", "Вопросы Пакеты Персоны", "Фильтры")

LABEL_TOKENS = frozenset(
    {
        "Ответ:",
        "Зачет:",
        "Зачёт:",
        "Комментарий:",
        "Источник:",
        "Источники:",
        "Вопрос",
        "Тур",
        "Скрыть",
        "ответ",
        "expand_less",
        "thumb_up",
        "thumb_down",
        "bookmark",
        "visibility_off",
    }
)


@dataclass(slots=True)
class ParsedExternalMatch:
    """Разобранная карточка вопроса из выдачи."""

    external_id: str
    external_url: str
    title: str
    question_text: str
    answer_text: str | None = None
    comment: str | None = None
    pass_criteria: str | None = None
    sources: str | None = None
    authors: list[str] = field(default_factory=list)
    pack: str | None = None
    position: int = 0


@dataclass(slots=True)
class ParsedSearchPage:
    """Результат разбора страницы выдачи."""

    matches: list[ParsedExternalMatch] = field(default_factory=list)
    has_results_block: bool = False
    has_empty_marker: bool = False
    query: str | None = None
    total: int | None = None
    page_size: int | None = None
    has_more: bool = False
    has_search_scaffold: bool = False
    recognized: bool = True


def parse_search_page(html: str) -> ParsedSearchPage:
    """Разобрать страницу выдачи поиска.

    `recognized=False` означает, что страница вообще не похожа на страницу
    поиска: это сетевая или серверная аномалия, а не результат. Отдельно
    возвращается `has_search_scaffold` (страница поиска узнана) и
    `has_results_block` (есть счётчик найденных вопросов): сочетание этих
    признаков позволяет отличить отвергнутый запрос от изменившейся разметки.
    """

    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    page = ParsedSearchPage()
    page.has_empty_marker = any(marker in text for marker in EMPTY_MARKERS)
    page.has_search_scaffold = any(marker in text for marker in SCAFFOLD_MARKERS)

    counter = COUNTER_RE.search(text)
    if counter:
        page.total = int("".join(counter.group("total").split()))
        page.has_more = int(counter.group("end")) < page.total
        page.has_results_block = True

    page_size = PAGE_SIZE_RE.search(text)
    if page_size:
        page.page_size = int(page_size.group(1))

    query_match = QUERY_RE.search(text)
    if query_match:
        page.query = query_match.group(1).strip()

    cards = _find_cards(soup)
    for position, card in enumerate(cards, start=1):
        match = _parse_card(card, position=position)
        if match is not None:
            page.matches.append(match)

    page.recognized = bool(cards) or page.has_empty_marker or page.has_search_scaffold
    return page


def _find_cards(soup: BeautifulSoup) -> list[Tag]:
    """Найти карточки вопросов по ссылкам вида `/question/<id>`."""

    cards: list[Tag] = []
    seen: set[str] = set()
    for link in soup.select('a[href^="/question/"]'):
        match = QUESTION_LINK_RE.match(link.get("href", ""))
        if match is None:
            continue
        question_id = match.group("id")
        if question_id in seen:
            continue

        card = _card_root(link)
        if card is None:
            continue
        seen.add(question_id)
        cards.append(card)
    return cards


def _card_root(link: Tag) -> Tag | None:
    """Подняться от ссылки до контейнера карточки вопроса."""

    node: Tag | None = link
    best: Tag | None = None
    for _ in range(8):
        if node is None:
            break
        if node.name == "div" and str(node.get("id", "")).isdigit():
            return node
        node = node.parent
    return best


def _parse_card(card: Tag, *, position: int) -> ParsedExternalMatch | None:
    """Разобрать одну карточку вопроса."""

    link = card.select_one('a[href^="/question/"]')
    if link is None:
        return None
    match = QUESTION_LINK_RE.match(link.get("href", ""))
    if match is None:
        return None

    question_id = match.group("id")
    block = _answer_block(card)
    question_text = _question_text(card, block)
    if not question_text:
        return None

    return ParsedExternalMatch(
        external_id=question_id,
        external_url=f"https://gotquestions.online/question/{question_id}",
        title=link.get_text(" ", strip=True),
        question_text=question_text,
        answer_text=_labeled_value(block, ANSWER_LABELS),
        comment=_labeled_value(block, COMMENT_LABELS),
        pass_criteria=_labeled_value(block, PASS_LABELS),
        sources=_labeled_value(block, SOURCE_LABELS),
        authors=_authors(card),
        pack=_pack(card),
        position=position,
    )


def _answer_block(card: Tag) -> Tag | None:
    """Найти блок с ответом, комментарием и источником."""

    for node in card.find_all("div"):
        text = node.get_text(" ", strip=True)
        if any(label in text for label in ANSWER_LABELS) and len(text) < 8000:
            return node
    return None


def _question_text(card: Tag, block: Tag | None) -> str:
    """Извлечь текст вопроса из карточки, исключая ответ и метаданные."""

    for node in card.find_all("div", class_="whitespace-pre-wrap"):
        if block is not None and node is block:
            break
        text = node.get_text(" ", strip=True)
        if text and not any(label in text for label in ANSWER_LABELS):
            return text

    header = card.find("div", class_="whitespace-pre-wrap")
    if header is not None:
        return header.get_text(" ", strip=True)
    return ""


def _labeled_value(block: Tag | None, labels: tuple[str, ...]) -> str | None:
    """Прочитать значение после подписи вида «Ответ:»."""

    if block is None:
        return None

    for node in block.find_all("div"):
        text = node.get_text(" ", strip=True)
        for label in labels:
            if not text.startswith(label):
                continue
            if node.find("div") is not None:
                continue
            value = text[len(label) :].strip(" .;")
            return value or None
    return None


def _authors(card: Tag) -> list[str]:
    """Собрать авторов вопроса по ссылкам на `/person/<id>`."""

    authors: list[str] = []
    for link in card.select('a[href^="/person/"]'):
        name = link.get_text(" ", strip=True)
        if name and name not in authors:
            authors.append(name)
    return authors


def _pack(card: Tag) -> str | None:
    """Название пакета, к которому относится вопрос."""

    link = card.select_one('a[href^="/pack/"]')
    if link is None:
        return None
    return link.get_text(" ", strip=True) or None
