"""Нормализация текста вопросов и вычисление хеша."""

import hashlib
import re
import unicodedata

_TYPOGRAPHY = {
    "\u00a0": " ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2039": "'",
    "\u203a": "'",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2026": "...",
}

_TRANSLATION = str.maketrans({source: target for source, target in _TYPOGRAPHY.items()})
_WHITESPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")


def normalize_text(text: str) -> str:
    """Привести текст к сопоставимому виду.

    Нормализация идемпотентна: повторное применение не меняет результат.
    """

    if text is None:
        return ""

    value = unicodedata.normalize("NFKC", text)
    value = value.translate(_TRANSLATION)
    value = value.casefold()
    value = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", value)
    value = _WHITESPACE.sub(" ", value)
    return value.strip()


def text_hash(text: str) -> str:
    """Вернуть sha256-хеш нормализованного текста."""

    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
