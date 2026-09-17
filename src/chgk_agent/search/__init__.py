"""Гибридный локальный поиск по вопросам ЧГК."""

from chgk_agent.search.local import LocalSearch, LocalSearchResult
from chgk_agent.search.rrf import reciprocal_rank_fusion

__all__ = ["LocalSearch", "LocalSearchResult", "reciprocal_rank_fusion"]
