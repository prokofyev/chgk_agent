"""Схема состояния графа поиска.

Ветки локального и внешнего поиска пишут в состояние параллельно, поэтому
ключи, обновляемые из fan-out, помечены редьюсерами: LangGraph не разрешает
двум узлам писать в один и тот же канал за один шаг без объединения.
"""

from typing import Annotated, Any, TypedDict


def _keep_last(left: Any, right: Any) -> Any:
    """Оставить значение, записанное последним."""

    return right


class SearchState(TypedDict, total=False):
    """Состояние одного поиска."""

    query: str
    query_text: str
    query_embedding: Any
    query_embedding_error: str | None
    limit: int
    min_score: float
    generate_answer: bool
    disable_lexical: bool
    request_id: str
    started_at: float

    local: Annotated[Any, _keep_last]
    local_seconds: Annotated[float, _keep_last]
    external: Annotated[Any, _keep_last]
    external_seconds: Annotated[float, _keep_last]

    matches: Any
    sources: Any
    answer: Any
    outcome: Any


__all__ = ["SearchState"]
