"""Объединение ранжированных списков через reciprocal rank fusion."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

DEFAULT_K = 60


@dataclass(slots=True)
class RankedItem:
    """Элемент ранжированного списка одной ветки поиска."""

    key: str
    score: float = 0.0
    rank: int = 0


def reciprocal_rank_fusion(
    branches: Iterable[Sequence[RankedItem]],
    *,
    k: int = DEFAULT_K,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Слить несколько ранжированных списков в единый рейтинг.

    Каждая ветка вносит вклад `weight / (k + rank)`, где `rank` — позиция
    элемента в ветке, начиная с единицы. Результат нормируется в `[0, 1]`
    делением на максимальный накопленный вес, поэтому оценки сопоставимы
    между собой независимо от числа веток.
    """

    branch_list = [list(branch) for branch in branches]
    weight_list = list(weights) if weights is not None else [1.0] * len(branch_list)

    fused: dict[str, float] = {}
    for branch, weight in zip(branch_list, weight_list, strict=False):
        for position, item in enumerate(branch, start=1):
            rank = item.rank or position
            fused[item.key] = fused.get(item.key, 0.0) + weight / (k + rank)

    if not fused:
        return {}

    best = max(fused.values())
    if best <= 0:
        return dict.fromkeys(fused, 0.0)
    return {key: value / best for key, value in fused.items()}


def ranks_of(items: Sequence[RankedItem]) -> dict[str, int]:
    """Вернуть карту `ключ -> позиция` для ранжированного списка."""

    return {
        item.key: (item.rank or position)
        for position, item in enumerate(items, start=1)
    }
