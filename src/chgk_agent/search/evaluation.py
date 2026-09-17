"""Оценка качества локального поиска на наборе «описание → ожидаемый вопрос».

Метрики считаются по рангу ожидаемого вопроса в выдаче: `recall@k`
показывает, попал ли он в первые `k` результатов, а `MRR` — насколько
высоко он расположен в среднем. Отчёт сохраняется как артефакт, чтобы
сравнивать качество между изменениями ранжирования и моделями.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from chgk_agent.logging_setup import get_logger
from chgk_agent.models.normalization import normalize_text
from chgk_agent.search.local import LocalSearch

logger = get_logger(__name__)

DEFAULT_K = 5


@dataclass(frozen=True, slots=True)
class QualityCase:
    """Один случай набора: описание и ожидаемый вопрос."""

    description: str
    expected_question: str
    expected_answer: str | None = None


@dataclass(slots=True)
class CaseResult:
    """Результат одного случая набора."""

    description: str
    expected_question: str
    rank: int | None = None
    top_question: str | None = None

    @property
    def hit(self) -> bool:
        """Найден ли ожидаемый вопрос вообще."""

        return self.rank is not None


@dataclass(slots=True)
class QualityReport:
    """Отчёт о качестве поиска на наборе."""

    cases: int
    k: int
    recall_at_k: float
    mrr: float
    hits: int = 0
    results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Представить отчёт словарём для сохранения или вывода."""

        payload = asdict(self)
        payload["recall_at_k"] = round(self.recall_at_k, 6)
        payload["mrr"] = round(self.mrr, 6)
        return payload

    def save(self, path: Path) -> Path:
        """Сохранить отчёт артефактом в JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


def recall_at_k(ranks: list[int | None], k: int) -> float:
    """Доля случаев, где ожидаемый вопрос попал в первые `k` результатов."""

    if not ranks:
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Средний обратный ранг ожидаемого вопроса."""

    if not ranks:
        return 0.0
    total = sum(1.0 / rank for rank in ranks if rank is not None and rank > 0)
    return total / len(ranks)


def load_cases(path: Path) -> list[QualityCase]:
    """Прочитать набор случаев из JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("cases", payload) if isinstance(payload, dict) else payload
    cases: list[QualityCase] = []
    for item in items:
        cases.append(
            QualityCase(
                description=item["description"],
                expected_question=item["expected_question"],
                expected_answer=item.get("expected_answer"),
            )
        )
    return cases


async def evaluate_search_quality(
    search: LocalSearch,
    cases: list[QualityCase],
    *,
    k: int = DEFAULT_K,
) -> QualityReport:
    """Прогнать набор и посчитать `recall@k` и `MRR`."""

    results: list[CaseResult] = []
    for case in cases:
        matches = await search.search(case.description, limit=max(k, 1))
        expected = normalize_text(case.expected_question)
        rank = None
        for position, match in enumerate(matches, start=1):
            if normalize_text(match.question_text) == expected:
                rank = position
                break
        results.append(
            CaseResult(
                description=case.description,
                expected_question=case.expected_question,
                rank=rank,
                top_question=matches[0].question_text if matches else None,
            )
        )

    ranks = [result.rank for result in results]
    report = QualityReport(
        cases=len(results),
        k=k,
        recall_at_k=recall_at_k(ranks, k),
        mrr=mean_reciprocal_rank(ranks),
        hits=sum(1 for rank in ranks if rank is not None and rank <= k),
        results=results,
    )
    logger.info(
        "качество поиска измерено",
        cases=report.cases,
        k=k,
        recall_at_k=round(report.recall_at_k, 4),
        mrr=round(report.mrr, 4),
    )
    return report


__all__ = [
    "DEFAULT_K",
    "CaseResult",
    "QualityCase",
    "QualityReport",
    "evaluate_search_quality",
    "load_cases",
    "mean_reciprocal_rank",
    "recall_at_k",
]
