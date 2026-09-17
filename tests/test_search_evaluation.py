"""Тесты метрик качества поиска."""

import asyncio
import json
from pathlib import Path

import pytest

from chgk_agent.search.evaluation import (
    QualityCase,
    evaluate_search_quality,
    load_cases,
    mean_reciprocal_rank,
    recall_at_k,
)
from chgk_agent.search.local import LocalSearchResult


def _match(text: str) -> LocalSearchResult:
    """Совпадение локального поиска для тестов ранжирования."""

    return LocalSearchResult(
        question_id=abs(hash(text)) % 10_000,
        question_text=text,
        answer_text="ответ",
        comment=None,
        score=1.0,
        score_kind="semantic",
    )


class StubSearch:
    """Поиск с фиксированной выдачей по описанию."""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    async def search(self, query: str, *, limit: int = 20, min_score: float = 0.0):
        return [_match(text) for text in self._mapping.get(query, [])][:limit]


def test_recall_at_k_counts_hits_within_depth() -> None:
    assert recall_at_k([1, 3, None, 5], k=3) == 0.5
    assert recall_at_k([1, 2], k=5) == 1.0
    assert recall_at_k([], k=5) == 0.0


def test_mean_reciprocal_rank_uses_rank_reciprocals() -> None:
    assert mean_reciprocal_rank([1, 2, None]) == pytest.approx((1 + 0.5 + 0) / 3)
    assert mean_reciprocal_rank([]) == 0.0


async def test_evaluate_reports_recall_and_mrr() -> None:
    cases = [
        QualityCase("описание 1", "Вопрос 1"),
        QualityCase("описание 2", "Вопрос 2"),
        QualityCase("описание 3", "Вопрос 3"),
    ]
    search = StubSearch(
        {
            "описание 1": ["Вопрос 1", "другое"],
            "описание 2": ["другое", "другое-2", "Вопрос 2"],
            "описание 3": ["совсем другое"],
        }
    )

    report = await evaluate_search_quality(search, cases, k=3)

    assert report.cases == 3
    assert report.k == 3
    assert report.hits == 2
    assert report.recall_at_k == pytest.approx(2 / 3)
    assert report.mrr == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert report.results[2].rank is None


async def test_evaluate_normalizes_question_text() -> None:
    cases = [QualityCase("описание", "Вопрос  «с типографикой»")]
    search = StubSearch({"описание": ['вопрос "с типографикой"']})

    report = await evaluate_search_quality(search, cases, k=1)

    assert report.results[0].rank == 1
    assert report.recall_at_k == 1.0


def test_load_cases_and_save_report(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            {
                "cases": [
                    {"description": "о", "expected_question": "в", "expected_answer": "а"}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_cases(cases_path)
    assert cases == [QualityCase("о", "в", "а")]

    report_path = tmp_path / "report.json"

    async def _run() -> None:
        search = StubSearch({"о": ["в"]})
        report = await evaluate_search_quality(search, cases, k=1)
        report.save(report_path)

    asyncio.run(_run())

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["recall_at_k"] == 1.0
    assert payload["mrr"] == 1.0
    assert payload["cases"] == 1


def test_repository_quality_cases_are_valid() -> None:
    cases = load_cases(Path("resources/search_quality_cases.json"))

    assert len(cases) >= 3
    assert all(case.description.strip() for case in cases)
    assert all(case.expected_question.strip() for case in cases)


@pytest.mark.integration
async def test_quality_report_on_real_database(session_factory, tmp_path: Path) -> None:
    """Измерить recall@k и MRR на реальной базе и сохранить отчёт-артефакт."""

    from sqlalchemy import delete

    from chgk_agent.db.base import Question, Source, get_embedding_dim
    from chgk_agent.ingestion.importer import QuestionImporter
    from chgk_agent.models.domain import ParsedQuestion, ParseResult
    from chgk_agent.models.normalization import normalize_text
    from chgk_agent.search.local import LocalSearch

    dimension = get_embedding_dim()

    class _KeywordEmbedder:
        VOCABULARY = ("порошок", "пороховница", "бульба", "жираф", "абиссинск", "многострочный")

        model = "KeywordEmbeddings"

        @property
        def dimension(self) -> int:
            return dimension

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return [self._vector(text) for text in texts]

        def _vector(self, text: str) -> list[float]:
            lowered = text.casefold()
            vector = [0.01] * dimension
            for index, term in enumerate(self.VOCABULARY):
                if term in lowered:
                    vector[index] = 1.0
            return vector

    location = "quality-fixture.html"
    questions = [
        (
            "1",
            "В них хранилась доза порошка, укрытого от непогоды. Но как звали украинца, "
            "спрашивающего у подчинённых, остался ли у них ещё этот порошок?",
            "Тарас Бульба",
        ),
        ("2", "Согласно словарю, ИКС — «животное Абиссинское». Что это за ИКС?", "жираф"),
        (
            "3",
            "Многострочный вопрос:\nпервая строка\nвторая строка",
            "многострочный ответ",
        ),
    ]

    async def _purge() -> None:
        async with session_factory() as session:
            await session.execute(delete(Source).where(Source.location == location))
            await session.execute(delete(Question).where(~Question.occurrences.any()))
            await session.commit()

    await _purge()
    try:
        async with session_factory() as session:
            importer = QuestionImporter(session, _KeywordEmbedder())
            report = await importer.import_results(
                [
                    ParseResult(
                        location=location,
                        questions=[
                            ParsedQuestion(
                                source_key=key, question_text=text, answer_text=answer
                            )
                            for key, text, answer in questions
                        ],
                    )
                ]
            )
            await session.commit()
        assert report.unembedded == 0

        cases = [
            QualityCase(
                description=(
                    "У кого спрашивали, остался ли порох в пороховницах?"
                ),
                expected_question=questions[0][1],
            ),
            QualityCase(
                description="Какое животное словарь называет абиссинским?",
                expected_question=questions[1][1],
            ),
            QualityCase(
                description="многострочный вопрос первая строка вторая строка",
                expected_question=questions[2][1],
            ),
        ]

        async with session_factory() as session:
            search = LocalSearch(session, _KeywordEmbedder())
            quality = await evaluate_search_quality(search, cases, k=5)

        artifact = tmp_path / "search_quality_report.json"
        quality.save(artifact)

        assert quality.cases == len(cases)
        assert 0.0 <= quality.recall_at_k <= 1.0
        assert 0.0 <= quality.mrr <= 1.0
        assert quality.recall_at_k > 0.0
        # Отчёт фиксирует ранги по каждому случаю набора.
        assert all(
            result.rank is None or result.rank >= 1 for result in quality.results
        )
        saved = json.loads(artifact.read_text(encoding="utf-8"))
        assert saved["cases"] == len(cases)
        assert "recall_at_k" in saved and "mrr" in saved
        assert (
            normalize_text(cases[0].expected_question)
            == normalize_text(questions[0][1])
        )
    finally:
        await _purge()
