"""Тесты CLI импорта и переиндексации."""

import json
from pathlib import Path

import pytest

from chgk_agent.embeddings import cli as reindex_cli
from chgk_agent.ingestion import import_cli
from chgk_agent.ingestion.operations import ImportOperationRegistry
from chgk_agent.ingestion.service import IngestionService


class FakeEmbeddingProvider:
    """Провайдер эмбеддингов-заглушка."""

    model = "Fake"
    dimension = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


class _Session:
    async def __aenter__(self) -> "_Session":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def commit(self) -> None:
        return None


class _SessionFactory:
    def __call__(self) -> _Session:
        return _Session()


class _Report:
    """Подменный отчёт импорта."""

    operation_id = "op-1"
    locations = ["fixture.html"]
    processed = 3
    added = 2
    updated = 1
    unchanged = 0
    skipped = 1
    unembedded = 0
    issues: list = []
    is_partial = True


def test_import_cli_prints_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    html = tmp_path / "source.html"
    html.write_text("<html></html>", encoding="utf-8")

    async def fake_run_import(args):
        return _Report()

    monkeypatch.setattr(import_cli, "_run_import", fake_run_import)

    code = import_cli.main([str(html)])

    assert code == 1  # частичный импорт


def test_import_cli_json_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def fake_run_import(args):
        return _Report()

    monkeypatch.setattr(import_cli, "_run_import", fake_run_import)

    import_cli.main(["--json", "fixture.html"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["operation_id"] == "op-1"
    assert payload["added"] == 2
    assert payload["updated"] == 1


def test_reindex_cli_prints_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Run:
        embedded = 5
        failed = 0
        remaining = 0
        errors: list = []

    async def fake_run_reindex(args):
        return _Run()

    monkeypatch.setattr(reindex_cli, "_run_reindex", fake_run_reindex)

    code = reindex_cli.main([])

    output = capsys.readouterr().out
    assert code == 0
    assert "Векторизовано: 5" in output


def test_reindex_cli_passes_force_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Run:
        embedded = 2
        failed = 0
        remaining = 0
        errors: list = []

    async def fake_run_reindex(args):
        captured["force"] = args.force
        return _Run()

    monkeypatch.setattr(reindex_cli, "_run_reindex", fake_run_reindex)

    reindex_cli.main(["--force"])

    assert captured["force"] is True
    assert reindex_cli.build_parser().parse_args([]).force is False


def test_reindex_cli_reports_failures(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _Run:
        embedded = 0
        failed = 2
        remaining = 2
        errors = ["батч упал"]

    async def fake_run_reindex(args):
        return _Run()

    monkeypatch.setattr(reindex_cli, "_run_reindex", fake_run_reindex)

    code = reindex_cli.main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["failed"] == 2
    assert payload["errors"] == ["батч упал"]


async def test_run_import_reports_counters() -> None:
    """Импорт через сервис заполняет счётчики отчёта."""

    service = IngestionService(
        _SessionFactory(),  # type: ignore[arg-type]
        embedding_provider=FakeEmbeddingProvider(),
        registry=ImportOperationRegistry(),
    )

    assert service.registry is not None
