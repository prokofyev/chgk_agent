"""Тесты структурированного логирования."""

import json

import pytest

from chgk_agent.logging_setup import (
    configure_logging,
    get_logger,
    get_request_id,
    new_request_id,
    request_context,
)


def _records_from_output(output: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def test_request_context_sets_and_resets_request_id() -> None:
    assert get_request_id() is None

    with request_context("abc123") as request_id:
        assert request_id == "abc123"
        assert get_request_id() == "abc123"

    assert get_request_id() is None


def test_generated_request_ids_are_unique() -> None:
    first = new_request_id()
    second = new_request_id()

    assert first != second
    assert first


def test_log_record_contains_request_id(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=True, level="INFO")
    logger = get_logger("test")

    with request_context("req-42"):
        logger.info("поиск выполнен", source="local")

    records = _records_from_output(capsys.readouterr().out)
    assert records
    assert records[0]["request_id"] == "req-42"
    assert records[0]["event"] == "поиск выполнен"
    assert records[0]["source"] == "local"


def test_secrets_are_masked_in_log_output(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(json_logs=True, level="INFO")
    logger = get_logger("test")
    secret_value = "super-secret-token-value"

    logger.info(
        "вызов внешнего сервиса",
        authorization=secret_value,
        access_token=secret_value,
        credentials=secret_value,
        base_url="https://gotquestions.online",
    )

    captured = capsys.readouterr()
    assert secret_value not in captured.out

    records = _records_from_output(captured.out)
    assert records[0]["authorization"] == "***"
    assert records[0]["access_token"] == "***"
    assert records[0]["credentials"] == "***"
    assert records[0]["base_url"] == "https://gotquestions.online"
