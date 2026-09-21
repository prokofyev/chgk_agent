"""Тесты схемы состояния графа поиска."""

from chgk_agent.search.state import SearchState


def test_state_carries_term_informativeness_and_reason() -> None:
    """Информативность терминов и причина её отсутствия живут в состоянии."""

    state: SearchState = {
        "term_weights": {"галстук": 3.2, "жираф": 4.1},
        "term_weights_error": None,
    }

    assert state["term_weights"] == {"галстук": 3.2, "жираф": 4.1}
    assert state["term_weights_error"] is None


def test_state_reports_unavailable_informativeness() -> None:
    """Причина недоступности сериализуется как обычная строка."""

    state: SearchState = {
        "term_weights": None,
        "term_weights_error": "индекс недоступен",
    }

    assert state["term_weights"] is None
    assert state["term_weights_error"] == "индекс недоступен"


def test_state_carries_both_answers() -> None:
    """Состояние несёт два ответа: без подгрузки и с подгрузкой."""

    state: SearchState = {
        "answer_without_context": "без подсказок",
        "answer_with_context": "с подсказками",
    }

    assert state["answer_without_context"] == "без подсказок"
    assert state["answer_with_context"] == "с подсказками"


def test_state_allows_absent_context_answer() -> None:
    """Второго прогона может не быть: отсутствие ответа выражается `None`."""

    state: SearchState = {
        "answer_without_context": "без подсказок",
        "answer_with_context": None,
    }

    assert state["answer_without_context"] == "без подсказок"
    assert state["answer_with_context"] is None


def test_state_has_no_generation_switch() -> None:
    """Признака отключения генерации в состоянии больше нет."""

    assert "generate_answer" not in SearchState.__annotations__
