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
