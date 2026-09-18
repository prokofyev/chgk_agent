"""Граф оркестрации поиска на LangGraph.

Схема: `parse_request → fan-out (local_search | external_search) →
merge_and_dedupe → rerank → generate → format_response`. Ветки поиска
выполняются параллельно и независимо: падение или таймаут одной не
отменяет вторую, а результат помечается частичным.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from chgk_agent.logging_setup import get_logger, request_context
from chgk_agent.search.models import SearchOutcome
from chgk_agent.search.nodes import (
    SearchDeps,
    embed_query,
    external_search,
    format_response,
    generate,
    has_results,
    local_search,
    merge_and_dedupe,
    needs_generation,
    parse_request,
    rerank,
    score_external,
    score_matches,
)
from chgk_agent.search.state import SearchState

logger = get_logger(__name__)

LOCAL_NODE = "local_search"
EXTERNAL_NODE = "external_search"
EXTERNAL_SCORE_NODE = "score_external"
SCORE_NODE = "score_matches"


@dataclass(slots=True)
class SearchGraph:
    """Скомпилированный граф поиска с зависимостями."""

    deps: SearchDeps
    graph: Any

    async def run(
        self,
        query: str,
        *,
        limit: int = 20,
        min_score: float = 0.0,
        generate_answer: bool = True,
        disable_lexical: bool = False,
        request_id: str | None = None,
    ) -> SearchOutcome:
        """Выполнить поиск и вернуть итоговый результат."""

        with request_context(request_id) as active_request_id:
            state = await self.graph.ainvoke(
                {
                    "query": query,
                    "limit": limit,
                    "min_score": min_score,
                    "generate_answer": generate_answer,
                    "disable_lexical": disable_lexical,
                    "request_id": active_request_id,
                }
            )
            outcome: SearchOutcome = state["outcome"]
            outcome.request_id = active_request_id
            return outcome


def build_search_graph(deps: SearchDeps) -> SearchGraph:
    """Собрать и скомпилировать граф поиска."""

    builder = StateGraph(SearchState)
    builder.add_node("parse_request", parse_request)
    builder.add_node("embed_query", partial(embed_query, deps=deps))
    builder.add_node(LOCAL_NODE, partial(local_search, deps=deps))
    builder.add_node(EXTERNAL_NODE, partial(external_search, deps=deps))
    builder.add_node(EXTERNAL_SCORE_NODE, partial(score_external, deps=deps))
    builder.add_node("merge_and_dedupe", merge_and_dedupe)
    builder.add_node(SCORE_NODE, partial(score_matches, deps=deps))
    builder.add_node("rerank", rerank)
    builder.add_node("generate", partial(generate, deps=deps))
    builder.add_node("format_response", partial(format_response, deps=deps))

    builder.add_edge(START, "parse_request")
    builder.add_edge("parse_request", "embed_query")
    builder.add_edge("embed_query", LOCAL_NODE)
    builder.add_edge("embed_query", EXTERNAL_NODE)
    builder.add_edge(EXTERNAL_NODE, EXTERNAL_SCORE_NODE)
    builder.add_edge([LOCAL_NODE, EXTERNAL_SCORE_NODE], "merge_and_dedupe")
    builder.add_edge("merge_and_dedupe", SCORE_NODE)
    builder.add_conditional_edges(
        SCORE_NODE,
        has_results,
        {"rerank": "rerank", "skip": "format_response"},
    )
    builder.add_conditional_edges(
        "rerank",
        needs_generation,
        {"generate": "generate", "skip": "format_response"},
    )
    builder.add_edge("generate", "format_response")
    builder.add_edge("format_response", END)

    return SearchGraph(deps=deps, graph=builder.compile())


async def run_search(
    deps: SearchDeps,
    query: str,
    **kwargs: Any,
) -> SearchOutcome:
    """Разовый поиск без переиспользования собранного графа."""

    return await build_search_graph(deps).run(query, **kwargs)


__all__ = ["SearchGraph", "build_search_graph", "run_search"]
