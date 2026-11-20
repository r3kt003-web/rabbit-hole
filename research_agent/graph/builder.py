import logging

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from research_agent.config.settings import get_settings
from research_agent.graph.nodes.evaluator import evaluator_node
from research_agent.graph.nodes.planner import planner_node
from research_agent.graph.nodes.researcher import researcher_node
from research_agent.graph.nodes.synthesizer import synthesizer_node
from research_agent.graph.state import ResearchState
from research_agent.verification.citations import verify_citations_node

logger = logging.getLogger(__name__)


def route_to_researchers(state: ResearchState) -> list[Send]:
    """Conditional edge from planner to research workers via Send API."""
    if not state.plan:
        logger.warning("No plan generated. Routing directly to END.")
        return []
    return [
        Send(
            "research_worker",
            {"sub_question": sq.model_dump(), "trace_id": state.trace_id},
        )
        for sq in state.plan
    ]


def route_after_evaluator(state: ResearchState) -> str | list[Send]:
    """Conditional edge from evaluator:
    - If passed or revision_count >= MAX_REVISIONS: route to END.
    - If failed with missing sub-questions: route back to research_worker for only missing sub-questions.
    - If failed with synthesis/grounding issues: route directly to synthesizer for fixes.
    """
    eval_res = state.eval_result
    settings = get_settings()

    # Finished condition
    if (
        eval_res is None
        or eval_res.passed
        or state.revision_count >= settings.max_revisions
    ):
        logger.info(
            f"Routing to END: passed={getattr(eval_res, 'passed', False)}, "
            f"revision_count={state.revision_count}/{settings.max_revisions}"
        )
        return END

    # Revision path A: Missing sub-questions need more research
    if eval_res.missing_sub_questions:
        missing_ids = set(eval_res.missing_sub_questions)
        missing_sqs = [
            sq
            for sq in state.plan
            if sq.id in missing_ids or sq.question in missing_ids
        ]
        if missing_sqs:
            logger.info(
                f"Routing {len(missing_sqs)} missing sub-question(s) back to research_worker for revision #{state.revision_count}."
            )
            return [
                Send(
                    "research_worker",
                    {"sub_question": sq.model_dump(), "trace_id": state.trace_id},
                )
                for sq in missing_sqs
            ]

    # Revision path B: Synthesis/citation fixes without missing sub-questions
    logger.info(
        f"Routing directly to synthesizer for revision #{state.revision_count}."
    )
    return "synthesizer"


def build_graph():
    """Builds the complete production LangGraph with Planner, Parallel Researchers,
    Synthesizer, Deterministic Citation Verifier, Evaluator, and Conditional Revision Loop.
    """
    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("planner", planner_node)
    graph.add_node("research_worker", researcher_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("citation_verifier", verify_citations_node)
    graph.add_node("evaluator", evaluator_node)

    # Add edges
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_to_researchers, ["research_worker"])
    graph.add_edge("research_worker", "synthesizer")
    graph.add_edge("synthesizer", "citation_verifier")
    graph.add_edge("citation_verifier", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        ["research_worker", "synthesizer", END],
    )

    return graph.compile()


# Backwards compatibility helpers
def build_phase4_graph():
    return build_graph()


def build_phase3_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("research_worker", researcher_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("citation_verifier", verify_citations_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_to_researchers, ["research_worker"])
    graph.add_edge("research_worker", "synthesizer")
    graph.add_edge("synthesizer", "citation_verifier")
    graph.add_edge("citation_verifier", END)
    return graph.compile()


def build_phase2_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("research_worker", researcher_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", route_to_researchers, ["research_worker"])
    graph.add_edge("research_worker", END)
    return graph.compile()
