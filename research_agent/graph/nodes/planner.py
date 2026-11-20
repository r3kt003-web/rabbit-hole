import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from research_agent.config.settings import get_settings
from research_agent.graph.state import ResearchState, SubQuestion
from research_agent.llm.client import get_chat_model
from research_agent.llm.rate_limiter import call_with_rate_limit

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are an expert research planner.
Your job is to break down a primary research question into 3 to 6 focused, independent sub-questions.
Each sub-question must target a distinct dimension of the topic to allow comprehensive, parallel research.

For each sub-question provide:
1. id: unique identifier, e.g. "sq_1", "sq_2", etc.
2. question: specific, factual research question.
3. search_intent: clear description of what information to search for.
4. answer_criteria: objective criteria for what makes a complete and factual answer.

Requirements:
- Minimum 3, maximum 6 sub-questions.
- Do not make sub-questions redundant or overlapping.
- Ensure all sub-questions directly contribute to answering the primary question."""


class PlannerOutput(BaseModel):
    sub_questions: list[SubQuestion] = Field(
        min_length=3,
        max_length=6,
        description="3 to 6 independent sub-questions covering the research topic",
    )


async def plan_research(
    question: str,
    model_name: str | None = None,
    mock_subquestions: list[SubQuestion] | None = None,
) -> PlannerOutput:
    """Decomposes a primary research question into 3-6 structured sub-questions.

    Uses structured output with Pydantic schema. Retries once on malformed output, then degrades gracefully.
    """
    if mock_subquestions is not None:
        return PlannerOutput(sub_questions=mock_subquestions)

    settings = get_settings()
    model = model_name or settings.planner_model

    chat_model = get_chat_model(model, temperature=0.0)
    structured_model = chat_model.with_structured_output(PlannerOutput)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Primary Research Question: {question}"),
    ]

    attempt = 0
    max_retries = 1

    while attempt <= max_retries:
        try:
            result = await call_with_rate_limit(structured_model.ainvoke, messages)
            if isinstance(result, PlannerOutput) and len(result.sub_questions) >= 3:
                return result
            elif isinstance(result, dict) and "sub_questions" in result:
                return PlannerOutput(**result)
            else:
                logger.warning(f"Planner structured output unexpected format: {result}")
        except (RuntimeError, ValueError, TypeError) as exc:
            attempt += 1
            logger.warning(
                f"Planner call failed (attempt {attempt}/{max_retries + 1}): {exc}"
            )
            if attempt > max_retries:
                break

    # Graceful degradation fallback: construct 3 basic sub-questions
    logger.warning("Degrading planner to fallback structured sub-questions.")
    return PlannerOutput(
        sub_questions=[
            SubQuestion(
                id="sq_1",
                question=f"What is the background and key concepts of '{question}'?",
                search_intent=f"Fundamental definitions, context and background of {question}",
                answer_criteria="Provides clear definitions, foundational context, and main components.",
            ),
            SubQuestion(
                id="sq_2",
                question=f"What are the recent developments and empirical evidence regarding '{question}'?",
                search_intent=f"Recent breakthroughs, statistics, studies and findings about {question}",
                answer_criteria="Cites recent data, studies, or concrete real-world examples.",
            ),
            SubQuestion(
                id="sq_3",
                question=f"What are the major challenges, trade-offs, and future outlook for '{question}'?",
                search_intent=f"Future outlook, limitations, risks, and expert perspectives on {question}",
                answer_criteria="Identifies key limitations, challenges, controversies, and future directions.",
            ),
        ]
    )


async def planner_node(state: ResearchState) -> dict[str, Any]:
    """LangGraph node for planning sub-questions."""
    output = await plan_research(state.question)
    return {"plan": output.sub_questions}
