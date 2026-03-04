"""Main research workflow using LangGraph.

Workflow: plan → research (agent) → report
"""

from typing import Any
import logging

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from src.workflows.states import ResearchState
from src.core.planner import Planner
from src.core.researcher import create_research_agent
from src.core.report_generator import ReportGenerator
from src.memory import MemoryManager
from src.llm import ModelRouter
from src.config import ResearchConfig

logger = logging.getLogger(__name__)


class ResearchWorkflow:
    """LangGraph workflow for deep research.

    Three-step pipeline:
      1. Plan    — LLM chain creates a research plan from the query.
      2. Research — A ReAct agent autonomously searches the web and gathers info.
      3. Report  — LLM chain synthesizes findings into a Markdown report.
    """

    def __init__(
        self,
        config: ResearchConfig,
        memory_manager: MemoryManager,
    ):
        self.config = config
        self.memory_manager = memory_manager

        # LLM setup
        self.model_router = ModelRouter(provider=config.llm_provider)
        self.llm = self.model_router.get_model(temperature=0.3)

        # Components
        self.planner = Planner(self.llm)
        self.research_agent = create_research_agent(
            llm=self.model_router.get_model(temperature=0.1),
        )
        self.report_generator = ReportGenerator(self.llm)

        # Build graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the workflow graph."""
        workflow = StateGraph(ResearchState)

        workflow.add_node("plan", self._plan_node)
        workflow.add_node("research", self._research_node)
        workflow.add_node("report", self._report_node)

        workflow.add_edge(START, "plan")
        workflow.add_edge("plan", "research")
        workflow.add_edge("research", "report")
        workflow.add_edge("report", END)

        return workflow.compile()

    # ── Nodes ────────────────────────────────────────────────────────────

    async def _plan_node(self, state: ResearchState) -> dict[str, Any]:
        """Create a research plan."""
        logger.info("Planning research for: %s", state["query"])

        memory_context = ""
        if self.config.memory_enabled:
            ctx = self.memory_manager.get_context()
            parts = []
            if ctx.get("short_term_summary"):
                parts.append(f"Recent Research:\n{ctx['short_term_summary']}")
            if ctx.get("working_summary"):
                parts.append(f"Current Session:\n{ctx['working_summary']}")
            memory_context = "\n\n".join(parts)

        plan = await self.planner.create_plan(state["query"], memory_context)

        if self.config.memory_enabled:
            self.memory_manager.add_to_working("plan", plan)

        logger.info("Plan created with %d search queries", len(plan.get("search_queries", [])))

        return {
            "plan": plan,
            "memory_context": memory_context,
        }

    async def _research_node(self, state: ResearchState) -> dict[str, Any]:
        """Run the research agent to gather information."""
        logger.info("Starting research agent...")

        plan = state["plan"]
        research_prompt = self._build_research_prompt(state["query"], plan)

        result = await self.research_agent.ainvoke({
            "messages": [HumanMessage(content=research_prompt)],
        })

        # Extract the agent's final text response
        messages = result.get("messages", [])
        findings = ""
        if messages:
            findings = messages[-1].content

        if self.config.memory_enabled:
            self.memory_manager.add_to_working("findings", findings[:500])

        logger.info("Research completed, findings length: %d chars", len(findings))

        return {"findings": findings}

    async def _report_node(self, state: ResearchState) -> dict[str, Any]:
        """Generate the final research report."""
        logger.info("Generating report...")

        report = await self.report_generator.generate(
            query=state["query"],
            plan=state["plan"],
            findings=state["findings"],
        )

        if self.config.memory_enabled:
            self.memory_manager.save_session(
                query=state["query"],
                plan=state["plan"],
                results=[{"findings": state["findings"][:1000]}],
                report=report,
            )

        logger.info("Report generated, length: %d chars", len(report))

        return {"report": report, "completed": True}

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_research_prompt(query: str, plan: dict[str, Any]) -> str:
        """Build the prompt that kicks off the research agent."""
        parts = [
            f"Research Query: {query}\n",
            "Research Plan:",
        ]

        if plan.get("objective"):
            parts.append(f"  Objective: {plan['objective']}")
        if plan.get("sub_topics"):
            parts.append(f"  Sub-topics: {', '.join(plan['sub_topics'])}")
        if plan.get("search_queries"):
            parts.append("  Search queries to execute:")
            for i, q in enumerate(plan["search_queries"], 1):
                parts.append(f"    {i}. {q}")
        if plan.get("priority_areas"):
            parts.append(f"  Priority areas: {', '.join(plan['priority_areas'])}")
        if plan.get("depth"):
            parts.append(f"  Depth: {plan['depth']}")

        parts.append(
            "\nPlease execute the research plan above. "
            "Use web_search for each search query, and fetch_webpage for the "
            "most relevant URLs to get detailed content. "
            "After gathering sufficient information, provide a structured "
            "summary of your findings."
        )

        return "\n".join(parts)

    async def run(self, query: str) -> dict[str, Any]:
        """Run the research workflow.

        Args:
            query: The research query.

        Returns:
            Final workflow state including the report.
        """
        initial_state: ResearchState = {
            "query": query,
            "plan": {},
            "memory_context": "",
            "findings": "",
            "report": "",
            "completed": False,
            "error": None,
        }

        try:
            final_state = await self.graph.ainvoke(initial_state)
            return final_state
        except Exception as e:
            logger.error("Workflow failed: %s", e, exc_info=True)
            return {
                **initial_state,
                "error": str(e),
                "completed": True,
            }
