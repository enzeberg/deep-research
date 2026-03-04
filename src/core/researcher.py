"""Research agent — a real tool-calling agent that autonomously gathers information."""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from src.tools.web_search import web_search, get_search_urls
from src.tools.content_fetch import fetch_webpage


RESEARCHER_SYSTEM_PROMPT = """\
You are an expert research agent. Your job is to thoroughly investigate a topic
based on a research plan.

## Tools

- **web_search**: Search the web for information on any topic.
- **get_search_urls**: Get URLs from search results for deeper fetching.
- **fetch_webpage**: Fetch full text content from a web page.

## Research Strategy

1. Execute each search query from the plan using **web_search**.
2. For the most promising results, use **fetch_webpage** to get full page content.
3. After completing all planned searches, check if there are information gaps.
   If so, formulate new search queries and continue.
4. When you have sufficient information, produce a structured summary.

## Output Format

When done, output your findings as:

**Key Findings:**
- Finding 1 (with source URL)
- Finding 2 (with source URL)
- ...

**Sources:**
- [Title](URL) - brief description
- ...

**Information Gaps:**
- Any areas where you could not find sufficient information.

Be thorough but efficient. Prioritize high-quality, authoritative sources.
Do NOT fabricate information — only report what you actually found.
"""


def get_default_tools() -> list[BaseTool]:
    """Get the default set of research tools."""
    return [web_search, get_search_urls, fetch_webpage]


def create_research_agent(
    llm: BaseChatModel,
    tools: list[BaseTool] | None = None,
):
    """Create a research agent with tool-calling capabilities.

    Uses LangChain v1 create_agent (replaces the deprecated
    langgraph.prebuilt.create_react_agent). The agent autonomously decides
    which tools to call, when to call them, and when it has gathered enough
    information.

    Args:
        llm: The language model to use.
        tools: Custom tools list. If None, uses default web tools.

    Returns:
        A compiled agent built on LangGraph.
    """
    tools = tools or get_default_tools()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
    )

    return agent
