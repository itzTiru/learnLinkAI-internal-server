# ragents/search_coordinator.py
from xagent.state import RecommendationState, AgentMessage

async def search_coordinator_node(state: RecommendationState) -> RecommendationState:
    """Node: Broadcasts query to YouTube and Web search agents (parallel)."""
    if state["error"] or not state["query"]:
        state["error"] = state["error"] or "No query for search"
        return state
    
    query_str = state["query"].get("query", "")
    state["messages"].append(AgentMessage(
        sender="search_coordinator",
        to="youtube_search",
        content={"query": query_str}
    ))
    state["messages"].append(AgentMessage(
        sender="search_coordinator",
        to="web_search",
        content={"query": query_str}
    ))
    return state