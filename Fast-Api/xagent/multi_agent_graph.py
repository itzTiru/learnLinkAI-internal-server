import json
from typing import Literal
from langgraph.graph import StateGraph, END
from xagent.state import RecommendationState, AgentMessage
from xagent.profile_agent import profile_agent_node
from xagent.query_agent import query_agent_node
from xagent.search_coordinator import search_coordinator_node
from xagent.youtube_search_agent import youtube_search_node
from xagent.web_search_agent import web_search_node
from xagent.ranking_agent import ranking_agent_node
from xagent.saver_agent import saver_agent_node

def route_after_profile(state: RecommendationState) -> Literal["query_generator", "__end__"]:
    """Router after profile: Direct to query_generator or END on error."""
    if state.get("error"):
        return END  
    return "query_generator"

def route_after_query(state: RecommendationState) -> Literal["search_coordinator", "__end__"]:
    """Router after query: To coordinator or END on error."""
    if state.get("error"):
        return END
    return "search_coordinator"


def route_after_ranking(state: RecommendationState) -> Literal["saver_agent", "__end__"]:
    if state.get("error"):
        return END
    return "saver_agent"


workflow = StateGraph(RecommendationState)


workflow.add_node("profile_agent", profile_agent_node)
workflow.add_node("query_generator", query_agent_node)
workflow.add_node("search_coordinator", search_coordinator_node)
workflow.add_node("youtube_search", youtube_search_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("ranking_agent", ranking_agent_node)
workflow.add_node("saver_agent", saver_agent_node)

workflow.set_entry_point("profile_agent")

workflow.add_conditional_edges(
    "profile_agent",
    route_after_profile,
    {"query_generator": "query_generator", END: END}
)

workflow.add_conditional_edges(
    "query_generator",
    route_after_query,
    {"search_coordinator": "search_coordinator", END: END}
)

workflow.add_edge("search_coordinator", "youtube_search")
workflow.add_edge("search_coordinator", "web_search")

# Fan-in to ranking
workflow.add_edge("youtube_search", "ranking_agent")
workflow.add_edge("web_search", "ranking_agent")

# Conditional after ranking
workflow.add_conditional_edges(
    "ranking_agent",
    route_after_ranking,
    {"saver_agent": "saver_agent", END: END}
)

workflow.add_edge("saver_agent", END)

# Compile
app = workflow.compile()

# Updated runner: Use stream for step-by-step visibility
async def run_a2a_recommendations(email: str, mode: str) -> dict:
    """Run the A2A graph with streaming for debugging."""
    initial_state: RecommendationState = {
        "email": email,
        "mode": mode,
        "messages": [],  # Ensure initialized
        "profile": None,
        "query": None,
        "youtube_results": None,
        "web_results": None,
        "ranked_results": None,
        "error": None
    }
    
    try:
        # Stream to see each step
        async for event in app.astream(initial_state):
            pass
        
        # Get final state (last event or invoke fallback)
        final_state = await app.ainvoke(initial_state)  # Safe fallback for extraction
        
        if final_state.get("error"):
            raise ValueError(final_state["error"])
        
        # Extract recommendations
        last_msg = final_state["messages"][-1] if final_state["messages"] else None
        recommendations = last_msg.content.get("recommendations", []) if last_msg else []
        
        return {"recommendations": recommendations}
    
    except Exception as e:
        raise