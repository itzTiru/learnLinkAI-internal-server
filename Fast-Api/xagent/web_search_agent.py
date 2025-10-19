# ragents/web_search_agent.py
from config import GOOGLE_API_KEY, GOOGLE_CSE_ID
import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from xagent.state import RecommendationState, AgentMessage
from typing import List, Dict, Any

class SearchGoogleInput(BaseModel):
    query: str = Field(..., description="Search query for Google CSE")

@tool(args_schema=SearchGoogleInput)
async def search_google_cse(query: str) -> List[Dict[str, Any]]:
    """Search Google Custom Search Engine for educational content."""
    try:
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": 3
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("https://www.googleapis.com/customsearch/v1", params=params)
            resp.raise_for_status()
            data = resp.json()
        items = data.get("items", []) or []
        results = [
            {
                "platform": "web",
                "title": it.get("title"),
                "description": it.get("snippet"),
                "url": it.get("link"),
                "thumbnail": None,
                "type": "article",
                "relevance_score": 0.95
            }
            for it in items
        ]
        return results
    except Exception as e:
        print("Google CSE search error:", str(e))
        return []

async def web_search_node(state: RecommendationState) -> RecommendationState:
    """Node: Searches web, sends results to ranking_agent."""
    if state["error"]:
        return state
    
    last_msg = state["messages"][-1].content if state["messages"] else {}
    query_str = last_msg.get("query", state["query"].get("query", ""))
    
    # FIXED: Use ainvoke with dict input to support async invocation properly
    results = await search_google_cse.ainvoke({"query": query_str})
    state["web_results"] = results
    state["messages"].append(AgentMessage(
        sender="web_search",
        to="ranking_agent",
        content={"web_results": results, "query": state["query"]}
    ))
    return state