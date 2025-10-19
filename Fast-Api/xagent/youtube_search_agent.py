# ragents/youtube_search_agent.py
from config import YOUTUBE_API_KEY
from googleapiclient.discovery import build
import asyncio
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from xagent.state import RecommendationState, AgentMessage
from typing import List, Dict, Any

class SearchYouTubeInput(BaseModel):
    query: str = Field(..., description="Search query for YouTube")

@tool(args_schema=SearchYouTubeInput)
async def search_youtube(query: str) -> List[Dict[str, Any]]:
    """Search YouTube for educational videos based on the query."""
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoCategoryId="27",  # Education category
            maxResults=3
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, request.execute)
        items = response.get("items", []) or []
        results = [
            {
                "platform": "youtube",
                "title": it["snippet"]["title"],
                "description": it["snippet"]["description"],
                "url": f"https://www.youtube.com/watch?v={it['id']['videoId']}",
                "thumbnail": it["snippet"]["thumbnails"]["default"]["url"],
                "type": "video",
                "relevance_score": 0.95
            }
            for it in items
        ]
        return results
    except Exception as e:
        print("YouTube search error:", str(e))
        return []

async def youtube_search_node(state: RecommendationState) -> RecommendationState:
    """Node: Searches YouTube, sends results to ranking_agent."""
    if state["error"]:
        return state
    
    last_msg = state["messages"][-1].content if state["messages"] else {}
    query_str = last_msg.get("query", state["query"].get("query", ""))
    
    # FIXED: Use ainvoke with dict input to support async invocation properly
    results = await search_youtube.ainvoke({"query": query_str})
    state["youtube_results"] = results
    state["messages"].append(AgentMessage(
        sender="youtube_search",
        to="ranking_agent",
        content={"youtube_results": results, "query": state["query"]}
    ))
    return state