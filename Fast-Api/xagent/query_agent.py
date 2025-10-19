# ragents/query_agent.py
from config import GEMINI_API_KEY
import json
import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from xagent.state import RecommendationState, AgentMessage
from typing import Dict, Any

class GenerateQueryInput(BaseModel):
    """Input schema for generate_query tool."""
    profile: Dict[str, Any] = Field(..., description="User profile dict")
    mode: str = Field(..., description="Mode: 'ai' or 'traditional'")

@tool(args_schema=GenerateQueryInput)
async def generate_query(profile: Dict[str, Any], mode: str) -> Dict[str, str]:
    """Generate a search query based on the user's profile and mode."""
    try:
        education = profile.get("education", [])
        query_base = ""
        if education:
            degree = education[0].get("degree", "")
            field = education[0].get("field", "")
            query_base = f"{degree} {field} {mode} courses"

        if mode == "ai":
            # Use Gemini API for AI mode
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY
            }
            prompt = f"""
            Generate a concise search query for educational content based on this profile:
            - Education: {json.dumps(education)}
            - Mode: {mode}
            Return only the query text, e.g., 'Data Science computing ai courses'
            """
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, content=json.dumps(payload))
                resp.raise_for_status()
                data = resp.json()
                query_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", query_base).strip()
        else:
            query_text = query_base

        return {"query": query_text}
    except Exception as e:
        print("Error generating query:", str(e))
        return {"query": query_base if query_base else "educational content"}

async def query_agent_node(state: RecommendationState) -> RecommendationState:
    """Node: Generates query from profile, sends to search_coordinator."""
    if state["error"]:
        return state  # Propagate error
    
    # Extract from last message or state
    last_msg = state["messages"][-1].content if state["messages"] else {}
    profile = last_msg.get("profile", state["profile"])
    mode = last_msg.get("mode", state["mode"])
    
    # FIXED: Use ainvoke with dict input to avoid deprecation and callback errors
    query_dict = await generate_query.ainvoke({"profile": profile, "mode": mode})
    if not query_dict.get("query"):
        state["error"] = "Failed to generate query"
        return state
    
    state["query"] = query_dict
    state["messages"].append(AgentMessage(
        sender="query_generator",
        to="search_coordinator",
        content={"query": query_dict["query"]}
    ))
    return state