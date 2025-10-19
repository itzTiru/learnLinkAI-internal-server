# ragents/saver_agent.py
from config import db
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from xagent.state import RecommendationState, AgentMessage

class SaveRecommendationsInput(BaseModel):
    email: str = Field(..., description="User email")
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommendations")

@tool(args_schema=SaveRecommendationsInput)
async def save_recommendations_tool(email: str, recommendations: List[Dict[str, Any]]) -> None:
    """Save recommendations to MongoDB for the given user."""
    try:
        await db.recommendations.update_one(
            {"email": email},
            {"$set": {"email": email, "recommendations": recommendations}},
            upsert=True
        )
    except Exception as e:
        print("Error saving recommendations:", str(e))
        raise

async def saver_agent_node(state: RecommendationState) -> RecommendationState:
    """Node: Transforms and saves ranked results."""
    if state["error"]:
        return state
    
    last_msg = state["messages"][-1].content if state["messages"] else {}
    ranked = last_msg.get("ranked_results", state["ranked_results"])
    email = last_msg.get("email", state["email"])
    
    # Transformation (moved from orchestrator)
    transformed = [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "type": item.get("type", "video" if item.get("platform", "").lower() == "youtube" else "article"),
            "description": item.get("description", ""),
            "thumbnail": item.get("thumbnail", None),
            "duration": item.get("duration", None),
            "category": item.get("category", "Video" if item.get("type", "video") == "video" else "Article"),
            "platform": item.get("platform", "YouTube" if item.get("type", "video") == "video" else "Web"),
            "relevance_score": int(item.get("relevance_score", 0.95) * 100)
        }
        for item in ranked
    ]
    
    # FIXED: Use ainvoke with structured dict to avoid deprecation and callback errors
    await save_recommendations_tool.ainvoke({"email": email, "recommendations": transformed})
    
    state["messages"].append(AgentMessage(
        sender="saver_agent",
        to="END",  # Workflow complete
        content={"recommendations": transformed}
    ))
    return state