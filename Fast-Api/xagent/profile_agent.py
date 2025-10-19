# ragents/profile_agent.py
from config import db
from xagent.state import RecommendationState, AgentMessage
from typing import Dict, Any

async def fetch_user_profile(email: str) -> Dict[str, Any]:
    profile = await db.personal.find_one({"email": email})
    if not profile:
        return {"error": "Profile not found"}
    profile["_id"] = str(profile["_id"])
    return profile

async def profile_agent_node(state: RecommendationState) -> RecommendationState:
    """Node: Fetches profile, sends to query_generator if successful."""
    profile = await fetch_user_profile(state["email"])
    if profile.get("error"):
        state["error"] = profile["error"]
        state["messages"].append(AgentMessage(
            sender="profile_agent",
            to="error_handler",  
            content={"error": profile["error"]}
        ))
        return state
    
    state["profile"] = profile
    state["messages"].append(AgentMessage(
        sender="profile_agent",
        to="query_generator",
        content={"profile": profile, "mode": state["mode"]}
    )) 
    return state