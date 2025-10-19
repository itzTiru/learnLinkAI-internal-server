from typing import Annotated, List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel
import operator

class AgentMessage(BaseModel):
    """Message format for A2A communication between agents."""
    to: str  # e.g., "query_generator"
    content: Dict[str, Any]  # e.g., {"profile": {...}, "mode": "ai"}
    sender: str  # e.g., "profile_agent"
    timestamp: Optional[str] = None  # Optional for auditing

def list_reducer(x: Optional[List], y: Optional[List]) -> List:
    """Custom reducer: Concat lists, treating None as empty."""
    return (x or []) + (y or [])

class RecommendationState(TypedDict):
    """Shared state passed between agents, with reducers for concurrency."""
    email: Annotated[str, lambda x, y: y]  # Latest wins (immutable)
    mode: Annotated[str, lambda x, y: y]  # Latest wins
    profile: Annotated[Optional[Dict[str, Any]], lambda x, y: y]  # Latest
    query: Annotated[Optional[Dict[str, Any]], lambda x, y: y]  # Latest
    youtube_results: Annotated[Optional[List[Dict[str, Any]]], list_reducer]  # Concat, handle None
    web_results: Annotated[Optional[List[Dict[str, Any]]], list_reducer]  # Concat, handle None
    ranked_results: Annotated[Optional[List[Dict[str, Any]]], lambda x, y: y]  # Latest
    messages: Annotated[List[AgentMessage], operator.add]  # Append messages
    error: Annotated[Optional[str], lambda x, y: y]  # Latest (propagate errors)