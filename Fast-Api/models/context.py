# models/context.py
from pydantic import BaseModel
from typing import List, Dict, Optional

class AgentContext(BaseModel):
    session_id: str
    user_id: str
    query: str
    stage: str
    mcqs: List[Dict] = []
    answers: Dict[str, str] = {}
    style: Optional[str] = None
    difficulty: Optional[Dict] = None
    roadmap: List[Dict] = []
    recommendations: Dict[str, List[Dict]] = {}
    explanations: List[Dict] = []
