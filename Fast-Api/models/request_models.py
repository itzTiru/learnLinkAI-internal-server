# models/request_models.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Union

class UserRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)

class AnswerSubmission(BaseModel):
    session_id: str = Field(..., min_length=1)
    answers: Dict[str, Union[str, int]] = Field(default_factory=dict)  # Allow str or int for answers