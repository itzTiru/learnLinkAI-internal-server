# ragents/saver_agent.py
from config import db
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict

class SaveRecommendationsInput(BaseModel):
    email: str = Field(..., description="User email")
    recommendations: List[Dict] = Field(..., description="List of recommendations")

@tool(args_schema=SaveRecommendationsInput)
async def save_recommendations_tool(email: str, recommendations: List[Dict]) -> None:
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