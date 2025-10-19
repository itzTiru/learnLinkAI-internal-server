# recommendations.py
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import jwt

from config import db, JWT_SECRET, JWT_ALGORITHM
# from ragents.custom_orchestrator import orchestrate_recommendations
from xagent.multi_agent_graph import run_a2a_recommendations

load_dotenv()

router = APIRouter(tags=["recommendations"])  # No prefix
security = HTTPBearer()

# ------------------------------
# JWT verification
# ------------------------------
async def verify_token(credentials: HTTPBearer = Depends(security)):
    
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        email = payload.get("email")
       
        return email
    except jwt.PyJWTError as e:
  
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

# ------------------------------
# Response Schemas
# ------------------------------
class RecommendationOut(BaseModel):
    title: str
    url: Optional[str] = None
    type: str
    description: str
    thumbnail: Optional[str] = None
    duration: Optional[str] = None
    category: str = Field(..., description="e.g., 'Video' or 'Article'")
    platform: str = Field(..., description="e.g., 'YouTube' or 'Web'")
    relevance_score: int = Field(..., description="Relevance score (%)")

class RecommendationsResponse(BaseModel):
    recommendations: List[RecommendationOut]

# ------------------------------
# GET endpoint — fetch recommendations
# ------------------------------
@router.get("/recommendations", response_model=dict)
async def get_recommendations(
    mode: str = Query("traditional"),
    current_user: str = Depends(verify_token)
):

    email = current_user

    doc = await db.recommendations.find_one({"email": email})

    if doc:
        recs = doc.get("recommendations", [])
        
        transformed = [
            {
                "title": rec.get("title", ""),
                "url": rec.get("url", ""),
                "type": rec.get("type", "video" if rec.get("platform", "").lower() == "youtube" else "article"),
                "description": rec.get("description", ""),
                "thumbnail": rec.get("thumbnail", None),
                "duration": rec.get("duration", None),
                "category": rec.get("category", "Video" if rec.get("type", "video") == "video" else "Article"),
                "platform": rec.get("platform", "YouTube" if rec.get("type", "video") == "video" else "Web"),
                "relevance_score": rec.get("relevance_score", 95)
            }
            for rec in recs
        ]
 
        return {"recommendations": transformed}

    return await generate_recommendations_for_user(mode=mode, current_user=current_user)

# ------------------------------
# POST endpoint — generate new recommendations
# ------------------------------
@router.post("/recommendations/generate", response_model=dict)
async def generate_recommendations_for_user(
    mode: str = Query("traditional"),
    current_user: str = Depends(verify_token)
):
    email = current_user
    try:
        result = await run_a2a_recommendations(email, mode)
        return result
    except Exception as e:
        print("Error in generate_recommendations_for_user:", str(e))
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")