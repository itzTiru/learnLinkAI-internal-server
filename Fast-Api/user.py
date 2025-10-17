# user.py (no changes needed, but confirm prefix removed in main.py)
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from fastapi.security import HTTPBearer
import jwt
import motor.motor_asyncio
from dotenv import load_dotenv
from bson import ObjectId
from datetime import datetime
from typing import List

load_dotenv()

# Reuse from your existing setup
MONGODB_URI = "mongodb+srv://education:education123@cluster0.nyv76s4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"  # From .env
DB_NAME = "education"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]

JWT_SECRET = "your-very-secure-secret-key-change-this-in-production"  # From .env
JWT_ALGORITHM = "HS256"
security = HTTPBearer()

async def verify_token(credentials: HTTPBearer = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("email")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def _doc_to_personal_out(doc: dict) -> dict:
    if not doc:
        return None
    doc = doc.copy()
    doc["_id"] = str(doc.get("_id"))

    for edu in doc.get("education", []):
        edu["id"] = str(edu.get("_id", ObjectId())) if "_id" in edu else None
        edu.pop("_id", None)
    for work in doc.get("work", []):
        work["id"] = str(work.get("_id", ObjectId())) if "_id" in work else None
        work.pop("_id", None)
    for proj in doc.get("projects", []):
        proj["id"] = str(proj.get("_id", ObjectId())) if "_id" in proj else None
        proj.pop("_id", None)
    return doc

router = APIRouter(tags=["user"], prefix="/user")


@router.get("/profile", response_model=dict)
async def get_profile(current_user: str = Depends(verify_token)):
    doc = await db.personal.find_one({"email": current_user})
    if not doc:
        raise HTTPException(status_code=404, detail="Profile not found")

    profile = _doc_to_personal_out(doc)
  
    profile["completed_courses"] = profile.get("completed_courses", 0)
    profile["study_hours"] = profile.get("study_hours", 0)
    profile["topics_explored"] = profile.get("topics_explored", 0)
    return {"profile": profile}

class SearchOut(BaseModel):
    query: str
    timestamp: datetime

@router.get("/recent-searches", response_model=dict)
async def get_recent_searches(current_user: str = Depends(verify_token)):
    cursor = db.searches.find({"user_email": current_user}).sort("timestamp", -1).limit(5)
    docs = await cursor.to_list(length=5)
    searches = [{"query": doc["query"]} for doc in docs]
    return {"searches": searches}