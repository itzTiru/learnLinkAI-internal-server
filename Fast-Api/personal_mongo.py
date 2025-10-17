import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
import motor.motor_asyncio
from dotenv import load_dotenv
from bson import ObjectId
import jwt
from fastapi.security import HTTPBearer

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://education:education123@cluster0.nyv76s4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
DB_NAME = os.getenv("MONGO_DBNAME", "education")

# Async Mongo client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]

router = APIRouter(tags=["personal"])

# JWT Config
JWT_SECRET = os.getenv("JWT_SECRET", "your-very-secure-secret-key-change-this-in-production")
JWT_ALGORITHM = "HS256"

security = HTTPBearer()

async def verify_token(credentials: HTTPBearer = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("email")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------- Schemas ----------
class EducationIn(BaseModel):
    university: str
    degree: str
    field: str
    start_year: int
    end_year: int

class WorkIn(BaseModel):
    company: str
    role: str
    field: str
    start_year: int
    end_year: int

# New: Project Schema
class ProjectIn(BaseModel):
    name: str
    description: str
    technologies: str  # e.g., "React, Node.js"
    start_year: int
    end_year: int

class ProjectOut(ProjectIn):
    id: Optional[str] = None

class PersonalIn(BaseModel):
    name: str
    email: EmailStr
    age: Optional[int] = None
    education: List[EducationIn] = []
    work: List[WorkIn] = []
    projects: List[ProjectIn] = []  # New: Projects field
    bookmarks: List[str] = []

class EducationOut(EducationIn):
    id: Optional[str] = None

class WorkOut(WorkIn):
    id: Optional[str] = None

class PersonalOut(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    age: Optional[int] = None
    email: EmailStr
    education: List[EducationOut] = []
    work: List[WorkOut] = []
    projects: List[ProjectOut] = []  # New: Projects field
    bookmarks: List[str] = []

    class Config:
        allow_population_by_field_name = True

# ---------- Helper ----------
def _doc_to_personal_out(doc: dict) -> dict:
    if not doc:
        return None
    doc = doc.copy()
    doc["_id"] = str(doc.get("_id"))
    # Handle subdocs IDs (Education, Work, Projects)
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

# ---------- Routes (unchanged except for projects support) ----------
@router.post("/personal/save", response_model=dict)
async def save_personal(payload: PersonalIn, current_user: str = Depends(verify_token)):
    if current_user != payload.email:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    doc = payload.dict(exclude_unset=True)
    try:
        await db.personal.update_one(
            {"email": payload.email}, {"$set": doc}, upsert=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    saved = await db.personal.find_one({"email": payload.email})
    return {"message": "saved", "profile": _doc_to_personal_out(saved)}

@router.get("/personal/{email}", response_model=PersonalOut)
async def get_personal(email: str, current_user: Optional[str] = Depends(verify_token)):
    print(f"Fetching profile for: {email}")

    if current_user and current_user != email:
        raise HTTPException(status_code=403, detail="Unauthorized to view this profile")
    
    doc = await db.personal.find_one({"email": email})
    print(f"Fetched doc: {doc}")
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")
    return _doc_to_personal_out(doc)

@router.delete("/personal/{email}", response_model=dict)
async def delete_personal(email: str, current_user: str = Depends(verify_token)):
    if current_user != email:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    res = await db.personal.delete_one({"email": email})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "deleted"}