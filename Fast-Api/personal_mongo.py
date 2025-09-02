# personal_mongo.py
import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr, Field
import motor.motor_asyncio
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://education:education123@cluster0.mo6lwcn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
DB_NAME = os.getenv("MONGO_DBNAME", "education_db")

# Async Mongo client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]

router = APIRouter(tags=["personal"])

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

class PersonalIn(BaseModel):
    name: str
    age: int
    email: EmailStr
    education: List[EducationIn] = []
    work: List[WorkIn] = []

class EducationOut(EducationIn):
    id: Optional[str] = None

class WorkOut(WorkIn):
    id: Optional[str] = None

class PersonalOut(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    age: int
    email: EmailStr
    education: List[EducationOut] = []
    work: List[WorkOut] = []

    class Config:
        allow_population_by_field_name = True


# ---------- Helper ----------
def _doc_to_personal_out(doc: dict) -> dict:
    if not doc:
        return None
    return {
        "_id": str(doc.get("_id")),
        "name": doc.get("name"),
        "age": doc.get("age"),
        "email": doc.get("email"),
        "education": doc.get("education", []),
        "work": doc.get("work", [])
    }


# ---------- Routes ----------
@router.post("/personal/save", response_model=dict)
async def save_personal(payload: PersonalIn):
    doc = payload.dict()
    try:
        await db.personal_details.update_one(
            {"email": payload.email}, {"$set": doc}, upsert=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    saved = await db.personal_details.find_one({"email": payload.email})
    return {"message": "saved", "profile": _doc_to_personal_out(saved)}


@router.get("/personal/{email}", response_model=PersonalOut)
async def get_personal(email: str):
    doc = await db.personal_details.find_one({"email": email})
    print(f"ffffff{doc}")
    if not doc:
        raise HTTPException(status_code=404, detail="User not found")
    return _doc_to_personal_out(doc)


@router.delete("/personal/{email}", response_model=dict)
async def delete_personal(email: str):
    res = await db.personal_details.delete_one({"email": email})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "deleted"}



@router.post("/personal/test_insert", response_model=dict)
async def test_insert():
    test_doc = {
        "name": "Alice Test",
        "age": 25,
        "email": "alice@test.com",
        "education": [
            {"university": "MIT", "degree": "BSc", "field": "CS", "start_year": 2016, "end_year": 2020}
        ],
        "work": [
            {"company": "Google", "role": "SWE", "field": "AI", "start_year": 2020, "end_year": 2023}
        ]
    }
    try:
        result = await db.personal_details.insert_one(test_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {e}")
    
    return {"message": "Test data inserted", "id": str(result.inserted_id)}

