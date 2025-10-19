from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import jwt
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import motor.motor_asyncio
import bcrypt
import certifi

load_dotenv()

router = APIRouter(tags=["users"])

JWT_SECRET = os.getenv("JWT_SECRET", "your-very-secure-secret-key-change-this-in-production")
JWT_ALGORITHM = "HS256"
MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb+srv://education:education123@cluster0.nyv76s4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)
DB_NAME = os.getenv("MONGO_DBNAME", "education")

# MongoDB client
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]

# Schemas
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str

# Helper: Get or create profile in "personal" collection
async def get_or_create_profile(email: str, name: str = None):
    profile = await db.personal.find_one({"email": email})
    if not profile:
        profile_doc = {
            "name": name or "User",
            "email": email,
            "age": None,
            "education": [],
            "work": [],
            "bookmarks": []
        }
        await db.personal.insert_one(profile_doc)
        profile = await db.personal.find_one({"email": email})
    return profile

# Registration
@router.post("/register")
async def register(request: RegisterRequest):
    try:
        existing_user = await db.users.find_one({"email": request.email})
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")

        hashed_password = bcrypt.hashpw(request.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        await db.users.insert_one({
            "email": request.email,
            "name": request.name,
            "password": hashed_password
        })
        
        # Map: Auto-create profile
        await get_or_create_profile(request.email, request.name)
        
        payload = {
            "email": request.email,
            "name": request.name,
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

        return {"token": token, "user": {"email": request.email, "name": request.name}}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Login (with merged profile)
@router.post("/login")
async def login(request: LoginRequest):
    try:
        print(f"Debug: Attempting login for {request}")

        user = await db.users.find_one({"email": request.username})
        print(f"Debug: User fetched from DB: {user}")
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        stored_hash = user["password"].encode("utf-8")
        if not bcrypt.checkpw(request.password.encode("utf-8"), stored_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Map: Fetch/create profile
        profile = await get_or_create_profile(user["email"], user["name"])
        
        # Merge for response
        merged_user = {
            "email": user["email"],
            "name": user["name"],
            "age": profile.get("age"),
            "education": profile.get("education", []),
            "work": profile.get("work", [])
        }

        payload = {
            "email": user["email"],
            "name": user["name"],
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

        print(f"Debug: Login successful for {request.username}, merged profile")
        return {"token": token, "user": merged_user}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Debug: Exception occurred - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")