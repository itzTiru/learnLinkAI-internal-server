# ragents/profile_agent.py
from config import db

async def fetch_user_profile(email: str) -> dict:
    profile = await db.personal.find_one({"email": email})
    if not profile:
       
        return {"error": "Profile not found"}
    profile["_id"] = str(profile["_id"])
 
    return profile  