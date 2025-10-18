from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import httpx
from typing import Dict, List
import logging
import jwt
from datetime import datetime

load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])
security = HTTPBearer()

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set in environment")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-here-change-in-prod")

# In-memory chat history
chat_history: Dict[str, List[Dict[str, str]]] = {}

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Token verification
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    if credentials.scheme != "Bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme.")
    
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("email")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token missing user ID")
        logger.info(f" Token validated for user: {user_id}")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


async def is_educational_message(message: str) -> bool:
    """
    Uses Gemini API to classify whether a user message is educational.
    """
    prompt = (
        "Classify the following user message as either 'educational' or 'non-educational'. "
        "Educational means anything related to learning, studying, knowledge, research, "
        "teaching, school, or academic subjects. "
        "Respond with only one word: educational or non-educational.\n\n"
        f"User message: \"{message}\""
    )

    data = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0}
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json=data
        )

    if response.status_code != 200:
        logger.error(f"Gemini classification failed: {response.status_code} - {response.text}")
        raise HTTPException(status_code=500, detail="AI classification error.")

    try:
        result = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
        logger.info(f"🎓 Gemini classified message as: {result}")
        return "educational" in result
    except Exception as e:
        logger.error(f"Failed to parse classification result: {e}")
        raise HTTPException(status_code=500, detail="Failed to interpret AI classification.")


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user_id: str = Depends(verify_token)
):
    # 🧩 Step 1: Check if educational via Gemini
    educational = await is_educational_message(request.message)
    if not educational:
        return ChatResponse(
            response="⚠️ This assistant only supports educational questions. "
                     "Please ask something related to learning or academic topics."
        )

    # 🧩 Step 2: Maintain chat history
    if user_id not in chat_history:
        chat_history[user_id] = []

    chat_history[user_id].append({"role": "user", "content": request.message})

    # 🧩 Step 3: Build prompt for educational context
    system_prompt = (
        "You are an educational AI assistant for a learning platform. "
        "You should respond clearly and helpfully to questions related to learning, "
        "academic subjects, study tips, or general education. "
        "Be encouraging, friendly, and concise."
    )

    contents = []
    for msg in chat_history[user_id]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    data = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.7}
    }

    # 🧩 Step 4: Send to Gemini for response
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json"
            },
            json=data
        )

    if response.status_code != 200:
        logger.error(f"Gemini response failed: {response.status_code} - {response.text}")
        raise HTTPException(status_code=500, detail="AI service error.")

    gemini_response = response.json()
    if "candidates" not in gemini_response or not gemini_response["candidates"]:
        raise HTTPException(status_code=500, detail="Invalid AI response structure.")

    try:
        content = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(status_code=500, detail="Failed to parse AI response.")

    # 🧩 Step 5: Save and return
    chat_history[user_id].append({"role": "assistant", "content": content})
    logger.info(f"✅ Educational response generated for user {user_id}")

    return ChatResponse(response=content)
