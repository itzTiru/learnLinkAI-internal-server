from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from database import get_db
from orchestrator import orchestrate

# Bootstrapping & globals
load_dotenv()

app = FastAPI(title="Educational Content Recommender (Agent-Based)")

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check API keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
if not YOUTUBE_API_KEY or not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise RuntimeError("Missing one or more API keys in .env (YOUTUBE_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID).")

# Models
class SearchQuery(BaseModel):
    query: str
    max_results: int = 5
    platforms: List[str] = ["youtube", "web"]

# API Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the Educational Content Recommender API (Agent-Based)"}

@app.post("/search")
async def search_content(payload: SearchQuery, db: Session = Depends(get_db)):
    """
    Uses agent-based orchestrator for enhanced processing (harvesting, summarization, ranking).
    Stores results in database.
    """
    try:
        result = await orchestrate(
            query=payload.query,
            max_results=payload.max_results,
            platforms=payload.platforms,
            db=db
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")