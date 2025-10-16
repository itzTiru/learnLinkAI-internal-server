from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import asyncio
import httpx
from dotenv import load_dotenv
import json
from sqlalchemy.orm import Session
from database import get_db, Content
from googleapiclient.discovery import build

from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from sentence_transformers.util import cos_sim

# ---------------------------
# Bootstrapping & globals
# ---------------------------
load_dotenv()

app = FastAPI(title="Educational Content Recommender (Fast & Batched)")

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not YOUTUBE_API_KEY or not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise RuntimeError("Missing one or more API keys in .env (YOUTUBE_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID).")

# YouTube API client (build once)
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Embeddings (load once; langchain wrapper over sentence-transformers)
# NOTE: If you want even faster cold-starts, set model_kwargs={"device": "cpu"} explicitly,
# or "cuda" if you have a GPU: model_kwargs={"device": "cuda"}
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------
# Models
# ---------------------------
class SearchQuery(BaseModel):
    query: str
    max_results: int = 5
    platforms: List[str] = ["youtube", "web"]


# ---------------------------
# Agents (Async)
# ---------------------------
async def fetch_youtube_videos(query: str, max_results: int) -> List[dict]:
    """
    YouTube search (runs request.execute in a thread to avoid blocking the event loop).
    """
    try:
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoCategoryId="27",  # Education category
            maxResults=max_results
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, request.execute)
        items = response.get("items", []) or []
        return [
            {
                "platform": "youtube",
                "title": it["snippet"]["title"],
                "description": it["snippet"]["description"],
                "url": f"https://www.youtube.com/watch?v={it['id']['videoId']}",
                "thumbnail": it["snippet"]["thumbnails"]["default"]["url"]
            }
            for it in items
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube API error: {e}")


async def fetch_google_web(query: str, max_results: int) -> List[dict]:
    """
    Google Programmable Search (Custom Search API) via httpx.AsyncClient (non-blocking).
    """
    try:
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": max_results
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("https://www.googleapis.com/customsearch/v1", params=params)
            resp.raise_for_status()
            data = resp.json()

        items = data.get("items", []) or []
        return [
            {
                "platform": "web",
                "title": it.get("title"),
                "description": it.get("snippet"),
                "url": it.get("link"),
                "thumbnail": None
            }
            for it in items
        ]
    except httpx.HTTPStatusError as e:
        # Bubble up the actual Google error body to help debugging keys / CSE config
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Web Search error: {e}")


# ---------------------------
# Utils (embeddings & ranking)
# ---------------------------
def batch_embed_texts(texts: List[str]) -> torch.Tensor:
    """
    Use LangChain's HuggingFaceEmbeddings to batch-embed documents and return a torch tensor.
    """
    if not texts:
        return torch.empty((0, 384), dtype=torch.float32)
    # embed_documents returns List[List[float]]
    embs = embedding_model.embed_documents(texts)
    return torch.tensor(embs, dtype=torch.float32)


def embed_query(text: str) -> torch.Tensor:
    """
    Embed a single query; return as shape (1, D) torch tensor.
    """
    q = embedding_model.embed_query(text)
    return torch.tensor([q], dtype=torch.float32)


def rank_by_similarity(query_text: str, items: List[dict]) -> List[dict]:
    """
    Use cosine similarity (sentence-transformers.util.cos_sim) to rank items.
    """
    if not items:
        return []

    texts = [f"{it.get('title','')} {it.get('description','')}".strip() for it in items]
    doc_embs = batch_embed_texts(texts)  # (N, D)
    query_emb = embed_query(query_text)  # (1, D)

    # cos_sim -> shape (1, N)
    sims = cos_sim(query_emb, doc_embs)[0].cpu().tolist()

    # attach scores
    for it, s in zip(items, sims):
        it["similarity_score"] = float(s)

    # sort desc by similarity
    items.sort(key=lambda x: x["similarity_score"], reverse=True)
    return items


# ---------------------------
# API Routes
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Welcome to the Educational Content Recommender API (Fast & Batched)"}


@app.post("/search")
async def search_content(payload: SearchQuery, db: Session = Depends(get_db)):
    """
    Orchestrates:
      - YouTube + Web search concurrently
      - Batched embeddings for ranking
      - Batched DB upserts (commit once)
    """
    try:
        tasks = []
        if "youtube" in payload.platforms:
            tasks.append(fetch_youtube_videos(payload.query, payload.max_results))
        if "web" in payload.platforms:
            tasks.append(fetch_google_web(payload.query, payload.max_results))

        results_lists = await asyncio.gather(*tasks) if tasks else []
        results: List[dict] = [item for lst in results_lists for item in (lst or [])]

        if not results:
            return {
                "query": payload.query,
                "counts": {"youtube": 0, "web": 0, "ranked": 0, "stored_new": 0},
                "results": []
            }

        # Rank with batched embeddings (fast)
        ranked = rank_by_similarity(payload.query, results)

        # Batch upsert into DB (commit once)
        existing_urls = {
            u for (u,) in db.query(Content.url).all()
        }  # grabs all URLs once; for large tables consider filtering by the small result set
        to_add = []
        for it in ranked:
            if it["url"] in existing_urls:
                continue
            # Store the *same* text we embedded above for consistency; recompute quickly here
            text = f"{it.get('title','')} {it.get('description','')}".strip()
            emb = embedding_model.embed_query(text)  # single vector is OK for few new rows
            to_add.append(Content(
                platform=it.get("platform"),
                title=it.get("title"),
                description=it.get("description"),
                url=it.get("url"),
                thumbnail=it.get("thumbnail"),
                embedding=emb
            ))

        stored_new = 0
        if to_add:
            db.add_all(to_add)
            db.commit()
            stored_new = len(to_add)

        # Trim to requested top-K for response
        topk = ranked[: payload.max_results]

        return {
            "query": payload.query,
            "counts": {
                "youtube": sum(1 for r in results if r["platform"] == "youtube"),
                "web": sum(1 for r in results if r["platform"] == "web"),
                "ranked": len(ranked),
                "stored_new": stored_new
            },
            "results": topk
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")
    



# ---------------------------
# Gemini API Wrapper
# ---------------------------
async def fetch_gemini_answers(query: str) -> list:
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }
        prompt = f"""
        You are an educational assistant. 
        Provide 9 concise, clear, and helpful answers to this query: "{query}"

        Each answer should:
        - Be short (2–3 sentences max)
        - End with a reliable reference link (Wikipedia, .edu, official docs, etc.)
        Format as:
        1. Answer — Reference: [URL]
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, headers=headers, content=json.dumps(payload))
            resp.raise_for_status()
            data = resp.json()

        text_response = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

        # Split into multiple answers
        answers = []
        for line in text_response.split("\n"):
            if line.strip() and (line[0].isdigit() or line.startswith("-")):
                parts = line.split("— Reference:")
                if len(parts) == 2:
                    answer_text = parts[0].strip("1234567890.- ")
                    ref_link = parts[1].strip()
                    answers.append({"answer": answer_text, "reference": ref_link})
                else:
                    answers.append({"answer": line.strip(), "reference": None})

        return answers[:9]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini fetch error: {e}")


# ---------------------------
# AI Info Endpoint
# ---------------------------
@app.post("/aiinfo")
async def ai_info(payload: SearchQuery):
    try:
        answers = await fetch_gemini_answers(payload.query)

        results = []
        for ans in answers:
            ref =ans.get("reference","")
            if ref:
                ref = ref.strip("[]") 
                if ref.startswith("https:/") and not ref.startswith("https://"):
                    ref = ref.replace("https:/", "https://", 1)
            results.append({
                "platform": "ai",
                "title": ans["answer"],
                "description": ans["answer"],
                "url": ans["reference"],
                "thumbnail": None
            })

        return {"query": payload.query, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Info error: {e}")


#-----------------
#personal endpoint
#-----------------
from personal_mongo import router as personal_router

app.include_router(personal_router)

from auth import router as auth_router
app.include_router(auth_router, prefix="/api/auth")

from recommendations import router as recommendation_router
app.include_router(recommendation_router)


from user import router as user_router
app.include_router(user_router,prefix="/api")

from ai_description import router as ai_router
app.include_router(ai_router)