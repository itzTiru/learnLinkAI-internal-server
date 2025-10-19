from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from models.request_models import UserRequest, AnswerSubmission
import orchestrator
from pydantic import BaseModel
from typing import List
import os
import asyncio
import httpx
import re
import unicodedata
import torch
import json
from fastapi import File, UploadFile
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from database import get_db, Content
from googleapiclient.discovery import build
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers.util import cos_sim
import roadmap


load_dotenv()

app = FastAPI(title="Educational Content Recommender")

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSEMBLY_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# API URLs
ASSEMBLY_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLY_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

# Clients
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")


# Utility Functions
def clean_text(text: str) -> str:
    """Normalize & clean input text for better embeddings & search"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # collapse spaces
    return text.strip().lower()

# Models
class SearchQuery(BaseModel):
    query: str
    max_results: int = 5
    platforms: List[str] = ["youtube", "web"]


# External Fetchers

async def fetch_youtube_videos(query: str, max_results: int):
    try:
        search_response = youtube.search().list(
            q=query, part="id,snippet", maxResults=max_results, type="video"
        ).execute()

        videos = []
        for item in search_response.get("items", []):
            videos.append({
                "platform": "youtube",
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["high"]["url"]
            })
        return videos
    except Exception as e:
        print("YouTube fetch error:", e)
        return []


async def fetch_google_web(query: str, max_results: int):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&num={max_results}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        data = response.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "platform": "web",
                "title": item["title"],
                "description": item.get("snippet", ""),
                "url": item["link"],
                "thumbnail": None
            })
        return results
    except Exception as e:
        print("Google fetch error:", e)
        return []


async def fetch_gemini_answers(query: str):
    try:
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {GEMINI_API_KEY}"}
        body = {
            "contents": [{
                "parts": [{
                    "text": f"Provide short, educationally useful answers with references (as URLs) for the following query: {query}"
                }]
            }]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                headers=headers,
                json=body,
            )
        data = response.json()

        answers = []
        for cand in data.get("candidates", []):
            text = cand["content"]["parts"][0]["text"]
            ref = ""
            match = re.search(r"https?://\S+", text)
            if match:
                ref = match.group(0)
            answers.append({"answer": text, "reference": ref})
        return answers
    except Exception as e:
        print("Gemini API error:", e)
        return []

# Ranking
def rank_by_similarity(query: str, results: List[dict]) -> List[dict]:
    try:
        query_emb = embedding_model.embed_query(query)
        contents = [f"{r['title']} {r['description']}" for r in results]
        doc_embs = embedding_model.embed_documents(contents)
        sims = [cos_sim(torch.tensor(query_emb), torch.tensor(doc)).item()
                for doc in doc_embs]
        for r, s in zip(results, sims):
            r["similarity"] = s
        return sorted(results, key=lambda x: x["similarity"], reverse=True)
    except Exception as e:
        print("Ranking error:", e)
        return results


# Ranking
def rank_by_similarity(query: str, results: List[dict]) -> List[dict]:
    try:
        query_emb = embedding_model.embed_query(query)
        contents = [f"{r['title']} {r['description']}" for r in results]
        doc_embs = embedding_model.embed_documents(contents)
        sims = [cos_sim(torch.tensor(query_emb), torch.tensor(doc)).item()
                for doc in doc_embs]
        for r, s in zip(results, sims):
            r["similarity"] = s
        return sorted(results, key=lambda x: x["similarity"], reverse=True)
    except Exception as e:
        print("Ranking error:", e)
        return results

# Routes

@app.post("/search")
async def search_content(payload: SearchQuery, db: Session = Depends(get_db)):
    try:
        query_clean = clean_text(payload.query)
        if not query_clean:
            return {"query": payload.query, "results": []}

        tasks = []
        if "youtube" in payload.platforms:
            tasks.append(fetch_youtube_videos(
                query_clean, payload.max_results))
        if "web" in payload.platforms:
            tasks.append(fetch_google_web(query_clean, payload.max_results))

        results_lists = await asyncio.gather(*tasks) if tasks else []
        results = [item for lst in results_lists for item in (lst or [])]

        if not results:
            return {"query": payload.query, "results": []}

        ranked = rank_by_similarity(query_clean, results)

        # Upsert into DB
        for res in ranked[: payload.max_results]:
            existing = db.query(Content).filter(
                Content.url == res["url"]).first()
            if not existing:
                try:
                    emb = embedding_model.embed_query(clean_text(
                        res["title"] + " " + res["description"]))
                except Exception:
                    emb = []
                new_item = Content(
                    platform=res["platform"],
                    title=res["title"],
                    description=res["description"],
                    url=res["url"],
                    thumbnail=res.get("thumbnail"),
                    embedding=json.dumps(emb)
                )
                db.add(new_item)
        db.commit()

        return {"query": payload.query, "results": ranked[: payload.max_results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")



# Gemini API Wrapper
async def fetch_gemini_answers(query: str) -> list:
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite-preview-09-2025:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

        prompt = f"""
        You are an educational assistant. 
        Provide 6 concise, clear, and helpful answers to this query: "{query}"

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
            resp = await client.post(url, headers=headers, json=payload)  # use 'json=payload'
            resp.raise_for_status()
            data = resp.json()

        text_response = data.get("candidates", [{}])[0].get(
            "content", {}).get("parts", [{}])[0].get("text", "")

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

        return answers[:6]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini fetch error: {e}")


@app.post("/aiinfo")
async def ai_info(payload: SearchQuery):
    try:
        answers = await fetch_gemini_answers(payload.query)

        results = []
        for ans in answers:
            ref = ans.get("reference")
            if ref and ref.startswith("https:/") and not ref.startswith("https://"):
                ref = ref.replace("https:/", "https://", 1)
            results.append({
                "platform": "ai",
                "title": ans["answer"],
                "description": ans["answer"],
                "url": ref,
                "thumbnail": None
            })

        return {"query": payload.query, "results": results}

    except Exception as e:
        import traceback
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"AI Info error: {e}")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        headers = {"authorization": ASSEMBLY_API_KEY,
                   "content-type": "application/octet-stream"}
        async with httpx.AsyncClient() as client:
            upload_resp = await client.post(ASSEMBLY_UPLOAD_URL, headers=headers, content=await file.read())
        audio_url = upload_resp.json()["upload_url"]

        transcript_req = {"audio_url": audio_url}
        async with httpx.AsyncClient() as client:
            trans_resp = await client.post(ASSEMBLY_TRANSCRIPT_URL, headers={"authorization": ASSEMBLY_API_KEY}, json=transcript_req)
        transcript_id = trans_resp.json()["id"]

        while True:
            async with httpx.AsyncClient() as client:
                poll_resp = await client.get(f"{ASSEMBLY_TRANSCRIPT_URL}/{transcript_id}", headers={"authorization": ASSEMBLY_API_KEY})
            status = poll_resp.json()["status"]
            if status == "completed":
                return {"text": clean_text(poll_resp.json().get("text", ""))}
            elif status == "error":
                raise Exception(poll_resp.json()["error"])
            await asyncio.sleep(3)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Transcription error: {e}")

# ---------------------------
# Health Check
# ---------------------------
@app.get("/")
async def root():
    return {"message": "Educational Content Recommender API running "}

# Parallel orchestration of 7 agents
@app.post("/start_session")
async def start_session(req: UserRequest):
    result = orchestrator.start_session(req.user_id, req.query)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/submit_answers")
async def submit_answers(sub: AnswerSubmission):
    result = orchestrator.submit_answers(sub.session_id, sub.answers)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    ctx = orchestrator.get_context(session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="Session not found")
    return ctx
  
# ---------------------------
# Pdf transcribe
# ---------------------------

from fastapi import FastAPI, UploadFile, File
import os
from pdf_utils import extract_text_from_pdf
from agents.pdf_query import analyze_pdf_text, parse_analysis

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        text = extract_text_from_pdf(temp_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")

        # Analyze large PDF with chunking
        result = analyze_pdf_text(text)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
class QuestionRequest(BaseModel):
    text: str
    question: str

@app.post("/ask-pdf/")
async def ask_pdf(request: QuestionRequest):
    """
    User asks a question about the uploaded PDF.
    We feed the full text + question into Gemini for contextual answer.
    """
    # Use GenerativeModel for generating content
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = (
        "You are a helpful assistant for understanding PDFs.\n"
        "Answer the question **based only** on the provided document content.\n"
        "If the answer is not present, say 'The document does not mention this clearly.'\n\n"
        f"Document Text:\n{request.text[:8000]}\n\n"
        f"User Question: {request.question}"
    )

    response = model.generate_content(prompt)
    clean_answer = re.sub(r'\*\*(.*?)\*\*', r'\1', response.text.strip())
    return {"answer": clean_answer}
  

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

from chat import router as chat_router
app.include_router(chat_router)

# Roadmap endpoints
@app.get("/roadmap/categories")
async def get_categories():
    return roadmap.get_categories_endpoint()

@app.post("/roadmap/generate")
async def generate_roadmap(request: roadmap.RoadmapRequest):
    return roadmap.generate_roadmap_endpoint(request)


# Working 1.0