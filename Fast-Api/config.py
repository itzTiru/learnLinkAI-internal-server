# config.py
import os
from dotenv import load_dotenv
import motor.motor_asyncio
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import torch
from sentence_transformers.util import cos_sim
from typing import List

load_dotenv()


MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://education:education123@cluster0.nyv76s4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
DB_NAME = os.getenv("MONGO_DBNAME", "education")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]


YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


JWT_SECRET = os.getenv("JWT_SECRET", "your-very-secure-secret-key-change-this-in-production")
JWT_ALGORITHM = "HS256"

model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def batch_embed_texts(texts: List[str]) -> torch.Tensor:
    if not texts:
        return torch.empty((0, 384), dtype=torch.float32)
    embs = embedding_model.embed_documents(texts)
    return torch.tensor(embs, dtype=torch.float32)

def embed_query(text: str) -> torch.Tensor:
    q = embedding_model.embed_query(text)
    return torch.tensor([q], dtype=torch.float32)

def rank_by_similarity(query_text: str, items: List[dict]) -> List[dict]:
    if not items:
        return []
    texts = [f"{it.get('title','')} {it.get('description','')}".strip() for it in items]
    doc_embs = batch_embed_texts(texts)
    query_emb = embed_query(query_text)
    sims = cos_sim(query_emb, doc_embs)[0].cpu().tolist()
    for it, s in zip(items, sims):
        it["similarity_score"] = float(s)
    items.sort(key=lambda x: x["similarity_score"], reverse=True)
    return items