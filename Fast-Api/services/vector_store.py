# services/vector_store.py
from typing import List, Dict

def search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Placeholder vector search. Replace with Pinecone/Weaviate/FAISS code.
    Return sample resource objects.
    """
    results = [
        {"id": "r1", "title": f"{query} — Intro (video)", "url": "https://youtu.be/example", "source": "youtube", "snippet": "Short intro video.", "duration_minutes": 25, "type": "video"},
        {"id": "r2", "title": f"{query} — Article (guide)", "url": "https://example.com/article", "source": "article", "snippet": "Comprehensive article.", "duration_minutes": 15, "type": "article"},
    ]
    return results[:top_k]
