from typing import List, Dict, Any
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer


# Shared embedding model instance
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def rank_results(query: str, items: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Rank search results by cosine similarity to the query using sentence-transformers.
    """
    if not items:
        return []

    # Compute embeddings
    query_emb = embedder.encode([query], convert_to_tensor=True)
    texts = [i.get("title", "") + " " + i.get("description", "") + " " + i.get("summary", "") for i in items]
    doc_embs = embedder.encode(texts, convert_to_tensor=True)

    # Cosine similarity (using sentence-transformers)
    sims = cos_sim(query_emb, doc_embs)[0].cpu().tolist()

    # Attach scores
    for i, sim in enumerate(sims):
        items[i]["similarity_score"] = sim

    # Sort by similarity
    ranked = sorted(items, key=lambda x: x["similarity_score"], reverse=True)

    return ranked[:top_k]
