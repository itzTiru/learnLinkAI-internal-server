# ragents/ranking_agent.py
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from sentence_transformers.util import cos_sim
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class RankResultsInput(BaseModel):
    query: dict = Field(..., description="Search query dictionary with 'query' key")
    youtube_results: List[Dict] = Field(..., description="List of YouTube search results")
    web_results: List[Dict] = Field(..., description="List of web search results")

@tool(args_schema=RankResultsInput)
async def rank_results(query: dict, youtube_results: List[Dict], web_results: List[Dict]) -> List[Dict]:
    """Rank search results based on relevance to the query."""
  
    try:
        results = youtube_results + web_results
        if not results:
            print("No results to rank")
            return []
        texts = [f"{it.get('title','')} {it.get('description','')}".strip() for it in results]
        doc_embs = embedding_model.embed_documents(texts)
        doc_embs = torch.tensor(doc_embs, dtype=torch.float32)
        query_emb = embedding_model.embed_query(query.get("query"))
        query_emb = torch.tensor([query_emb], dtype=torch.float32)
        sims = cos_sim(query_emb, doc_embs)[0].cpu().tolist()
        for it, s in zip(results, sims):
            it["relevance_score"] = float(s)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
     
        return results
    except Exception as e:
        print("Ranking error:", str(e))
        return results