# ragents/web_search_agent.py
from config import GOOGLE_API_KEY, GOOGLE_CSE_ID
import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchGoogleInput(BaseModel):
    query: str = Field(..., description="Search query for Google CSE")

@tool(args_schema=SearchGoogleInput)
async def search_google_cse(query: str) -> list:
    """Search Google Custom Search Engine for educational content."""
    
    try:
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "q": query,
            "num": 3
        }
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get("https://www.googleapis.com/customsearch/v1", params=params)
            resp.raise_for_status()
            data = resp.json()
        items = data.get("items", []) or []
        results = [
            {
                "platform": "web",
                "title": it.get("title"),
                "description": it.get("snippet"),
                "url": it.get("link"),
                "thumbnail": None,
                "type": "article",
                "relevance_score": 0.95
            }
            for it in items
        ]
        return results
    except Exception as e:
        print("Google CSE search error:", str(e))
        return []