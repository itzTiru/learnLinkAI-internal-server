from typing import List, Dict, Any
import os
import httpx
import logging

# Set up logging
logging.basicConfig(filename='search_errors.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

async def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logging.error("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID")
        return []
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
        results = []
        for item in data.get("items", [])[:max_results]:
            results.append({
                "platform": "web",
                "title": item.get("title"),
                "description": item.get("snippet"),
                "url": item.get("link"),
                "thumbnail": None,
            })
        return results
    except httpx.HTTPStatusError as e:
        logging.error(f"Google Web Search HTTP error: {e.response.status_code} - {e.response.text}")
        return []
    except Exception as e:
        logging.error(f"Web search error for query '{query}': {str(e)}")
        return []