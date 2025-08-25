from typing import List, Dict, Any
import os, requests

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


# GOOGLE_API_KEY ='AIzaSyCEzNUMQe7OAZXt6K_U0bSs8Rl_T4Rqpq0'
# GOOGLE_CSE_ID ='136302d29adba4a18'

def web_search_sync(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return []
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": max_results
    }
    resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=15)
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
