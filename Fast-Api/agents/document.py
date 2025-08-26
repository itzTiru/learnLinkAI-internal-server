from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup

def extract_main_text(html: str) -> str:
    """Extract text content from HTML using BeautifulSoup only (no readability)."""
    try:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator=" ", strip=True)
        return text
    except Exception:
        return ""

def harvest_documents(results: List[Dict[str, Any]], max_bytes: int = 35000) -> List[Dict[str, Any]]:
    """Download and extract text for web results (skip YouTube)."""
    docs = []
    for r in results:
        if r.get("platform") != "web":
            continue
        url = r.get("url")
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            text = extract_main_text(resp.text)[:max_bytes]
            if text:
                docs.append({
                    "url": url,
                    "title": r.get("title"),
                    "text": text
                })
        except Exception:
            continue
    return docs