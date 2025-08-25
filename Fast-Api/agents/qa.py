from typing import List, Dict, Any
import re

def simple_summarize(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:max_sentences])

def summarize_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for d in docs:
        summaries.append({
            "url": d["url"],
            "title": d.get("title"),
            "summary": simple_summarize(d.get("text", ""))
        })
    return summaries
