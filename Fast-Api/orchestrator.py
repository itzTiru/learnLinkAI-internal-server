from typing import Dict, Any, List
import asyncio
from sqlalchemy.orm import Session
from database import Content
from agents.web_search import web_search
from agents.video_search import video_search
from agents.document import harvest_documents
from agents.qa import summarize_documents
from agents.ranking import rank_results
from agents.recommendation import recommend
from embeddings import get_embedding_model

async def orchestrate(query: str, max_results: int = 5, platforms: List[str] = None, db: Session = None) -> Dict[str, Any]:
    platforms = platforms or ["youtube", "web"]

    # Concurrently fetch results
    tasks = []
    if "web" in platforms:
        tasks.append(web_search(query, max_results))
    if "youtube" in platforms:
        tasks.append(video_search(query, max_results))

    fetched_lists = await asyncio.gather(*tasks, return_exceptions=True)
    web_results, video_results = [], []
    for lst in fetched_lists:
        if isinstance(lst, Exception):
            continue
        if lst and lst[0].get("platform") == "web":
            web_results = lst
        else:
            video_results = lst

    # Harvest and summarize web documents
    docs = await asyncio.to_thread(harvest_documents, web_results)
    summaries = summarize_documents(docs)

    # Enhance items with summaries
    summary_map = {s["url"]: s for s in summaries}
    enhanced_items = []
    for r in web_results + video_results:
        s = summary_map.get(r.get("url"), {})
        enhanced_items.append({**r, "summary": s.get("summary", "")})

    # Rank results
    ranked = rank_results(query, enhanced_items, top_k=max_results * 2)  # Fetch extra for filtering

    # Store in database if provided
    stored_new = 0
    if db:
        existing_urls = {u for (u,) in db.query(Content.url).all()}
        embed_model = get_embedding_model()
        to_add = []
        for it in ranked:
            if it["url"] in existing_urls:
                continue
            text = f"{it.get('title','')} {it.get('description','')} {it.get('summary','')}".strip()
            emb = embed_model.encode(text).tolist()
            to_add.append(Content(
                platform=it.get("platform"),
                title=it.get("title"),
                description=it.get("description"),
                url=it.get("url"),
                thumbnail=it.get("thumbnail"),
                summary=it.get("summary"),
                embedding=emb
            ))
        if to_add:
            db.add_all(to_add)
            db.commit()
            stored_new = len(to_add)

    # Recommend final results
    final = recommend(ranked, top_k=max_results)

    return {
        "query": query,
        "counts": {
            "youtube": len(video_results),
            "web": len(web_results),
            "documents": len(docs),
            "ranked": len(ranked),
            "stored_new": stored_new,
            "recommended": len(final),
        },
        "results": final
    }