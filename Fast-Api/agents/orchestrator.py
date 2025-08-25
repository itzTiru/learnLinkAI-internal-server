from typing import Dict, Any, List
import asyncio
from agents.web_search import web_search_sync
from agents.video_search import video_search
from agents.document import harvest_documents
from agents.qa import summarize_documents
from agents.ranking import rank_results
from agents.recommendation import recommend

async def orchestrate(query: str, max_results: int = 5, platforms: List[str] = None) -> Dict[str, Any]:
    platforms = platforms or ["youtube", "web"]

    tasks = []
    if "web" in platforms:
        tasks.append(asyncio.to_thread(web_search_sync, query, max_results))
    if "youtube" in platforms:
        tasks.append(video_search(query, max_results))

    fetched_lists = await asyncio.gather(*tasks) if tasks else []
    web_results, video_results = [], []
    for lst in fetched_lists:
        if not lst:
            continue
        if lst and lst[0].get("platform") == "web":
            web_results = lst
        else:
            video_results = lst

    docs = await asyncio.to_thread(harvest_documents, web_results)
    summaries = summarize_documents(docs)

    summary_map = {s["url"]: s for s in summaries}
    enhanced_items = []
    for r in web_results + video_results:
        s = summary_map.get(r.get("url"), {})
        enhanced_items.append({**r, "summary": s.get("summary", "")})

    ranked = rank_results(query, enhanced_items)
    final = recommend(ranked, top_k=max_results)

    return {
        "query": query,
        "counts": {
            "web_results": len(web_results),
            "video_results": len(video_results),
            "documents": len(docs),
            "ranked": len(ranked),
            "recommended": len(final),
        },
        "results": final
    }
