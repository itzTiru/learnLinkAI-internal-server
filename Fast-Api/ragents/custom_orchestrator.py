# ragents/custom_orchestrator.py
import json
from asyncio import gather
from ragents.profile_agent import fetch_user_profile
from ragents.query_agent import generate_query
from ragents.youtube_search_agent import search_youtube
from ragents.web_search_agent import search_google_cse
from ragents.ranking_agent import rank_results
from ragents.saver_agent import save_recommendations_tool

async def orchestrate_recommendations(email: str, mode: str):
    """Custom orchestrator that coordinates all agents step-by-step."""
    try:
       
        profile = await fetch_user_profile(email)
        if profile.get("error"):
          
            raise ValueError("Profile not found")
       

      
        query_result = await generate_query.ainvoke({"profile": profile, "mode": mode})
        if isinstance(query_result, str):
           
            query = {"query": query_result}
        else:
            query = query_result
        if not query.get("query"):
           
            raise ValueError("Failed to generate query")
  

  
        youtube_results, web_results = await gather(
            search_youtube.ainvoke({"query": query["query"]}),
            search_google_cse.ainvoke({"query": query["query"]}),
            return_exceptions=True
        )
        youtube_results = [] if isinstance(youtube_results, Exception) else youtube_results
        web_results = [] if isinstance(web_results, Exception) else web_results



        ranked = await rank_results.ainvoke({
            "query": query,
            "youtube_results": youtube_results,
            "web_results": web_results
        })

        transformed = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "type": item.get("type", "video" if item.get("platform", "").lower() == "youtube" else "article"),
                "description": item.get("description", ""),
                "thumbnail": item.get("thumbnail", None),
                "duration": item.get("duration", None),
                "category": item.get("category", "Video" if item.get("type", "video") == "video" else "Article"),
                "platform": item.get("platform", "YouTube" if item.get("type", "video") == "video" else "Web"),
                "relevance_score": int(item.get("relevance_score", 0.95) * 100)
            }
            for item in ranked
        ]

        await save_recommendations_tool.ainvoke({"email": email, "recommendations": transformed})
   
        return {"recommendations": transformed}
    except Exception as e:
 
        if isinstance(e, dict):
            raise ValueError(f"Orchestration failed with dict error: {e}")
        raise ValueError(f"Orchestration failed: {str(e)}")