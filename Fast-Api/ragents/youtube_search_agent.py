# ragents/youtube_search_agent.py
from config import YOUTUBE_API_KEY
from googleapiclient.discovery import build
import asyncio
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchYouTubeInput(BaseModel):
    query: str = Field(..., description="Search query for YouTube")

@tool(args_schema=SearchYouTubeInput)
async def search_youtube(query: str) -> list:
    """Search YouTube for educational videos based on the query."""
   
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoCategoryId="27",  # Education category
            maxResults=3
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, request.execute)
        items = response.get("items", []) or []
        results = [
            {
                "platform": "youtube",
                "title": it["snippet"]["title"],
                "description": it["snippet"]["description"],
                "url": f"https://www.youtube.com/watch?v={it['id']['videoId']}",
                "thumbnail": it["snippet"]["thumbnails"]["default"]["url"],
                "type": "video",
                "relevance_score": 0.95
            }
            for it in items
        ]
       
        return results
    except Exception as e:
        print("YouTube search error:", str(e))
        return []