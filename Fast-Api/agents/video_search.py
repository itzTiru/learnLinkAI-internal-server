from typing import List, Dict, Any
import os
import asyncio
from googleapiclient.discovery import build
import logging

# Set up logging
logging.basicConfig(filename='search_errors.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
_youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY) if YOUTUBE_API_KEY else None

async def video_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    if _youtube is None:
        logging.error("YouTube API client not initialized: Missing YOUTUBE_API_KEY")
        return []
    try:
        request = _youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoCategoryId="27",
            maxResults=max_results
        )
        response = await asyncio.get_event_loop().run_in_executor(None, request.execute)
        items = []
        for item in response.get("items", []):
            items.append({
                "platform": "youtube",
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["default"]["url"],
            })
        return items
    except Exception as e:
        logging.error(f"YouTube API error for query '{query}': {str(e)}")
        return []