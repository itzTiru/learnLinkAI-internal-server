from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import asyncio
import httpx
import json
import re
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY =  os.getenv("GOOGLE_API_KEY")

router = APIRouter(tags=["ai-description"])

class AiDescriptionRequest(BaseModel):
    url: str
    platform: str  
    original_description: str
    user_query: Optional[str] = None

class AiDescriptionResponse(BaseModel):
    detailed_description: str
    original_description: str
    enhanced: bool


def clean_markdown(text: str) -> str:
    """
    Remove markdown formatting from text while preserving structure and readability.
    """
    if not text:
        return text
    
   
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)  
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)     
    text = re.sub(r'\*(.+?)\*', r'\1', text)         
    text = re.sub(r'___(.+?)___', r'\1', text)       
    text = re.sub(r'__(.+?)__', r'\1', text)         
    text = re.sub(r'_(.+?)_', r'\1', text)           
    
 
    text = re.sub(r'^\s*[\*\-\+]\s+', '• ', text, flags=re.MULTILINE)
    

    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
   
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    

    text = re.sub(r'^[\*\-_]{3,}$', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
  
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


async def enhance_description_with_gemini(original_desc: str, url: str, platform: str, user_query: Optional[str] = None) -> str:
    """
    Use Gemini to generate an enhanced description, but if GEMINI_API_KEY is missing
    or request fails, return the original description as a graceful fallback.
    """
    # If no key, return original early (do not raise)
    if not GEMINI_API_KEY:
        # Optionally log to stdout/stderr for debugging
        print("Warning: GEMINI_API_KEY not set — returning original description.")
        return original_desc

    try:
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

        context = f"Platform: {platform}. URL: {url}."
        if user_query:
            context += f" User interested in: {user_query}."

        prompt = f"""
        You are an educational content enhancer. Given this original description from {context}:

        "{original_desc}"

        Generate an enhanced description for a learning platform. Make it:
        - Educational: Explain key concepts simply.
        - Structured: Start with a summary, add 2-3 key takeaways, suggest next steps.
        - Concise: 150-250 words max.
        - Engaging: Use simple bullet points for takeaways.
        - Preserve original info: Build upon it, don't contradict.
        - IMPORTANT: Use PLAIN TEXT ONLY. Do NOT use markdown formatting like **, *, __, #, or other special characters.
        - Use simple "Key Takeaways:" heading and bullet points with • symbol only.
        - Write in clear, readable paragraphs without any formatting symbols.

        Output only the enhanced description text in plain format.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.7
            }
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(api_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        enhanced_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        enhanced_text = (enhanced_text or "").strip()

        # Clean any remaining markdown formatting
        enhanced_text = clean_markdown(enhanced_text)

        # Basic sanitation / fallback
        if not enhanced_text or len(enhanced_text) <= len(original_desc):
            return original_desc
        return enhanced_text

    except Exception as e:
        # Log for debugging, but return original as fallback
        print(f"Gemini enhancement error: {e}")
        return original_desc


@router.post("/ai-description", response_model=AiDescriptionResponse)
async def get_ai_description(request: AiDescriptionRequest):
    try:
        enhanced_desc = await enhance_description_with_gemini(
            request.original_description,
            request.url,
            request.platform,
            request.user_query
        )
        is_enhanced = enhanced_desc != request.original_description and len(enhanced_desc) > len(request.original_description)
        return AiDescriptionResponse(
            detailed_description=enhanced_desc,
            original_description=request.original_description,
            enhanced=is_enhanced
        )
    except Exception as e:
        # Give a readable 500 if something unexpected happens
        raise HTTPException(status_code=500, detail=f"AI Description error: {e}")