from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
import asyncio
import httpx
import json
import re
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY =  os.getenv("GEMINI_API_KEY")

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

class RelatedTopicsRequest(BaseModel):
    user_query: str

class RelatedTopicsResponse(BaseModel):
    topics: List[str]

class TopicDescriptionRequest(BaseModel):
    topic: str

class TopicDescriptionResponse(BaseModel):
    detailed_description: str


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

async def generate_related_topics(user_query: str) -> List[str]:
    print(f"hareeeeeee {user_query}")
    print(f"GEMINI_API_KEY: {'set' if GEMINI_API_KEY else 'not set'}")
    if not GEMINI_API_KEY:
        fallback_topics = [
            'Advanced Concepts',
            'Beginner Guide',
            'Practical Examples',
            'Deep Dive'
        ]
        return fallback_topics

    try:
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

        prompt = f"""
        Given user interested in: "{user_query}", suggest 4 related learning topics.
        Output ONLY a valid JSON array of strings, like: ["Topic 1", "Topic 2", "Topic 3", "Topic 4"]
        Make topics engaging and educational.
        """

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 200,
                "temperature": 0.7
            }
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(api_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        print(f"enooooooooooooooo {data}")

        generated_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        generated_text = (generated_text or "").strip()

        # Clean markdown if any
        generated_text = clean_markdown(generated_text)
        

        # Extract JSON from code block if present
        json_match = re.search(r'```json\s*(.*?)\s*```', generated_text, re.DOTALL)
        if json_match:
            generated_text = json_match.group(1).strip()

        # Parse JSON
        try:
            topics = json.loads(generated_text)
            if isinstance(topics, list) and len(topics) >= 4:
                return topics[:4]
            elif isinstance(topics, list) and 0 < len(topics) < 4:
                # Pad with fallbacks if fewer
                fallback = [
                    'Advanced Concepts',
                    'Beginner Guide',
                    'Practical Examples',
                    'Deep Dive'
                ]
                return topics + fallback[len(topics):4]
            else:
                raise ValueError("Invalid JSON structure")
        except (json.JSONDecodeError, ValueError) as parse_error:
            print(f"JSON parsing error: {parse_error}")
            # Additional fallback: try to extract strings between quotes
            extracted = re.findall(r'"([^"]+)"', generated_text)
            if len(extracted) >= 4:
                return extracted[:4]
            elif len(extracted) > 0:
                fallback = [
                    'Advanced Concepts',
                    'Beginner Guide',
                    'Practical Examples',
                    'Deep Dive'
                ]
                return extracted + fallback[len(extracted):4]

    except Exception as e:
        print(f"Gemini topics generation error: {e}")

    # Always return fallback if everything fails
    fallback = [
        'Advanced Concepts',
        'Beginner Guide',
        'Practical Examples',
        'Deep Dive'
    ]
    return fallback

async def generate_topic_description(topic: str) -> str:
    if not GEMINI_API_KEY:
        return f"Explore the fundamentals and advanced aspects of {topic}. This topic covers key concepts, practical applications, and resources for deeper learning."

    try:
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

        prompt = f"""
        Generate an educational description for the topic: "{topic}"

        Make it:
        - Educational: Explain key concepts simply.
        - Structured: Start with a summary, add 2-3 key takeaways, suggest next steps.
        - Concise: 150-250 words max.
        - Engaging: Use simple bullet points for takeaways.
        - IMPORTANT: Use PLAIN TEXT ONLY. Do NOT use markdown formatting like **, *, __, #, or other special characters.
        - Use simple "Key Takeaways:" heading and bullet points with • symbol only.
        - Write in clear, readable paragraphs without any formatting symbols.

        Output only the description text in plain format.
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
        if not enhanced_text:
            return f"Learn about {topic}: This topic introduces core principles and practical insights for beginners and experts alike."
        return enhanced_text

    except Exception as e:
        print(f"Gemini topic description error: {e}")
        return f"Discover {topic}: A foundational topic in learning, covering essential principles and real-world applications. Key takeaways include understanding basics, applying concepts, and exploring further resources."


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
        
        raise HTTPException(status_code=500, detail=f"AI Description error: {e}")

@router.post("/related-topics", response_model=RelatedTopicsResponse)
async def get_related_topics(request: RelatedTopicsRequest):
    try:
        topics = await generate_related_topics(request.user_query)
        return RelatedTopicsResponse(topics=topics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Related topics error: {e}")

@router.post("/topic-description", response_model=TopicDescriptionResponse)
async def get_topic_description(request: TopicDescriptionRequest):
    try:
        desc = await generate_topic_description(request.topic)
        return TopicDescriptionResponse(detailed_description=desc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic description error: {e}")