# agents/content.py
import json
from services.sqlite_store import search_resources
from services.llm_client import call_gemini
from utils.json_utils import extract_and_parse_json
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def recommend_content(roadmap: List[Dict], domain: str, difficulty: str, top_k: int = 3) -> Dict[str, List[Dict]]:
    recommendations = {}
    
    for step in roadmap:
        step_title = step["title"]
        resources = search_resources(step_title, limit=top_k)
        
        if not resources:
            prompt = (
                f"Generate exactly {top_k} educational resources for learning '{step_title}' "
                f"in the {domain} domain at {difficulty} level.\n\n"
                "Return ONLY a JSON array with this EXACT structure:\n"
                "[\n"
                '  {"title": "Resource Name", "url": "https://example.com/path", '
                '"description": "Brief one-sentence description.", "tags": ["tag1", "tag2"]},\n'
                "  ...\n"
                "]\n\n"
                "Rules:\n"
                "- Output ONLY the JSON array, no markdown, no explanations\n"
                "- Use real, working URLs (Khan Academy, Coursera, YouTube, MDN, etc.)\n"
                "- Escape ALL quotes inside strings with backslash\n"
                "- NO trailing commas\n"
                f"- Return EXACTLY {top_k} resources\n\n"
                "Example:\n"
                '[{"title":"Khan Academy Basics","url":"https://www.khanacademy.org/computing",'
                '"description":"Free interactive lessons.","tags":["free","interactive"]}]'
            )
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    text = call_gemini(prompt, max_tokens=600, temperature=0.2).strip()
                    logger.info(f"Content LLM response for '{step_title}' (attempt {attempt+1}): {len(text)} chars")
                    
                    # Parse and validate
                    resources = extract_and_parse_json(text, expected_type="array")
                    
                    if len(resources) < 1:
                        raise ValueError(f"Got {len(resources)} resources, need at least 1")
                    
                    # Validate structure
                    valid_resources = []
                    for res in resources:
                        if not isinstance(res, dict):
                            continue
                        if not all(k in res for k in ["title", "url", "description", "tags"]):
                            logger.warning(f"Resource missing required fields: {res}")
                            continue
                        if not res["url"].startswith("http"):
                            res["url"] = "https://www.google.com/search?q=" + res["title"].replace(" ", "+")
                        valid_resources.append(res)
                    
                    if valid_resources:
                        resources = valid_resources[:top_k]
                        logger.info(f"Successfully generated {len(resources)} resources for '{step_title}'")
                        break
                    else:
                        raise ValueError("No valid resources after filtering")
                        
                except Exception as e:
                    logger.warning(f"Content generation attempt {attempt+1} failed for '{step_title}': {e}")
                    if attempt == max_retries - 1:
                        # Final fallback with better placeholders
                        logger.error(f"All attempts failed for '{step_title}'. Using curated fallbacks.")
                        resources = [
                            {
                                "title": f"Google Search: {step_title}",
                                "url": f"https://www.google.com/search?q={step_title.replace(' ', '+')}+tutorial",
                                "description": "Search Google for tutorials and guides on this topic.",
                                "tags": ["search", "general"]
                            },
                            {
                                "title": f"YouTube: {step_title}",
                                "url": f"https://www.youtube.com/results?search_query={step_title.replace(' ', '+')}+tutorial",
                                "description": "Find video tutorials on YouTube.",
                                "tags": ["video", "tutorial"]
                            },
                            {
                                "title": f"Khan Academy: {domain.title()}",
                                "url": "https://www.khanacademy.org/",
                                "description": "Free educational resources and courses.",
                                "tags": ["free", "interactive"]
                            }
                        ][:top_k]
        
        recommendations[step_title] = resources[:top_k]
    
    return recommendations