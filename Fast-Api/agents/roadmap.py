# agents/roadmap.py
from services.llm_client import call_gemini
from utils.json_utils import extract_and_parse_json  # New
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def generate_roadmap(query: str, difficulty: str, style: str, domain: str, length: int = 6) -> List[Dict]:
    prompt = (
        f"Generate a personalized learning roadmap for the topic: '{query}'.\n"
        f"User difficulty: {difficulty}. Learning style: {style}. Domain: {domain}. Length: {length} steps.\n"
        "Each step: 'id' (string e.g. 's1'), 'title' (string), 'description' (1-2 lines string), "
        "'estimated_time_hours' (number), 'prereq' (array of ids). "
        "STRICTLY valid JSON array. No markdown, escape quotes, no trailing commas. "
        "Example: [{'id':'s1','title':'Basics','description':'Intro concepts','estimated_time_hours':4,'prereq':[]}, ...]"
        "Output raw JSON only."
    )
    try:
        text = call_gemini(prompt, max_tokens=900, temperature=0.2).strip()
        logger.info(f"Roadmap generation: {len(text)} chars")
        roadmap = extract_and_parse_json(text, expected_type="array")
        if len(roadmap) != length:
            raise ValueError(f"Generated {len(roadmap)} steps, expected {length}")
        for step in roadmap:
            if not all(k in step for k in ["id", "title", "description", "estimated_time_hours", "prereq"]):
                raise ValueError("Invalid step structure")
        return roadmap
    except Exception as e:
        logger.error(f"Roadmap failed: {e}")
        # Fallback linear roadmap
        return [
            {"id": "s1", "title": f"Foundations of {query}", "description": "Basic concepts and terminology.", "estimated_time_hours": 4, "prereq": []},
            {"id": "s2", "title": "Core Concepts", "description": "Important tools and techniques.", "estimated_time_hours": 8, "prereq": ["s1"]},
            {"id": "s3", "title": "Advanced Project", "description": "Hands-on application.", "estimated_time_hours": 12, "prereq": ["s2"]},
        ][:length]  # Truncate to length
    
    #Workin 1.0