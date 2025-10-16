# agents/explainability.py
from services.llm_client import call_gemini
from utils.json_utils import extract_and_parse_json  # New
from typing import List, Dict
import json
import logging

logger = logging.getLogger(__name__)

def generate_explanations(roadmap: List[Dict], difficulty_info: Dict, answers: Dict, style: str) -> List[Dict]:
    explanations = []
    context = {
        "difficulty": difficulty_info.get("difficulty"),
        "score": difficulty_info.get("score"),
        "total": difficulty_info.get("total"),
        "answers_summary": f"{difficulty_info.get('score')}/{difficulty_info.get('total')}"
    }

    for step in roadmap:
        prompt = (
            f"Explain why step '{step.get('title')}' fits a {context['difficulty']} learner ({style} style, {context['answers_summary']} score). "
            "Output JSON: {{'step_id':'{step.get('id')}', 'explanation':'1-2 sentence string', 'confidence':0.7}}. "
            "STRICTLY valid JSON object. No markdown, escape quotes. Output raw JSON only."
        )
        try:
            text = call_gemini(prompt, max_tokens=200, temperature=0.2).strip()
            parsed = extract_and_parse_json(text, expected_type="object")
            if parsed.get("step_id") != step.get("id"):
                raise ValueError("Step ID mismatch")
            explanations.append(parsed)
        except Exception as e:
            logger.warning(f"Explanation failed for {step.get('title')}: {e}")
            explanations.append({
                "step_id": step.get("id"),
                "explanation": f"This step builds on your {context['difficulty']} level.",
                "confidence": 0.6
            })
    return explanations