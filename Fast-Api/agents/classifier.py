# agents/classifier.py
from services.llm_client import call_gemini
from utils.json_utils import extract_and_parse_json  # New import
import logging

logger = logging.getLogger(__name__)

DOMAIN_CATEGORIES = [
    "programming", "data-science", "math", "physics", "chemistry", "biology",
    "history", "languages", "art", "music", "business", "finance", "general"
]

def run_classifier(query: str) -> dict:
    prompt = (
        "Classify the user's query into one educational domain and an intent.\n"
        "Return only JSON like: {\"domain\": \"programming\", \"intent\": \"learn\"}. "
        "Output raw JSON without markdown or code blocks.\n\n"
        "Possible domains: " + ", ".join(DOMAIN_CATEGORIES) + "\n\n"
        f"Query: {query}\n\nAnswer:"
    )
    try:
        text = call_gemini(prompt, max_tokens=200, temperature=0.0).strip()
        logger.info(f"Classifier LLM response: {text[:50]}...")
        parsed = extract_and_parse_json(text, expected_type="object")
        return {"domain": parsed.get("domain", "general"), "intent": parsed.get("intent", "learn")}
    except ValueError as e:
        logger.warning(f"JSON parse failed in classifier: {e}. Using fallback.")
        # Enhanced fallback heuristics
        low = query.lower()
        if any(k in low for k in ("python", "java", "javascript", "program")):
            return {"domain": "programming", "intent": "learn"}
        if "history" in low or "ww2" in low:
            return {"domain": "history", "intent": "learn"}
        if any(k in low for k in ("math", "tensor", "algebra", "calculus")):
            return {"domain": "math", "intent": "learn"}
        return {"domain": "general", "intent": "learn"}
    except Exception as e:
        logger.error(f"Classifier error: {e}")
        return {"domain": "general", "intent": "learn"}