# agents/style_detector.py
from services.llm_client import call_gemini
import logging

logger = logging.getLogger(__name__)

def detect_style(query: str, user_profile: dict = None) -> str:
    profile_text = f"\nUser profile: {user_profile}" if user_profile else ""
    prompt = (
        "Decide the most likely learning style for the user from the choices: visual, auditory, reading, kinesthetic, mixed.\n"
        "Reply with a single word (visual|auditory|reading|kinesthetic|mixed).\n\n"
        f"Query: {query}\n{profile_text}\nAnswer:"
    )
    try:
        text = call_gemini(prompt, max_tokens=10, temperature=0.0).strip().lower()
        logger.info(f"Style detector response: {text}")  # For debugging
        for s in ["visual", "auditory", "reading", "kinesthetic", "mixed"]:
            if s in text:
                return s
        return "mixed"
    except Exception as e:
        logger.error(f"Style detector error: {e}")
        return "mixed"  # Graceful fallback