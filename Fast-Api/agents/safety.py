# agents/safety.py
import re

PROHIBITED_PATTERNS = [
    r"bypass", r"jailbreak", r"ignore previous instructions", r"exploit", r"hack", r"illegal", r"nsfw", r"bomb", r"malware"
]

def run_safety(query: str) -> dict:
    lowered = query.lower()
    for p in PROHIBITED_PATTERNS:
        if re.search(p, lowered):
            return {"status": "unsafe", "reasons": [f"Matched pattern: {p}"]}

    # Bypass LLM safety check for development/testing
    return {"status": "safe", "reasons": []}
