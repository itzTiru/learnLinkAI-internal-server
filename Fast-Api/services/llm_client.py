# services/llm_client.py
import os
import requests
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


GEMINI_API_KEY='AIzaSyDf7ZeG5iwS8hwSiEDFkfqvQIu4hUQc4QY'

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # Configurable via env

def call_gemini(prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """
    Call Gemini API and return the generated text.
    Raises ValueError on API failure for clear errors.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        # Parse nested response
        if "candidates" in data and data["candidates"]:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            logger.info(f"Gemini call successful: {len(text)} chars generated")
            return text
        else:
            raise ValueError("No candidates in Gemini response")
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API request failed: {e}")
        raise ValueError(f"Gemini API error: {e}")
    except (KeyError, IndexError) as e:
        logger.error(f"Gemini response parsing failed: {e}. Raw: {data}")
        raise ValueError(f"Invalid Gemini response: {e}")