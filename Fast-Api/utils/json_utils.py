# utils/json_utils.py
import re
import json
import logging
from typing import Any, Union, List, Dict

logger = logging.getLogger(__name__)

def repair_json(text: str) -> str:
    """
    Enhanced repair for common LLM JSON malformations.
    - Removes trailing commas in arrays/objects.
    - Unescapes common quote issues (e.g., fix "It's" if needed).
    - Fixes invalid escape sequences by escaping lone backslashes.
    Returns repaired string.
    """
    # Remove trailing commas before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Fix invalid escapes: replace lone \ with \\ (if not preceded by \ and not followed by valid escape char)
    text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
    # Basic unescape: Fix potential "It's" → "It\'s" (but since LLM should escape, this is rare)
    text = re.sub(r'"(It\'s|Don\'t|Can\'t|Won\'t|Shouldn\'t)"', r'"\1"', text)  # Placeholder; expand if needed
    # Strip any lingering whitespace/newlines that could confuse
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text.strip()

def extract_and_parse_json(text: str, expected_type: str = "object_or_array") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract and repair JSON from LLM response.
    """
    if not text:
        raise ValueError("Empty input text")

    # Strip markdown (unchanged)
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\n?```(?:json)?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^```\s*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = text.strip()

    if not text:
        raise ValueError("Text empty after stripping markdown")

    # Repair syntax
    text = repair_json(text)
    logger.debug(f"Repaired JSON text (first 100 chars): {text[:100]}...")

    try:
        parsed = json.loads(text)
        
        # Type validation (unchanged)
        if expected_type == "array" and not isinstance(parsed, list):
            raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
        elif expected_type == "object" and not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        elif expected_type == "object_or_array":
            if not isinstance(parsed, (dict, list)):
                raise ValueError(f"Expected JSON object or array, got {type(parsed).__name__}")
        
        size = len(str(parsed))
        logger.debug(f"JSON parsed successfully: {type(parsed).__name__} ({size} chars)")
        return parsed
        
    except json.JSONDecodeError as e:
        truncated = text[:200] + "..." if len(text) > 200 else text
        logger.warning(f"JSON parse failed after repair: {e}. Raw (truncated): {truncated}. Line: {e.lineno}, Col: {e.colno}, Pos: {e.pos}")
        raise ValueError(f"Invalid JSON after extraction and repair: {e}")