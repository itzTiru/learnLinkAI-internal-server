# agents/learner_understanding.py
from services.llm_client import call_gemini
from utils.json_utils import extract_and_parse_json  # Uses enhanced version
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def generate_mcqs(topic: str, domain: str, n: int = 5, max_retries: int = 1) -> List[Dict[str, Any]]:
    """
    Generate MCQs with retry on parse failure.
    """
    prompt = (
        f"Generate exactly {n} multiple-choice questions for the topic: '{topic}' in the domain '{domain}'. "
        "Each must have: 'id' (e.g., 'q1'), 'question' (string), 'options' (array of exactly 4 strings), 'correct' (int 0-3). "
        "Ensure STRICTLY valid JSON – no trailing commas, escape quotes in strings (e.g., \"It\\'s\"), no markdown. "
        "Example output: [{'id':'q1','question':'Test?','options':['A','B','C','D'],'correct':0}, ...]"
    )

    for attempt in range(max_retries + 1):
        text = None
        try:
            text = call_gemini(prompt).strip()
            logger.info(f"MCQ generation response (attempt {attempt+1}): {len(text)} chars")
            # TEMP: Log full text for debugging (remove in prod to avoid sensitive logs)
            logger.debug(f"Full MCQ response: {text}")
            mcqs = extract_and_parse_json(text, expected_type="array")
            
            # Validate (with softer errors)
            if len(mcqs) != n:
                raise ValueError(f"Generated {len(mcqs)} MCQs, expected {n}")
            for q in mcqs:
                if not isinstance(q, dict) or "question" not in q or len(q.get("options", [])) != 4:
                    raise ValueError("Invalid MCQ structure")
                if "correct" not in q or not (0 <= q["correct"] < 4):
                    raise ValueError("Invalid 'correct' index")
                if "id" not in q:
                    q["id"] = str(hash(q["question"]))
            logger.info(f"Successfully generated {n} valid MCQs")
            return mcqs
        except Exception as e:
            logger.warning(f"MCQ attempt {attempt+1} failed: {e}")
            if attempt == max_retries:
                break
            # Retry with same prompt (could vary temperature if needed)

    # Fallback after retries
    logger.error(f"All {max_retries+1} MCQ attempts failed. Using dummies. Last text: {text[:500] if text else 'N/A'}...")
    logger.warning("Using fallback dummy MCQs")
    dummies = []
    for i in range(n):
        dummies.append({
            "id": f"q{i+1}",
            "question": f"Placeholder Q{i+1}: Basic {domain} concept on '{topic}'?",
            "options": [f"A: Correct demo answer", f"B: Wrong", f"C: Wrong", f"D: Wrong"],
            "correct": 0
        })
    return dummies

# evaluate_answers unchanged (from previous)
def evaluate_answers(mcqs: List[Dict[str, Any]], answers: Dict[str, Any]) -> Dict[str, Any]:
    total = len(mcqs)
    score = 0

    for q in mcqs:
        qid = str(q.get("id", ""))
        correct_idx = q.get("correct")

        if correct_idx is None or qid == "":
            continue

        user_ans = answers.get(qid)
        if user_ans is None:
            continue

        user_idx = None
        if isinstance(user_ans, int):
            user_idx = user_ans
        elif isinstance(user_ans, str):
            try:
                user_idx = int(user_ans)
            except ValueError:
                try:
                    user_idx = q["options"].index(user_ans)
                except ValueError:
                    user_idx = None

        if user_idx == correct_idx:
            score += 1

    pct = (score / total) if total else 0
    if pct < 0.3:
        difficulty = "beginner"
    elif pct < 0.8:
        difficulty = "intermediate"
    else:
        difficulty = "advanced"

    return {"score": score, "total": total, "difficulty": difficulty}