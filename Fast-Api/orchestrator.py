# orchestrator.py
import uuid
import logging
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.context import AgentContext
from services.sqlite_store import save_context, load_context
from agents.safety import run_safety
from agents.classifier import run_classifier
from agents.style_detector import detect_style
from agents.learner_understanding import generate_mcqs, evaluate_answers
from agents.roadmap import generate_roadmap
from agents.content import recommend_content
from agents.explainability import generate_explanations

logger = logging.getLogger(__name__)

def start_session(user_id: str, query: str) -> dict:
    session_id = str(uuid.uuid4())

    try:
        # 1️ Safety
        safety = run_safety(query)
        if safety.get("status") != "safe":
            return {"error": "Query blocked by safety filter", "details": safety.get("reasons", [])}

        # 2️ Classification
        classification = run_classifier(query)
        domain = classification.get("domain", "general")
        intent = classification.get("intent", "learn")

        # 3️ Style detection
        style = detect_style(query, user_profile=None)

        # 4️ Generate MCQs
        mcqs = generate_mcqs(query, domain, n=5)

        context = {
            "session_id": session_id,
            "user_id": user_id,
            "query": query,
            "stage": "awaiting_answers",
            "mcqs": mcqs,
            "answers": {},
            "style": style,
            "difficulty": None,
            "roadmap": [],
            "recommendations": {},
            "explanations": []
        }

        save_context(session_id, context)

        # Remove correct answers before sending to frontend
        mcqs_public = [{**q, 'correct': None} for q in mcqs]

        return {"session_id": session_id, "mcqs": mcqs_public, "domain": domain, "intent": intent, "style": style}

    except ValueError as e:
        logger.error(f"ValueError in start_session: {e}")
        return {"error": "Invalid input or generation failed. Please try a different query."}
    except Exception as e:
        logger.error(f"Unexpected error in start_session: {e}")
        return {"error": "Internal service error. Please try again later."}


def submit_answers(session_id: str, answers: dict) -> dict:
    # Load context
    context = load_context(session_id)
    if not context:
        return {"error": "Invalid session id"}

    try:
        # 1️ Evaluate answers
        eval_result = evaluate_answers(context["mcqs"], answers)
        context["difficulty"] = eval_result
        context["answers"] = answers

        # 2️ Classification (re-run for consistency)
        classification = run_classifier(context["query"])
        domain = classification.get("domain", "general")
        difficulty = eval_result.get("difficulty")
        style = context.get("style", "mixed")

        # 3️ Generate roadmap
        roadmap = generate_roadmap(context["query"], difficulty, style, domain)
        context["roadmap"] = roadmap

        # 4️ PARALLEL: Generate recommendations for all steps at once
        logger.info(f"Starting parallel content generation for {len(roadmap)} steps...")
        recommendations = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all content generation tasks
            future_to_step = {
                executor.submit(recommend_content, [step], domain, difficulty, 5): step["title"]
                for step in roadmap
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_step):
                step_title = future_to_step[future]
                try:
                    result = future.result()
                    recommendations.update(result)
                    logger.info(f"✓ Completed recommendations for '{step_title}'")
                except Exception as e:
                    logger.error(f"✗ Failed to generate recommendations for '{step_title}': {e}")
                    # Add fallback for this step
                    recommendations[step_title] = [
                        {
                            "title": f"Google Search: {step_title}",
                            "url": f"https://www.google.com/search?q={step_title.replace(' ', '+')}+tutorial",
                            "description": "Search Google for tutorials on this topic.",
                            "tags": ["search", "general"]
                        }
                    ]
        
        context["recommendations"] = recommendations
        logger.info(f"Completed all content recommendations")

        # 5️ PARALLEL: Generate explanations for all steps at once
        logger.info(f"Starting parallel explanation generation for {len(roadmap)} steps...")
        explanations = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_step = {
                executor.submit(generate_single_explanation, step, eval_result, answers, style): step
                for step in roadmap
            }
            
            for future in as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    explanation = future.result()
                    explanations.append(explanation)
                    logger.info(f"✓ Completed explanation for '{step['title']}'")
                except Exception as e:
                    logger.error(f"✗ Failed explanation for '{step['title']}': {e}")
                    explanations.append({
                        "step_id": step.get("id"),
                        "explanation": f"This step builds on your {difficulty} level.",
                        "confidence": 0.6
                    })
        
        context["explanations"] = explanations
        logger.info(f"Completed all explanations")

        context["stage"] = "complete"
        save_context(session_id, context)

        # Remove correct answers before sending to frontend
        public_ctx = context.copy()
        for q in public_ctx.get("mcqs", []):
            q.pop("correct", None)
        
        logger.info(f"✓ Session {session_id} completed successfully")
        return public_ctx

    except ValueError as e:
        logger.error(f"ValueError in submit_answers: {e}")
        return {"error": "Evaluation or generation failed. Please try again."}
    except Exception as e:
        logger.error(f"Unexpected error in submit_answers: {e}")
        return {"error": "Internal service error. Please try again later."}


def generate_single_explanation(step: Dict, difficulty_info: Dict, answers: Dict, style: str) -> Dict:
    """Helper function to generate a single explanation (for parallel execution)"""
    from services.llm_client import call_gemini
    from utils.json_utils import extract_and_parse_json
    
    context = {
        "difficulty": difficulty_info.get("difficulty"),
        "score": difficulty_info.get("score"),
        "total": difficulty_info.get("total"),
        "answers_summary": f"{difficulty_info.get('score')}/{difficulty_info.get('total')}"
    }
    
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
        return parsed
    except Exception as e:
        logger.warning(f"Explanation failed for {step.get('title')}: {e}")
        return {
            "step_id": step.get("id"),
            "explanation": f"This step builds on your {context['difficulty']} level.",
            "confidence": 0.6
        }


def get_context(session_id: str):
    return load_context(session_id)