import os
from dotenv import load_dotenv
from google import genai


# Load .env
load_dotenv()

# Configure Gemini API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def analyze_pdf_text(text):
    """
    Single API call to Gemini AI:
    - Extract keywords
    - Assess difficulty
    - Generate search queries
    """
    prompt = (
        "You are an AI assistant. For the following PDF text, do all of the following in a single response:\n"
        "1. Extract 10 important keywords.\n"
        "2. Assess the difficulty: Easy, Medium, or Hard.\n"
        "3. Generate 5 concise search queries for research purposes using the keywords and difficulty.\n"
        "Provide output in this format:\n"
        "Keywords: <comma-separated>\n"
        "Difficulty: <Easy/Medium/Hard>\n"
        "Search Queries: <semicolon-separated>\n\n"
        f"PDF Text:\n{text}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()

def parse_analysis(analysis_text):
    """Parse Gemini response into structured data"""
    keywords = ""
    difficulty = ""
    search_queries = ""
    try:
        lines = analysis_text.splitlines()
        for line in lines:
            if line.lower().startswith("keywords"):
                keywords = line.split(":",1)[1].strip()
            elif line.lower().startswith("difficulty"):
                difficulty = line.split(":",1)[1].strip()
            elif line.lower().startswith("search queries"):
                search_queries = line.split(":",1)[1].strip()
    except:
        pass
    return {
        "keywords": keywords,
        "difficulty": difficulty,
        "search_queries": search_queries
    }
