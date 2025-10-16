import os
from dotenv import load_dotenv
from google import genai


# Load .env
load_dotenv()

# Configure Gemini API
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CHUNK_SIZE = 6000  # characters per chunk

def analyze_pdf_text(text):
    """
    Summarize large PDFs using Gemini 2.5 Flash.
    - Splits text into chunks
    - Summarizes each chunk
    - Combines summaries into one final summary
    - Extracts keywords, difficulty, search queries from first chunk
    """
    text_chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    final_summary = ""

    # First chunk: also extract keywords/difficulty/search queries
    first_prompt = (
        "You are an AI assistant. For the following PDF text, do all of the following:\n"
        "1. Extract 10 important keywords.\n"
        "2. Assess difficulty: Easy, Medium, or Hard.\n"
        "3. Generate 5 concise research search queries.\n"
        "4. Write a short summary of this text  (200–300 words).\n"
        "Output format:\n"
        "Keywords: <comma-separated>\n"
        "Difficulty: <Easy/Medium/Hard>\n"
        "Search Queries: <semicolon-separated>\n"
        "Summary: <summary text>\n\n"
        f"PDF Text:\n{text_chunks[0]}"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=first_prompt
    )
    analysis_text = response.text.strip()
    parsed = parse_analysis(analysis_text)
    final_summary += parsed["summary"]

    # Summarize remaining chunks
    for chunk in text_chunks[1:]:
        chunk_prompt = (
            "You are an AI assistant. Summarize the following PDF text chunk concisely "
            "(keep it short, 150-250 words), maintaining context from previous text:\n\n"
            f"{chunk}"
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=chunk_prompt
        )
        final_summary += "\n\n" + resp.text.strip()

    # Replace summary field with combined summary
    parsed["summary"] = final_summary.strip()

    ## Add word count + reading time
    word_count = len(text.split())
    reading_time = round(word_count / 200)  # ~200 words per minute

    parsed["summary"] = final_summary.strip()
    parsed["word_count"] = word_count
    parsed["reading_time"] = reading_time
    #---------------------------------------
    return parsed

def parse_analysis(analysis_text):
    """Parse Gemini response into structured data"""
    keywords = ""
    difficulty = ""
    search_queries = ""
    summary = ""
    try:
        lines = analysis_text.splitlines()
        for line in lines:
            if line.lower().startswith("keywords"):
                keywords = line.split(":", 1)[1].strip()
            elif line.lower().startswith("difficulty"):
                difficulty = line.split(":", 1)[1].strip()
            elif line.lower().startswith("search queries"):
                search_queries = line.split(":", 1)[1].strip()
            elif line.lower().startswith("summary"):
                summary = line.split(":", 1)[1].strip()
    except:
        pass
    return {
        "keywords": keywords,
        "difficulty": difficulty,
        "search_queries": search_queries,
        "summary": summary
    }





