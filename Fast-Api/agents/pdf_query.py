import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CHUNK_SIZE = 6000  # characters per chunk

def clean_markdown_bold(text):
    """Remove Markdown bold markers (**text**) from text"""
    return re.sub(r'\*\*(.*?)\*\*', r'\1', text)

def analyze_pdf_text(text):
    """
    Summarize large PDFs using Gemini 2.5 Flash (via google-generativeai).
    - Splits text into chunks
    - Summarizes each chunk
    - Extracts keywords, difficulty, and search queries from first chunk
    """
    text_chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    final_summary = ""

    model = genai.GenerativeModel("gemini-2.5-flash")

    # First chunk prompt: extract structured info + summary
    first_prompt = f"""
    You are an AI assistant. For the following PDF text, do all of the following:
    1. Extract 10 important keywords.
    2. Assess difficulty: Easy, Medium, or Hard.
    3. Generate 5 concise research search queries.
    4. Write a short summary of this text (200–300 words).

    Output format:
    Keywords: <comma-separated>
    Difficulty: <Easy/Medium/Hard>
    Search Queries: <semicolon-separated>
    Summary: <summary text>

    PDF Text:
    {text_chunks[0]}
    """

    response = model.generate_content(first_prompt)
    analysis_text = clean_markdown_bold(response.text.strip())
    parsed = parse_analysis(analysis_text)
    final_summary += parsed["summary"]

    # Summarize remaining chunks (if any)
    for chunk in text_chunks[1:]:
        chunk_prompt = f"""
        You are an AI assistant. Summarize the following PDF PDF text chunk concisely
        (150–250 words), maintaining context from the previous part of the text:
        {chunk}
        """
        resp = model.generate_content(chunk_prompt)
        final_summary += "\n\n" + clean_markdown_bold(resp.text.strip())

    # Compute word count and reading time
    word_count = len(text.split())
    reading_time = round(word_count / 200)  # ~200 words/min

    parsed["summary"] = final_summary.strip()
    parsed["word_count"] = word_count
    parsed["reading_time"] = reading_time

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
                keywords = clean_markdown_bold(line.split(":", 1)[1].strip())
            elif line.lower().startswith("difficulty"):
                difficulty = clean_markdown_bold(line.split(":", 1)[1].strip())
            elif line.lower().startswith("search queries"):
                search_queries = clean_markdown_bold(line.split(":", 1)[1].strip())
            elif line.lower().startswith("summary"):
                summary = clean_markdown_bold(line.split(":", 1)[1].strip())
    except Exception as e:
        print("Error parsing Gemini output:", e)

    return {
        "keywords": keywords,
        "difficulty": difficulty,
        "search_queries": search_queries,
        "summary": summary
    }