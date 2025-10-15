import os
from typing import List
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_chunk(chunk: str) -> str:
    prompt = f"Summarize the following text in 3-6 bullet points and a one-sentence summary:\n\n{chunk}"
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1
    )
    return resp.choices[0].message.content.strip()
def summarize_document(text: str) -> str:
    from pdf_utils import split_into_chunks, clean_text
    clean = clean_text(text)
    chunks = split_into_chunks(clean, max_tokens=1000, overlap=200)
    chunk_summaries = [summarize_chunk(c) for c in chunks]

    # Reduce step: summarize the summaries
    combined = "\n\n".join(chunk_summaries)
    prompt = f"Combine and condense these chunk summaries into a concise summary with key points and one-paragraph summary:\n\n{combined}"
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        max_tokens=500,
        temperature=0.1
    )
    return resp.choices[0].message.content.strip()
