import sqlite3, json
from pathlib import Path

DB_PATH = Path(__file__).parent / "sessions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Table for session context
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            context TEXT
        )
    """)
    # Table for educational resources
    c.execute("""
        CREATE TABLE IF NOT EXISTS resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            url TEXT,
            description TEXT,
            tags TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_context(session_id: str, context: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("REPLACE INTO sessions (session_id, context) VALUES (?, ?)",
              (session_id, json.dumps(context)))
    conn.commit()
    conn.close()

def load_context(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT context FROM sessions WHERE session_id=?", (session_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def add_resource(title: str, url: str, description: str, tags: list):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO resources (title, url, description, tags) VALUES (?, ?, ?, ?)",
              (title, url, description, ",".join(tags)))
    conn.commit()
    conn.close()

def search_resources(keyword: str, limit: int = 5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    like_query = f"%{keyword}%"
    c.execute("""SELECT title, url, description, tags 
                 FROM resources 
                 WHERE title LIKE ? OR description LIKE ? OR tags LIKE ? 
                 LIMIT ?""", (like_query, like_query, like_query, limit))
    rows = c.fetchall()
    conn.close()
    results = []
    for row in rows:
        results.append({
            "title": row[0],
            "url": row[1],
            "description": row[2],
            "tags": row[3].split(",")
        })
    return results

# Initialize DB
init_db()
