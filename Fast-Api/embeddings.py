import threading
from sentence_transformers import SentenceTransformer

_model = None
_lock = threading.Lock()

def get_embedding_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model

def embed_text(text: str):
    model = get_embedding_model()
    return model.encode([text])[0]