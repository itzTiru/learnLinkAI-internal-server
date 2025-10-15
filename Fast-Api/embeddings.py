# import threading
# from sentence_transformers import SentenceTransformer

# _model = None
# _lock = threading.Lock()

# def get_embedding_model():
#     global _model
#     if _model is None:
#         with _lock:
#             if _model is None:
#                 _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     return _model

# def embed_text(text: str):
#     model = get_embedding_model()
#     return model.encode([text])[0]



#---------------------------------------------------------------

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts: List[str]) -> np.ndarray:
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def search_index(index, query_embedding, k=5):
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k)
    return D, I
