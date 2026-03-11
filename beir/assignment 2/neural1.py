# neural_rerank_mini.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from retrieve import retrieval

model = SentenceTransformer("all-MiniLM-L6-v2")

CORPUS_PATH = "dataset/scifact/corpus.jsonl"
documents, doc_ids = [], []

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        documents.append(doc["title"] + " " + doc["text"])
        doc_ids.append(doc["_id"])

doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

doc_vectors = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
doc_vectors /= np.linalg.norm(doc_vectors, axis=1, keepdims=True)
print(f"{len(doc_vectors)} document embeddings ready!")

def rerank(query_text, top_k=100, alpha=0.5):
    # Step 1: first-stage retrieval
    top_docs = retrieval(query_text, top_k=top_k)
    if len(top_docs) == 0:
        return []

    # Step 2: compute query embedding
    query_vec = model.encode(query_text, convert_to_numpy=True)
    query_vec /= np.linalg.norm(query_vec)

    # Step 3: vectorized cosine similarity
    top_idxs = [doc_id_to_idx[doc_id] for doc_id, _ in top_docs]
    top_vecs = doc_vectors[top_idxs]
    neural_scores = np.dot(top_vecs, query_vec)

    # Step 4: normalize classical scores
    classical_scores = np.array([score for _, score in top_docs])

    # Step 5: combine
    final_scores = alpha * classical_scores + (1 - alpha) * neural_scores

    # Step 6: sort descending
    ranked = [(top_docs[i][0], final_scores[i]) for i in range(len(top_docs))]
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked