import json
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from retrieve import retrieval

CORPUS_PATH = "dataset/scifact/corpus.jsonl"
DOC2VEC_MODEL_PATH = "doc2vec_model"
VECTOR_SIZE = 1024
EPOCHS = 70

documents, doc_ids = [], []

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        text = doc["title"] + " " + doc["text"]
        documents.append(text)
        doc_ids.append(doc["_id"])

doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
tagged_docs = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(documents)]

# Train or load Doc2Vec depending on if they have been already created
try:
    print("Loading existing Doc2Vec model...")
    model = Doc2Vec.load(DOC2VEC_MODEL_PATH)
except FileNotFoundError:
    print("Training new Doc2Vec model...")
    model = Doc2Vec(
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=4,
        epochs=EPOCHS,
        dm=1
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(DOC2VEC_MODEL_PATH)


# Precompute document vectors
doc_vectors = np.array([model.dv[str(i)] for i in range(len(documents))])
doc_vectors /= np.linalg.norm(doc_vectors, axis=1, keepdims=True)
print(f"{len(doc_vectors)} document embeddings ready!")


def rerank(query_text, top_k=100, alpha=0.5):
    # Step 1: first-stage retrieval
    top_docs = retrieval(query_text, top_k=top_k)
    if len(top_docs) == 0:
        return []

    # Step 2: compute query embedding
    query_vec = model.infer_vector(query_text.split())
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

print(rerank(query1)[:10])
print(rerank(query3)[:10])
