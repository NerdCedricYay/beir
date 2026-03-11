from collections import defaultdict, Counter
import math
from preprocessing import preprocess
from indexer import document_count, document_freq, inverted_index

def compute_idf(term):
    df = document_freq.get(term, 0)
    if df == 0:
        return 0
    return math.log2(document_count / df)

def retrieval(query, top_k=100):
    query_weights = defaultdict() 
    scores = defaultdict()

    tokens = preprocess(query)
    query_tf = Counter(tokens)
    max_tf = max(query_tf.values())
    doc_norms = defaultdict(float)
    query_norm = 0 

    for term, tf in query_tf.items():
        idf = compute_idf(term)
        weight = (tf/max_tf) * idf
        query_weights[term] = weight
        query_norm += weight ** 2

    similarity_measure = defaultdict(float)

    for term, postings in inverted_index.items():
        idf = compute_idf(term)
        for doc_id, tf in postings.items():
            weight = tf * idf
            if term in query_tf:
                similarity_measure[doc_id] += weight * query_weights[term]
            doc_norms[doc_id] += weight ** 2
    
    for doc_id in doc_norms:
        if (doc_norms[doc_id] * query_norm) != 0:
            scores[doc_id] = similarity_measure[doc_id]/math.sqrt((doc_norms[doc_id] * query_norm))
    
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_docs[:top_k]
