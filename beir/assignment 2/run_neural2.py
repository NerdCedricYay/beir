import json
from tqdm import tqdm
from neural2 import rerank  
import csv
from retrieval.evaluation import EvaluateRetrieval


# Load queries
queries = []
with open("dataset/scifact/queries.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        queries.append((q["_id"], q["text"]))
queries.sort(key=lambda x: int(x[0]))

# Load qrels
qrels = {}
with open("dataset/scifact/qrels/test.tsv", "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)  # skip header
    for row in reader:
        qid, doc_id, rel = row
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = int(rel)

# Run reranker and save results
results = {}
with open("Results_doc2vec", "w", encoding="utf-8") as out:
    for qid, query_text in tqdm(queries, desc="Processing queries"):
        ranked = rerank(query_text, top_k=100, alpha=0.5)
        results[qid] = {}
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} doc2vec\n")
            results[qid][doc_id] = float(score)

# Evaluate MAP@100
_, map_scores, _, _ = EvaluateRetrieval.evaluate(qrels, results, k_values=[100])

# Evaluate P@10
_, _, _, precision_10 = EvaluateRetrieval.evaluate(qrels, results, k_values=[10])

# Print results
print(f"MAP: {map_scores['MAP@100']:.4f}")
print(f"P@10: {precision_10['P@10']:.4f}")
