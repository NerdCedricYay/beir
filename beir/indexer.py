import json
from preprocessing import preprocess

CORPUS_PATH = "datasets/scifact/corpus.jsonl"

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        doc_id = doc["_id"]
        text = doc["title"] + " " + doc["text"]
        tokens = preprocess(text)
