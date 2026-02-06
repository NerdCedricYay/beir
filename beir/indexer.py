import json
from collections import defaultdict, Counter
from preprocessing import preprocess

CORPUS_PATH = "datasets/scifact/corpus.jsonl"

inverted_index = defaultdict(dict)
document_freq = defaultdict(int)
document_count = 0

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        
        doc_id = doc["_id"]
        text = doc["title"] + " " + doc["text"]
        
        tokens = preprocess(text)
        
        token_freq = Counter(tokens)
        
        for term, freq in token_freq.items():
            inverted_index[term][doc_id] = freq
            
        for term in token_freq:
            document_freq[term] += 1
            
        document_count += 1