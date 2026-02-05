import json
from preprocessing import preprocess

CORPUS_PATH = "datasets/scifact/corpus.jsonl"

def test_preprocessing(num_docs=2):
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_docs:
                break

            doc = json.loads(line)
            doc_id = doc["_id"]
            text = doc["title"] + " " + doc["text"]

            tokens = preprocess(text)

            print("=" * 60)
            print(f"Document ID: {doc_id}")
            print("First 30 tokens:")
            print(tokens[:30])
            print("Total tokens:", len(tokens))

if __name__ == "__main__":
    test_preprocessing()
