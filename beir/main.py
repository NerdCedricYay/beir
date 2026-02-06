import json
from preprocessing import preprocess
from indexer import inverted_index

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
            
def test_indexing():
    print("="*60)
    print("Vocabulary size:", len(inverted_index))

    sample_tokens = list(inverted_index.keys())[:100]
    print("\nSample 100 tokens:")
    print(sample_tokens)

    for test_token in ["claim", "human", "brain"]:
        if test_token in inverted_index:
            print(f"\nPosting list for token '{test_token}':")
            print(inverted_index[test_token])
            print(f"Appears in {len(inverted_index[test_token])} documents")
        else:
            print(f"Token '{test_token}' not found in index")

if __name__ == "__main__":
    test_preprocessing()
    test_indexing()
