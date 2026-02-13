import json
from preprocessing import preprocess
from indexer import inverted_index
from retrieve import retrieval

CORPUS_PATH = "datasets/scifact/corpus.jsonl"
QUERIES_PATH = "datasets/scifact/queries.jsonl"

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


def test_retrieval(query_ids=["0", "1"], top_k=10):
    print("=" * 60)
    print("Testing Retrieval")


    queries = {}
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    for qid in query_ids:

        query_text = queries[qid]
        print("\n" + "-" * 60)
        print(f"Query ID: {qid}")
        print(f"Query Text: {query_text}")

        results = retrieval(query_text, top_k=top_k)

        print("\nTop Results:")
        for rank, (doc_id, score) in enumerate(results, start=1):
            print(f"{rank}. DocID: {doc_id} | Score: {score:.4f}")

# This test was used to create the Results file.
def run_all_queries(output_file="Results", top_k=100, run_tag="cosine_run"):
    
    queries = []

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries.append((q["_id"], q["text"]))

    queries.sort(key=lambda x: int(x[0]))

    with open(output_file, "w", encoding="utf-8") as out:

        for qid, query_text in queries:
            results = retrieval(query_text, top_k=top_k)

            for rank, (doc_id, score) in enumerate(results, start=1):

                out.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

    print(f"Results written to {output_file}")

if __name__ == "__main__":
    test_preprocessing()
    test_indexing()
    test_retrieval()