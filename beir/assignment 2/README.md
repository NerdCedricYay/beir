# Assignment 2 — Neural Information Retrieval System

**Course:** CSI4107 – Internet Retrieval  
**Assignment:** 2 – Neural Information Retrieval System
**Group Members:**  
- Cedric Luiz Dimatulac: 300171173
- Joseph Sreih: 300290385
- Tanner Frisch: 300294742  

**Division of Tasks:**  
- Cedric:
- Tanner: Neural reranking with all-MiniLM-L6-v2 model
- Joseph:

---

## 1. Program Functionality

This project implements a two stage IR system

1. Classic IR:
- Same as in assignment 1
- Uses inverted index and cosine similarity

2. Neural Reranking
- Uses a neural model to rerank the top k candidates
- Tested with 3 models:
    - all-MiniLM-L6-v2
    - Model 2
    - Model 3

### Reranking process
1. Encode query and top k candidate documents using the neural model
2. Normalize embeddings
3. Compute cosine similarity between query and document embeddings
4. Combine assignment 1 and the neural model scores 
5. Sort documents by final scores to produce the new ranked list

---

## 2. How to Run

**Requirements:** Python 3.x, SciFact dataset in `datasets/scifact/`  

### 

```bash
python main.py
```

---

## Algorithms, Data Structures, and Results

### Assignment 1 IR
- Algorithm
    - Tokenize
    - remove stopwords
    - create inverted index
    - compute cosine similarity.
- Data structures
    - dict-of-dict inverted index
    - document frequency map.
- Optimizations
    - stopwords as a set
    - posting lists as Counter for constant lookup time and avoiding duplicates.

### Neural Reranking

### all-MiniLM-L6-v2
- Algorithm: Encode query/documents, compute cosine similarity, combine with normalized classical score.
- Data structures: numpy arrays for embeddings and final scores.
- Optimizations: precompute document embeddings, vectorized cosine similarity with NumPy.

### Model 2

### Model 3

## Results

### Model 1 — all-MiniLM-L6-v2
MAP: 0.6285
P@10: 0.0920
The `Results` file shows that the neural reranker caused a significant improvement in the retrieval performance over the IR system from Assignment 1. For example, the top-ranked document for query 0 in the neural output has a score of `0.1660`, compared to a score of `0.0775` in the cosine-run. Overall, MAP increased from 0.51 to 0.6285, showing that the neural model was better at identifying relevant documents and pushes them higher in the ranking. This demonstrates that combining the classical IR system from Assignment 1 with neural embeddings produces more accurate results than classical IR alone.

### Model 2 — 
MAP: 
P@10:

### Model 3 — 
MAP: 
P@10: 

---

## Notes
- Neural reranking performed better with full text, not preprocessed text
- Alpha can be tuned to change the balance between neural and classical