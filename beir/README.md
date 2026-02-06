# Assignment 1 — Information Retrieval System

**Course:** CSI4107 – Internet Retrieval  
**Assignment:** 1 – Information Retrieval System (Vector Space Model)  
**Group Members:**  
- Cedric Luiz Dimatulac: 
- Joseph Sreih: 
- Tanner Frisch: 300294742  

**Division of Tasks:**  
- Cedric: Step 1 - Preprocessing
- Tanner: Step 2 - Indexing
- Joseph: Step 3 - Retrieval and Ranking

---

## 1. Program Functionality

This project implements an Information Retrieval (IR) system for the SciFact dataset. The system performs the following tasks:

1. **Preprocessing**  
   - Tokenization of documents  
   - Stopword removal using provided stopword list  
   - Filtering of punctuation, numbers, and markup  

2. **Indexing (Step 2)**  
   - Create an inverted index: each term maps to a dictionary of document IDs and term frequencies  
   - Compute document frequency for each term for later use
   - Track the total number of documents in the corpus  

3. **Retrieval & Ranking (Step 3)**
   - Use inverted index to find a query word
   - Computes cosine similarity between queries and documents  
   - Produces ranked results for test queries  

---

## 2. How to Run

**Requirements:** Python 3.x, SciFact dataset in `datasets/scifact/`  

### Build Inverted Index (Step 2)

```bash
python indexer.py
```

---

## Algorithms, Data Structures, and Results

### Step 1 — Preprocessing
**Algorithm:**  
- Tokenize text into lowercase words using regex `[a-z]+`.  
- Remove stopwords using a provided stopword list.  

**Data structures:**  
- Input: string (document text)  
- Output: list of tokens (Python list)  

**Optimizations:**  
- Stopwords stored as a set for O(1) lookups.

**Results:**
```bash
Document ID: 4983
First 30 tokens:
['microstructural', 'development', 'human', 'newborn', 'cerebral', 'white', 'matter', 'assessed', 'vivo', 'diffusion', 'tensor', 'magnetic', 'resonance', 'imaging', 'alterations', 'architecture', 'cerebral', 'white', 'matter', 'developing', 'human', 'brain', 'affect', 'cortical', 'development', 'result', 'functional', 'disabilities', 'line', 'scan'] 
Total tokens: 168
```

### Step 2 - Indexing
**Algorithm:**  
1. Read each document from `corpus.jsonl`.  
2. Preprocess text with Step 1.  
3. For each token, add to inverted index:  
   - `inverted_index[token][doc_id] = term frequency`  
   - Track document frequency for each token.  

**Data structures:**  
- **Inverted index:** dict of dicts  
```python
inverted_index = {
    "token": {"document_x": 2, "document_y": 1},
}
```

**Results:**

```python
Posting list for token 'brain':
{'4983': 1, '169264': 2, '279052': 1, '654735': 2, '695938': 1, '803312': 5, '1220287': 2, '1292369': 1, '1389264': 3, '1410197': 3, '1583041': 1, '1616661': 1, '1733337': 4, '1771079': 2, '1836154': 1, '1871499': 3, '1889358': 4, '1964163': 2, '2194320': 1, '2225918': 1, '2356950': 1, '2424794': 1, '2436602': 2, '2437807': 2, ...}
```

### Step 3 - Retrieval and Ranking
**Algorithm**  
1. Compute cosine similarity between queries and candidate documents
2. Generate ranked list for each test query
3. Output results in ```Results``` file

**Results**  
- First 10 answers for first 2 queries
- MAP score using ```trec_eval```