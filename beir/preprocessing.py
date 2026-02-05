import re

def load_stopwords(path="stopwords.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

STOPWORDS = load_stopwords()

def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z]+", text)

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def preprocess(text):
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return tokens
