# TBI_TP2 - Search Engine from Scratch
Collaborators: Alfan Farizki Wicaksono, Claude AI

A search engine built from scratch using Python standard libraries,
developed as part of the Information Retrieval (TBI) course assignment.

## Project Overview
This project implements a search engine using the BSBI (Blocked Sort-Based Indexing)
scheme with an inverted index. It supports multiple retrieval and compression methods,
along with evaluation metrics to measure search effectiveness.

## Features

### Indexing Modes
- **BSBI (Blocked Sort-Based Indexing)** — collects all term-doc pairs per block,
  sorts them globally by termID, then writes to disk
- **SPIMI (Single-Pass In-Memory Indexing)** — builds a hashtable dictionary directly
  from the token stream, sorting only within each block before writing to disk.
  More memory efficient than BSBI as it avoids global sorting of term-doc pairs

### Index Compression
- **Standard Postings** — raw integer encoding using Python's array library
- **Variable Byte Encoding (VBE)** — gap-based compression using variable-length bytes
- **Elias-Gamma Encoding** — bit-level gap-based compression, efficient for small numbers

### Retrieval Methods
- **TF-IDF** — term frequency-inverse document frequency scoring
- **BM25** — probabilistic scoring with TF saturation and document length normalization
- **WAND (Weak or Weighted AND)** — Top-K retrieval optimization using BM25 scoring with upper bound
pruning to skip documents that cannot make it into top-K results

### Evaluation Metrics
- **RBP** (Rank-Biased Precision) — models user behavior with a persistence probability
- **DCG** (Discounted Cumulative Gain) — rewards relevant documents found at higher ranks
- **NDCG** (Normalized DCG) — DCG normalized by ideal ranking score, range [0, 1]
- **AP** (Average Precision) — average of precision values at each relevant document rank

## Requirements
Install the required dependency before running:
```bash
pip install tqdm
```

## How to Run

**1. Build the index (choose one)**

Using BSBI:
```bash
python bsbi.py
```

Using SPIMI:
```bash
python spimi.py
```

**2. Test retrieval with sample queries**
```bash
python search.py
```

**3. Evaluate search effectiveness**
```bash
python evaluation.py
```

## Project Structure
```
TBI_TP2/
├── collection/       # Document collection to be indexed
├── index/            # Generated inverted index files (auto-generated)
├── tmp/              # Temporary index files during indexing (auto-generated)
├── bsbi.py           # BSBI indexing and retrieval methods
├── compression.py    # Postings list compression algorithms
├── evaluation.py     # Evaluation metrics
├── index.py          # Inverted index read/write logic
├── search.py         # Sample retrieval queries
├── util.py           # Utility classes and functions
├── qrels.txt         # Query relevance judgments
└── queries.txt       # 30 evaluation queries
```