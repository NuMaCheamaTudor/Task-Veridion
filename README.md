## Overview
This project addresses the problem of detecting and grouping HTML documents that are visually or structurally similar from a user's perspective in a web browser. The goal was to find a scalable and flexible way to compare and cluster such documents without relying on hardcoded heuristics.

The approach is based on two complementary dimensions:

Textual content, captured through TF-IDF and semantic similarity.

Structural layout, modeled using HTML tag sequences and Jaccard similarity.

Instead of guessing a fixed number of clusters, the algorithm uses a configurable distance threshold to group documents based on their actual closeness in meaning and structure. The solution was developed iteratively, starting from a simple one and evolving into a parallelized, complex system capable of handling the increasing complexity of all dataset tiers.

Each design choice (TF-IDF, Jaccard, SVD, AgglomerativeClustering) was made based on interpretability, performance, and scalability, ensuring the system remains robust even when datasets grow or change structure.

## Features

### 1. HTML Parsing
- Uses **BeautifulSoup** with the `lxml` parser to extract:
  - Page titles
  - Clean visible text (excluding script, style, noscript)
  - Sequences of structural tags (`div`, `p`, `img`, `a`, etc.)
- Ensures normalization through lowercasing and whitespace cleanup.

### 2. Text Vectorization
- Applies **TF-IDF** to convert content into weighted numerical vectors.
- Captures contextual relevance of words across the dataset.

### 3. Dimensionality Reduction
- Uses **Truncated SVD** to reduce TF-IDF vectors.
- Enhances efficiency without compromising clustering accuracy.

### 4. Structural Similarity
- Computes **Jaccard similarity** between HTML tag sequences.
- Reflects page layout and DOM resemblance.

### 5. Combined Clustering
- Combines both text and structure similarity using a weighted model:
  - `alpha`: weight of textual similarity
  - `beta`: weight of structural similarity
- Uses **Agglomerative Clustering** with configurable `distance_threshold`.

### 6. Visualization and Debug
- Produces interactive **2D t-SNE plots** using Plotly (if enough samples exist).
- Outputs groupings as:
  - `grouped_<tier>.py`
  - `grouped_<tier>.json`
- Generates `debug_features.csv` for transparent inspection.

### 7. Performance Optimization
- Uses **ProcessPoolExecutor** and **joblib.Parallel** for parallelism.
- Compatible with Windows through `if __name__ == '__main__'` guard.

---

## Development Process and Decision Rationale

### Stage 1 — Initialization and Testing (April 13)
**Commit**: Uploading the project and the first results

- Extracted titles, text, TF-IDF vectors
- Calculated cosine similarity
- Ran basic clustering

**Limitations:**
- Sequential processing (slow)
- Basic visualizations
- Minimal logging

### Stage 2 — Interactivity and Visual Output (April 13)
**Commit**: Modifying the plots, adding interactive graphs and some test outputs

- Introduced Plotly-based interactive plots
- Validated HTML distribution in vector space visually

### Stage 3 — Optimization and Scaling (April 14)
**Commit**: Making the plots interactive, adding parallel processing for more efficient management and other things to optimize and scale the code even further

- Parallelized HTML parsing (via `ProcessPoolExecutor`)
- Added detailed logging and runtime tracking
- Introduced Jaccard similarity for structural comparison
- Fine-tuned `alpha`, `beta`, and `distance_threshold`
- Ensured stable execution across all dataset tiers

---


This solution is designed to be robust, scalable, and explainable. 
