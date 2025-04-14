A Python-based tool designed to process and cluster HTML files by analyzing their textual content and structural elements. It leverages natural language processing and machine learning techniques to group similar HTML documents, facilitating tasks such as template detection, duplicate identification, and content analysis.​

Features HTML Parsing: Utilizes BeautifulSoup to extract titles, visible text, and tag sequences from HTML files.​

Text Vectorization: Applies TF-IDF to convert textual data into numerical vectors, capturing the importance of words across documents.​

Dimensionality Reduction: Employs Truncated SVD to reduce the dimensionality of TF-IDF vectors, enhancing computational efficiency.​

Structural Similarity: Calculates Jaccard similarity between tag sequences to assess structural resemblance between HTML files.​

Clustering: Implements Agglomerative Clustering to group similar HTML documents based on combined textual and structural similarities.​

Visualization: Generates interactive 2D scatter plots using t-SNE and Plotly to visualize the clustering results.​

Parallel Processing: Incorporates multiprocessing to expedite the processing of large datasets.
Key Functionalities
1. Accurate HTML Content Extraction
The script uses BeautifulSoup with the lxml parser to extract relevant content from each HTML file. It:

Retrieves the document title.

Removes irrelevant tags such as script, style, and noscript.

Captures a meaningful sequence of tags (div, p, img, a, etc.).

Normalizes the extracted text and converts the title to lowercase.

 Clean, robust, and structured data extraction.

2. Parallel Feature Extraction
To boost performance, the script leverages ProcessPoolExecutor for parallel processing of HTML files. It:

Uses a safe multiprocessing pattern (if __name__ == '__main__'), which is essential for compatibility with Windows.

Filters out corrupted or unreadable files (returns None on failure).

 Real speedup for processing dozens or hundreds of files in parallel.

3. Dual Similarity Computation
The system combines textual and structural analysis to assess similarity between files:

Textual similarity is based on TF-IDF, followed by SVD (dimensionality reduction), then cosine similarity.

Structural similarity is computed via the Jaccard index on HTML tag sequences, optimized with joblib.Parallel.

 Combining content and structure leads to more accurate and meaningful clustering.

4. Flexible and Interpretable Clustering
It applies Agglomerative Clustering with a distance threshold, without forcing a fixed number of clusters. You can configure:

alpha – weight of textual similarity.

beta – weight of structural similarity.

distance_threshold – controls granularity of clustering.

 Ideal for exploratory analysis and adaptable to various use cases.

5. Debug-Friendly Output and Visualization
The tool generates outputs in multiple formats for easy inspection:

Saves clusters as Python (.py) and JSON (.json) files.

Exports a CSV (debug_features.csv) with parsed HTML features for human debugging.

Displays an interactive 2D visualization using Plotly + t-SNE, if there are at least 3 samples.

 Ready for both visual exploration and in-depth debugging.

Development Process & Decision Rationale 
Stage 1 — Initialization and Testing (April 13) Commit: Uploading The project and the first results

Goal: Kickstart the project with a functional initial version for processing HTML files.

What the code did: Extracted titles and text, created TF-IDF vectors, computed similarity, and applied clustering.

Limitations:

Sequential processing (slow for large volumes).

Static and basic visualizations.

Lack of detailed logging.

Stage 2 — Interactivity and Visual Output (April 13) Commit: Modifying the plots, adding interactive graphs and some test outputs

Motivation: The need for better visualization and interpretation of the resulting clusters.

Changes made:

Added an interactive plot (likely using Plotly).

Initial testing to display the distribution of HTML files in vector space.

Stage 3 — Optimization and Scaling (April 14) Commit: Making the plots interractive, adding parallel processing for more efficient management nad other things to oprimize and scale the code even further

Motivation: Significantly improve execution speed and clarity of the results.

Improvements:

Added ProcessPoolExecutor to parallelize HTML data extraction.

Integrated interactive and scalable t-SNE visualization using Plotly.

Added more detailed logging + execution time tracking per processing stage.

Combined text similarity (cosine) with structural similarity (Jaccard).

Ensured compatibility and safe execution on Windows (if name == 'main').
