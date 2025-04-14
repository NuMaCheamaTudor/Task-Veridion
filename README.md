A Python-based tool designed to process and cluster HTML files by analyzing their textual content and structural elements. It leverages natural language processing and machine learning techniques to group similar HTML documents, facilitating tasks such as template detection, duplicate identification, and content analysis.​

Features HTML Parsing: Utilizes BeautifulSoup to extract titles, visible text, and tag sequences from HTML files.​

Text Vectorization: Applies TF-IDF to convert textual data into numerical vectors, capturing the importance of words across documents.​

Dimensionality Reduction: Employs Truncated SVD to reduce the dimensionality of TF-IDF vectors, enhancing computational efficiency.​

Structural Similarity: Calculates Jaccard similarity between tag sequences to assess structural resemblance between HTML files.​

Clustering: Implements Agglomerative Clustering to group similar HTML documents based on combined textual and structural similarities.​

Visualization: Generates interactive 2D scatter plots using t-SNE and Plotly to visualize the clustering results.​

Parallel Processing: Incorporates multiprocessing to expedite the processing of large datasets.

Development Process & Decision Rationale Stage 1 — Initialization and Testing (April 13) Commit: Uploading The project and the first results

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
