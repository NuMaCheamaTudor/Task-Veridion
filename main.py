import os
import glob
import json
import time
import logging
from collections import defaultdict
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import plotly.express as px
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# Example initialization for debugging purposes
data = [{"filename": "example.html", "title": "Example", "text": "Example text", "tag_sequence": ["div", "p"]}]
df = pd.DataFrame(data)
df.to_csv("debug_features.csv", index=False)

logging.basicConfig(level=logging.INFO)

def extract_html_features(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            soup = BeautifulSoup(file, 'lxml')
            title = soup.title.string.strip() if soup.title and soup.title.string else ''
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()
            text = soup.get_text(separator=' ', strip=True).lower()
            tags = [tag.name for tag in soup.find_all() if tag.name in ['div', 'h1', 'h2', 'p', 'img', 'a', 'ul', 'li']]
            return {
                "filename": os.path.basename(filepath),
                "title": title.lower(),
                "text": text,
                "tag_sequence": tags
            }
    except Exception as e:
        logging.warning(f"Failed to parse {filepath}: {e}")
        return None

def parallel_extract_html_features(filepaths):
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(extract_html_features, filepaths))
    return [r for r in results if r is not None]

def jaccard_similarity(seq1, seq2):
    set1, set2 = set(seq1), set(seq2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

def visualize_clusters(distance_matrix, labels, filenames=None):
    n_samples = distance_matrix.shape[0]
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, metric="precomputed", init='random', random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(distance_matrix)
    df_data = {
        'x': coords[:, 0],
        'y': coords[:, 1],
        'cluster': labels,
        'filename': filenames if filenames else [f"file_{i}" for i in range(len(labels))]
    }
    fig = px.scatter(
        df_data, x='x', y='y', color=df_data['cluster'].astype(str),
        hover_data=['filename'], title="Cluster Visualization"
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(legend_title_text='Cluster', height=600)
    fig.show()

def process_folder(input_pattern, output_filename, alpha=0.7, beta=0.3, distance_threshold=0.3):
    files = glob.glob(input_pattern)
    if not files:
        logging.warning("[!] No files found in: %s", input_pattern)
        return

    logging.info("✓ [%s] %d files found. Starting processing...", output_filename, len(files))

    t0 = time.time()
    data = parallel_extract_html_features(files)
    logging.info("✓ HTML parsed in %.2f seconds", time.time() - t0)

    titles = [d['title'] for d in data]
    texts = [d['text'] for d in data]
    structures = [d['tag_sequence'] for d in data]
    filenames = [d['filename'] for d in data]

    structure_strings = [' '.join(tags) for tags in structures]
    combined_texts = [f"{t} {x} {s}" for t, x, s in zip(titles, texts, structure_strings)]

    t1 = time.time()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    svd = TruncatedSVD(n_components=100, random_state=42)
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    text_sim = cosine_similarity(reduced_matrix)
    logging.info("✓ TF-IDF + SVD + cosine similarity in %.2f seconds", time.time() - t1)

    t2 = time.time()
    n = len(structures)
    struct_sim = np.zeros((n, n))
    for i in range(n):
        struct_sim[i][i] = 1.0
        for j in range(i + 1, n):
            sim = jaccard_similarity(structures[i], structures[j])
            struct_sim[i][j] = sim
            struct_sim[j][i] = sim
    logging.info("✓ Structural similarity computed in %.2f seconds", time.time() - t2)

    combined_sim = alpha * text_sim + beta * struct_sim
    distance_matrix = 1 - combined_sim
    distance_matrix = np.clip(distance_matrix, 0, 1)

    t3 = time.time()
    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=distance_threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(distance_matrix)
    logging.info("✓ Clustering done in %.2f seconds", time.time() - t3)

    clusters = defaultdict(list)
    for label, fname in zip(labels, filenames):
        clusters[label].append(fname)

    grouped_files = list(clusters.values())
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("grouped_html = [\n")
        for group in grouped_files:
            f.write("    " + str(group) + ",\n")
        f.write("]\n")
    json_output = output_filename.replace(".py", ".json")
    with open(json_output, "w", encoding="utf-8") as jf:
        json.dump(grouped_files, jf, indent=2, ensure_ascii=False)
    logging.info("[✓] Results saved to: %s and %s", output_filename, json_output)

    visualize_clusters(distance_matrix, labels, filenames)


if __name__ == '__main__':
    base_path = "C:/Users/Tudor/Downloads/clones 2/clones"
    for subfolder in os.listdir(base_path):
        folder_path = os.path.join(base_path, subfolder)
        if os.path.isdir(folder_path):
            pattern = os.path.join(folder_path, "*.html")
            output_file = f"grouped_{subfolder}.py"
            process_folder(pattern, output_file)
            logging.info("[✓] Finished folder: %s", folder_path)