import os
import glob
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import numpy as np


def extract_html_features(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        soup = BeautifulSoup(file, 'lxml')

        title = soup.title.string.strip() if soup.title and soup.title.string else ''

        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator=' ', strip=True)

        tags = [tag.name for tag in soup.find_all() if tag.name in ['div', 'h1', 'h2', 'p', 'img', 'a', 'ul', 'li']]

        return {
            "filename": os.path.basename(filepath),
            "title": title,
            "text": text,
            "tag_sequence": tags
        }


def jaccard_similarity(seq1, seq2):
    set1, set2 = set(seq1), set(seq2)
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def process_tier(tier_path_pattern, output_filename):
    files = glob.glob(tier_path_pattern)
    if not files:
        print(f"[!] Niciun fișier găsit în: {tier_path_pattern}")
        return

    data = [extract_html_features(f) for f in files]

    titles = [d['title'] for d in data]
    texts = [d['text'] for d in data]
    structures = [d['tag_sequence'] for d in data]
    filenames = [d['filename'] for d in data]

    combined_texts = [f"{t} {x}" for t, x in zip(titles, texts)]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    text_sim = cosine_similarity(tfidf_matrix)

    n = len(structures)
    struct_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            struct_sim[i][j] = jaccard_similarity(structures[i], structures[j])

    alpha, beta = 0.7, 0.3
    combined_sim = alpha * text_sim + beta * struct_sim
    distance_matrix = 1 - combined_sim

    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=0.3,
        n_clusters=None
    )
    labels = clustering.fit_predict(distance_matrix)

    clusters = defaultdict(list)
    for label, fname in zip(labels, filenames):
        clusters[label].append(fname)

    grouped_files = list(clusters.values())

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("grouped_html = [\n")
        for group in grouped_files:
            f.write("    " + str(group) + ",\n")
        f.write("]\n")

    print(f"[✓] Grupuri salvate în: {output_filename}")


base_path = "C:/Users/Tudor/Downloads/clones 2/clones"
for i in range(1, 5):
    tier_pattern = f"{base_path}/tier{i}/*.html"
    output_file = f"grouped_html_tier{i}.py"
    process_tier(tier_pattern, output_file)
