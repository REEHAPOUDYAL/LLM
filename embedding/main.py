import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

#Data Loading
def load_corpus_from_directory(directory_path):
    corpus = []
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return []

    print(f"Loading articles from: {directory_path}")
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if content.strip(): 
                            corpus.append(content)
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")
    print(f"Loaded {len(corpus)} articles.")
    return corpus

def get_word_embeddings_and_info(corpus):
    
    #TF-IDF embeddings
    
    vectorizer = TfidfVectorizer(max_features=5000,  
                                   stop_words='english',
                                   min_df=5)  
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    word_vectors = tfidf_matrix.T.toarray() 
    return vectorizer, tfidf_matrix, feature_names, word_vectors

#Reducing dimensionality
def reduce_dimensions(embeddings, method='tsne', n_components=2, random_state=42, perplexity=30):
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif method == 'tsne':
        valid_perplexity = min(perplexity, len(embeddings) - 1)
        if valid_perplexity <= 0:
            print(f"Warning...Skipping t-SNE.")
            return None
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=valid_perplexity)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings

#Visualization
def visualize_embeddings(reduced_embeddings, words, title="Word Embeddings Visualization", method=""):
    if reduced_embeddings is None:
        print(f"Cannot visualize...as embeddings are None.")
        return

    plt.figure(figsize=(12, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.7, s=10)

    num_words_to_annotate = 50
    indices_to_annotate = np.random.choice(len(words), min(num_words_to_annotate, len(words)), replace=False)

    for i in indices_to_annotate:
        plt.annotate(words[i], (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                     xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=8)

    plt.title(f"{title} ({method.upper()})")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.grid(True)
    plt.show()
    
def visualize_embeddings_interactive(reduced_embeddings, words, title="Interactive Word Embeddings Visualization", method=""):
    if reduced_embeddings is None:
        print(f"Cannot visualize interactively...as embeddings are None.")
        return

    df = pd.DataFrame({
        'word': words,
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1]
    })
    fig = px.scatter(df, x='x', y='y', text='word',
                     title=f"{title} ({method.upper()})",
                     hover_data={'word': True, 'x': False, 'y': False},
                     height=700)
    fig.update_traces(textposition='top center', marker=dict(size=5))
    fig.show()
#Nearest Neighbour
def find_nearest_neighbors(word, feature_names, word_vectors, k=5):
    target_word_lower = word.lower()
    feature_name_list = [w.lower() for w in feature_names]
    if target_word_lower not in feature_name_list:
        return []
    word_index = feature_name_list.index(target_word_lower)

    word_index = feature_names.index(target_word_lower)
    target_word_embedding = word_vectors[word_index]
    similarities = cosine_similarity(target_word_embedding.reshape(1, -1), word_vectors)[0]

    word_similarities = []
    for i, sim in enumerate(similarities):
        if i == word_index:
            continue
        word_similarities.append((feature_names[i], sim))

    sorted_neighbors = sorted(word_similarities, key=lambda item: item[1], reverse=True)
    return sorted_neighbors[:k]

if __name__ == "__main__":
    bbc_articles_directory = r'C:\Users\poudy\Downloads\llm_1\entertainment'  
    custom_corpus = load_corpus_from_directory(bbc_articles_directory)

    if not custom_corpus:
     raise RuntimeError("No articles found in the directory. Exiting...")


    print("Generating TF-IDF embeddings...")
    vectorizer, tfidf_matrix, feature_names, word_vectors = get_word_embeddings_and_info(custom_corpus)
    print(f"Vocabulary size: {len(feature_names)}")
    print(f"TF-IDF matrix shape (documents x words): {tfidf_matrix.shape}")
    print(f"Word vectors shape (words x documents): {word_vectors.shape}")
    print("\nPerforming Dimensionality Reduction")
    print("Reducing dimensions with PCA")
    reduced_embeddings_pca = reduce_dimensions(word_vectors, method='pca')
    print(f"PCA reduced shape: {reduced_embeddings_pca.shape if reduced_embeddings_pca is not None else 'N/A'}")
    visualize_embeddings(reduced_embeddings_pca, feature_names, title="TF-IDF Word Embeddings (BBC Corpus)", method="PCA")
    visualize_embeddings_interactive(reduced_embeddings_pca, feature_names, title="Interactive TF-IDF Word Embeddings (BBC Corpus)", method="PCA")
    print("Reducing dimensions with t-SNE...")
    perplexity_val = min(30, len(word_vectors) - 1)
    if perplexity_val <= 0:
        print("Not enough words for t-SNE...Skipping...")
        reduced_embeddings_tsne = None
    else:
        reduced_embeddings_tsne = reduce_dimensions(word_vectors, method='tsne', perplexity=perplexity_val)
    print(f"t-SNE reduced shape: {reduced_embeddings_tsne.shape if reduced_embeddings_tsne is not None else 'N/A'}")
    visualize_embeddings(reduced_embeddings_tsne, feature_names, title="TF-IDF Word Embeddings (BBC Corpus)", method="t-SNE")
    visualize_embeddings_interactive(reduced_embeddings_tsne, feature_names, title="Interactive TF-IDF Word Embeddings (BBC Corpus)", method="t-SNE")

    print("\nTesting Nearest Neighbors:")
    example_words = ["economy", "politics", "sport", "technology", "business", "government"]
    for word_to_test in example_words:
        print(f"Nearest neighbors for '{word_to_test}':")
        neighbors = find_nearest_neighbors(word_to_test, feature_names, word_vectors, k=5)
        if not neighbors:
            print(f"'{word_to_test}' not found in vocabulary.")
        else:
            for word, sim in neighbors:
                print(f"- {word}: {sim:.4f}")
        print("-" * 20)