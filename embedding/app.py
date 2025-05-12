import os
from flask import Flask, jsonify, request
from flask_cors import CORS 

from main import (  
    load_corpus_from_directory,
    get_word_embeddings_and_info,
    reduce_dimensions,
    find_nearest_neighbors,
)

app = Flask(__name__)
CORS(app)

# Globals to store data
vectorizer = None
tfidf_matrix = None
feature_names = None
word_vectors = None
reduced_embeddings_pca = None
reduced_embeddings_tsne = None

BBC_ARTICLES_DIRECTORY = r'C:\Users\poudy\Downloads\llm_1\entertainment'  

def load_data_and_train_models():
    global vectorizer, tfidf_matrix, feature_names, word_vectors, reduced_embeddings_pca, reduced_embeddings_tsne

    print("--- Loading and Training Models for Flask App ---")
    custom_corpus = load_corpus_from_directory(BBC_ARTICLES_DIRECTORY)

    if not custom_corpus:
        print("ERROR: No articles loaded. Cannot start the API without corpus data.")
        return

    print("Generating TF-IDF embeddings...")
    vectorizer, tfidf_matrix, feature_names, word_vectors = get_word_embeddings_and_info(custom_corpus)
    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Word vectors shape: {word_vectors.shape}")

    print("Performing Dimensionality Reduction (PCA)...")
    reduced_embeddings_pca = reduce_dimensions(word_vectors, method='pca')
    print(f"PCA reduced shape: {reduced_embeddings_pca.shape}")

    print("Performing Dimensionality Reduction (t-SNE)...")
    perplexity_val = min(30, len(word_vectors) - 1)
    if perplexity_val <= 0:
        print("Warning: Not enough words for t-SNE.")
        reduced_embeddings_tsne = None
    else:
        reduced_embeddings_tsne = reduce_dimensions(word_vectors, method='tsne', perplexity=perplexity_val)
        print(f"t-SNE reduced shape: {reduced_embeddings_tsne.shape}")

    print("Models Loaded and Ready")

@app.route("/")
def home():
    return "Word Embeddings API is running! Visit /embeddings_data or /nearest_neighbors"

@app.route("/embeddings_data")
def get_embeddings_data():
    if reduced_embeddings_pca is None or reduced_embeddings_tsne is None:
        return jsonify({"error": "Models not loaded. Please check server logs."}), 500
    if feature_names is None:
        return jsonify({"error": "Vocabulary not loaded. Please check server logs."}), 500

    pca_data = [
        {"word": word, "x": float(reduced_embeddings_pca[i, 0]), "y": float(reduced_embeddings_pca[i, 1])}
        for i, word in enumerate(feature_names)
    ]

    tsne_data = [
        {"word": word, "x": float(reduced_embeddings_tsne[i, 0]), "y": float(reduced_embeddings_tsne[i, 1])}
        for i, word in enumerate(feature_names)
    ]

    return jsonify({
        "pca_embeddings": pca_data,
        "tsne_embeddings": tsne_data,
        "vocabulary": feature_names,
    })

@app.route("/nearest_neighbors")
def get_nearest_neighbors():
    word = request.args.get("word")
    k = request.args.get("k", default=5, type=int)

    if not word:
        return jsonify({"error": "Please provide a 'word' parameter."}), 400
    if feature_names is None or word_vectors is None:
        return jsonify({"error": "Models not loaded. Please check server logs."}), 500

    neighbors = find_nearest_neighbors(word, feature_names, word_vectors, k)

    if not neighbors:
        return jsonify({"message": f"'{word}' not found in vocabulary or no neighbors found."})

    formatted_neighbors = [{"word": n_word, "similarity": float(n_sim)} for n_word, n_sim in neighbors]
    return jsonify({
        "query_word": word,
        "neighbors": formatted_neighbors
    })

if __name__ == "__main__":
    load_data_and_train_models()
    app.run(debug=True, host="127.0.0.1", port=5000)
