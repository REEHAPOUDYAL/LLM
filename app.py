# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag  # Import pos_tag from NLTK
import os  # Import os for path manipulation

# --- NLTK Data Path Setup ---
# Ensure NLTK data is looked for in a predictable local directory.
nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# --- NLTK and spaCy Setup ---
app = Flask(__name__)
CORS(app)  # Enable CORS for local testing with frontend

try:
    # Ensure NLTK data is downloaded.
    print("Checking NLTK data presence...")
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')  # Check for the general tagger
    # Explicitly check for the _eng version if the general one fails
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("averaged_perceptron_tagger_eng not found. Will attempt to download.")
    nltk.data.find('corpora/omw-1.4')  # Needed for WordNet
    print("NLTK data found.")
except LookupError:
    print("NLTK data missing. Attempting download...")
    # Use the download_dir to ensure they go into our controlled path
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path) # Download the general tagger
    # Explicitly download the _eng version as well, if needed
    try:
        nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)
    except Exception as e:
        print(f"Could not download averaged_perceptron_tagger_eng: {e}. This might be expected if it's bundled.")
    nltk.download('omw-1.4', download_dir=nltk_data_path)
    print("NLTK data download complete.")

# Verify that averaged_perceptron_tagger is actually in the specified path
tagger_path = os.path.join(nltk_data_path, 'taggers', 'averaged_perceptron_tagger')
tagger_eng_path = os.path.join(nltk_data_path, 'taggers', 'averaged_perceptron_tagger_eng')

if not os.path.exists(tagger_path) and not os.path.exists(tagger_eng_path):
    print(f"WARNING: Neither averaged_perceptron_tagger nor averaged_perceptron_tagger_eng found at expected paths within {nltk_data_path} even after download attempt.")
    print("This might indicate a deeper NLTK installation/permission issue.")
elif os.path.exists(tagger_path) and not os.path.exists(tagger_eng_path):
    print(f"Note: averaged_perceptron_tagger found at {tagger_path}, but averaged_perceptron_tagger_eng is not. This is common.")


# Download spaCy model (check if it exists, download if not)
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' is loaded.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Downloading...")
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model downloaded and loaded.")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    nlp = None  # set nlp to None to prevent errors later.


# --- Initialize Resources ---
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    """
    Convert Treebank POS tags to WordNet POS tags for accurate lemmatization.
    """
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun if POS tag is unknown


def process_text(text, task):
    """Processes the input text based on the specified task."""
    print(f"Processing task: {task} | Text: {text}")
    try:
        if task == "tokenize":
            return word_tokenize(text)
        elif task == "stem":
            return [ps.stem(token) for token in word_tokenize(text)]
        elif task == "lemmatize":
            tokens = word_tokenize(text)
            # Get POS tags using NLTK for lemmatization context
            try:
                # The key is that NLTK's pos_tag *itself* needs to locate the tagger.
                # Our nltk.data.path.append(nltk_data_path) should make it findable.
                pos_tags = pos_tag(tokens)
            except LookupError as e:
                print(f"LookupError during pos_tag in lemmatize: {e}")
                print("This error indicates the tagger cannot be found by pos_tag even after attempts.")
                print("Please verify 'averaged_perceptron_tagger' or 'averaged_perceptron_tagger_eng' is in your nltk_data/taggers folder.")
                # Re-raise or return an informative error if it fails again
                raise RuntimeError(f"Failed to perform POS tagging required for lemmatization: {e}")
            lemmas = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tags]
            return lemmas
        elif task == "pos":
            if nlp:  # Use spaCy for POS tagging for the /pos endpoint
                doc = nlp(text)
                return [(token.text, token.tag_) for token in doc]
            else:
                # Fallback to NLTK's pos_tag if spaCy not loaded, but recommend spaCy for /pos
                try:
                    return pos_tag(word_tokenize(text))
                except LookupError as e:
                    return [f"Error: NLTK POS tagger not found for fallback: {e}"]
        elif task == "ner":
            if nlp:  # Check if spacy was loaded correctly
                doc = nlp(text)
                return [(ent.text, ent.label_) for ent in doc.ents]
            else:
                return ["spaCy model was not loaded. NER task cannot be performed"]
        else:
            return ["Invalid task."]
    except Exception as e:
        return [f"Error during processing: {str(e)}"]


@app.route('/<task>', methods=['POST'])
def handle_task(task):
    """Handles the API endpoint for text processing tasks."""
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        result = process_text(text, task)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)