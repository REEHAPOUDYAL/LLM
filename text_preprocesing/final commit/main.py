import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag  
import os  
import nltk

nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

try:
    print("Checking NLTK data presence for main.py...")
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        print("averaged_perceptron_tagger_eng not found. Will attempt to download.")
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    print("NLTK data found for main.py.")
except LookupError:
    print("NLTK data missing for main.py. Attempting download...")
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
    try:
        nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)
    except Exception as e:
        print(f"Could not download averaged_perceptron_tagger_eng: {e}. This might be expected if it's bundled.")
    nltk.download('wordnet', download_dir=nltk_data_path)
    nltk.download('omw-1.4', download_dir=nltk_data_path)
    print("NLTK data download complete for main.py.")

tagger_path = os.path.join(nltk_data_path, 'taggers', 'averaged_perceptron_tagger')
tagger_eng_path = os.path.join(nltk_data_path, 'taggers', 'averaged_perceptron_tagger_eng')

if not os.path.exists(tagger_path) and not os.path.exists(tagger_eng_path):
    print(f"WARNING: Neither averaged_perceptron_tagger nor averaged_perceptron_tagger_eng found at expected paths within {nltk_data_path} even after download attempt.")
    print("This might indicate a deeper NLTK installation/permission issue.")
elif os.path.exists(tagger_path) and not os.path.exists(tagger_eng_path):
    print(f"Note: averaged_perceptron_tagger found at {tagger_path}, but averaged_perceptron_tagger_eng is not. This is common.")


ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' is loaded for main.py.")
except OSError:
    print("spaCy model 'en_core_web_sm' not found for main.py. Downloading...")
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model downloaded and loaded for main.py.")
except Exception as e:
    print(f"Error loading spaCy model in main.py: {e}")
    nlp = None 


def get_wordnet_pos(treebank_tag):
    """
    Convert Treebank POS tags to WordNet POS tags.
    """
    if treebank_tag.startswith('J'):
        return 'a' 
    elif treebank_tag.startswith('V'):
        return 'v'  
    elif treebank_tag.startswith('N'):
        return 'n'  
    elif treebank_tag.startswith('R'):
        return 'r'  
    else:
        return 'n'  


def process_text(text, task):
    print(f"Processing task: {task} | Text: {text}")
    try:
        if task == "tokenize":
            return word_tokenize(text)
        elif task == "stem":
            return [ps.stem(token) for token in word_tokenize(text)]
        elif task == "lemmatize":
            tokens = word_tokenize(text)
            try:
                pos_tags = pos_tag(tokens)
            except LookupError as e:
                print(f"LookupError during pos_tag in lemmatize: {e}")
                print("This error indicates the tagger cannot be found by pos_tag even after attempts.")
                print("Please verify 'averaged_perceptron_tagger' or 'averaged_perceptron_tagger_eng' is in your nltk_data/taggers folder.")
                raise RuntimeError(f"Failed to perform POS tagging required for lemmatization: {e}")
            lemmas = [lemmatizer.lemmatize(token, pos=get_wordnet_pos(tag)) for token, tag in pos_tags]
            return lemmas
        elif task == "pos":
            if nlp:
                doc = nlp(text)
                pos_tags = [(token.text, token.tag_) for token in doc]  
                return pos_tags
            else:
                try:
                    return pos_tag(word_tokenize(text))
                except LookupError as e:
                    return [f"Error: NLTK POS tagger not found for fallback: {e}"]
        elif task == "ner":
            if nlp:
                doc = nlp(text)
                return [(ent.text, ent.label_) for ent in doc.ents]
            else:
                return ["spaCy model was not loaded. NER task cannot be performed"]
        else:
            return ["Invalid task."]
    except Exception as e:
        return [f"Error during processing: {str(e)}"]

if __name__ == "__main__":
    sample_text = "This is an example sentence for natural language processing. The dogs are running quickly, and it is better to be good. They are dancing. I am swimming."

    print("\n--- Tokenization ---")
    tokens = process_text(sample_text, "tokenize")
    print(tokens)

    print("\n--- Stemming ---")
    stemmed_words = process_text(sample_text, "stem")
    print(stemmed_words)

    print("\n--- Lemmatization ---")
    lemmatized_words = process_text(sample_text, "lemmatize")
    print(lemmatized_words)

    print("\n--- Part-of-Speech Tagging ---")
    pos_tags = process_text(sample_text, "pos")
    print(pos_tags)

    print("\n--- Named Entity Recognition ---")
    ner_entities = process_text(sample_text, "ner")
    print(ner_entities)