import wikipedia
import time
import ast
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from util import *

from nltk.corpus import words
vocab = set(word.lower() for word in words.words())

# -------- Step 1: Extract unique terms --------
def extract_unique_terms(cranfield_docs):
    cranfield_terms = set()
    for doc in cranfield_docs:
        for word in doc:
            if word in vocab:
                cranfield_terms.add(word)
    return cranfield_terms

# -------- Step 2: Fetch Wikipedia articles --------
def fetch_wikipedia_articles(terms, delay=1.0):
    articles = {}
    count = 1
    for term in terms:
        try:
            page = wikipedia.page(term)
            articles[term] = page.content
            print(f"Fetched article for {count}-th term: {term}")
        except Exception as e:
            print(f"Skipped term '{term}': {e}")
        time.sleep(delay)  # avoid rate limiting
        count += 1

    return articles

def fetch_wikipedia_articles_fast(terms, delay=0.5):
    articles = {}
    count = 1
    for term in terms:
        try:
            search_results = wikipedia.search(term)
            if search_results:
                top_title = search_results[0]
                summary = wikipedia.summary(top_title)
                articles[term] = summary
                print(f"Fetched summary for {count}-th term: {term} → {top_title}")
        except Exception as e:
            print(f"Skipped term '{term}': {e}")
        time.sleep(delay)
        count += 1
    return articles


# -------- Step 3: Build term-concept matrix --------
def build_term_concept_matrix_from_articles(articles, max_features=5000):
    texts = list(articles.values())
    vectorizer = TfidfVectorizer(max_features=max_features)
    term_concept_matrix_sparse = vectorizer.fit_transform(texts)
    term_concept_matrix = term_concept_matrix_sparse.T.toarray()
    vocab = vectorizer.get_feature_names_out()
    print(f"Matrix shape: {term_concept_matrix.shape}")
    return term_concept_matrix, vocab

# -------- Step 4: Save ESA model --------
def save_esa_model(term_concept_matrix, vocab, prefix='cranfield_esa_model'):
    np.save(f"{prefix}_matrix.npy", term_concept_matrix)
    with open(f"{prefix}_vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Saved ESA matrix to {prefix}_matrix.npy and vocab to {prefix}_vocab.pkl")

def load_esa_model(prefix='cranfield_esa_model'):
    # Load matrix
    term_concept_matrix = np.load(f"{prefix}_matrix.npy")
    # Load vocab
    with open(f"{prefix}_vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print(f"Loaded ESA matrix shape: {term_concept_matrix.shape}")
    print(f"Loaded vocab size: {len(vocab)}")
    return term_concept_matrix, vocab

def preprocess(docs):
    """
    Flattens the input (list of documents -> list of sentences -> list of words)
    into strings where each document becomes a single string.
    """
    flattened_docs = []
    for doc in docs:
        for sentence in doc:
            flattened_docs.append(sentence)
    # for doc in docs:
    #     sentences = [' '.join(sentence) for sentence in doc]  # join words in each sentence
    #     doc_text = ' '.join(sentences)  # join all sentences
    #     flattened_docs.append(doc_text)
    return flattened_docs



# -------- MAIN --------
if __name__ == "__main__":
    # Example: cranfield_docs input
    with open('full_output/stopword_removed_docs.txt', 'r', encoding='utf-8') as f:
        data_docs_str = f.read()

    data_docs = ast.literal_eval(data_docs_str)

    processed_cranfield_docs = preprocess(data_docs)
    # processed_cranfield_docs = [
    #     ['what', 'are', 'the', 'aeroelastic', 'problems'],
    #     ['heat', 'transfer', 'in', 'supersonic', 'flow'],
    #     ['boundary', 'layer', 'control', 'techniques'],
    #     ['compressible', 'flow', 'theory', 'and', 'applications'],
    #     # Add your full Cranfield token lists here...
    # ]

    #print(processed_cranfield_docs)

    # Extract terms
    cranfield_terms = extract_unique_terms(processed_cranfield_docs)
    print(f"Extracted {len(cranfield_terms)} unique terms.")
    #print(cranfield_terms)

    # Fetch Wikipedia articles
    #wiki_articles = fetch_wikipedia_articles(cranfield_terms, delay=1.0)
    wiki_articles = fetch_wikipedia_articles_fast(cranfield_terms, delay=0.5)
    print(f"Fetched {len(wiki_articles)} Wikipedia articles.")

    # Build term-concept matrix
    term_concept_matrix, vocab = build_term_concept_matrix_from_articles(wiki_articles, max_features=5000)

    # Save ESA model
    save_esa_model(term_concept_matrix, vocab)

