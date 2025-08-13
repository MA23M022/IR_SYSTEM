# lsa.py
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt

class LSA:
    def __init__(self, n_components=2):
        self.vectorizer = TfidfVectorizer()
        self.svd = TruncatedSVD(n_components=n_components)
        self.doc_vectors = None
        self.docIDs = None

    def preprocess(self, docs):
        """
        Flattens the input (list of documents -> list of sentences -> list of words)
        into strings where each document becomes a single string.
        """
        flattened_docs = []
        for doc in docs:
            sentences = [' '.join(sentence) for sentence in doc]  # join words in each sentence
            doc_text = ' '.join(sentences)  # join all sentences
            flattened_docs.append(doc_text)
        return flattened_docs

    def fit(self, docs, docIDs):
        self.docIDs = docIDs
        flattened = self.preprocess(docs)
        tfidf = self.vectorizer.fit_transform(flattened)
        self.doc_vectors = self.svd.fit_transform(tfidf)
        print("Shape of doc_vectors (U Σ):", self.doc_vectors.shape)
        # # print(self.doc_vectors)  # Each row represents a document in LSA space
        # print("Shape of Vᵗ (components_):", self.svd.components_.shape)
        # print(self.svd.components_)  # Each row is a topic, each column is a term
        # print("Singular values (Σ):", self.svd.singular_values_.shape)
    
        # Plot the singular values
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(1, len(self.svd.explained_variance_) + 1), self.svd.explained_variance_, marker='o')
        # plt.title('Singular Values from Truncated SVD (LSA)')
        # plt.xlabel('Component Index')
        # plt.ylabel('Explained Variance (Singular Value²)')
        # plt.grid(True)
    
        # # Save the figure
        # plt.savefig('singular_values.png', dpi=300, bbox_inches='tight')
        

    def rank(self, queries):
        flattened_queries = self.preprocess(queries)
        query_vecs = self.vectorizer.transform(flattened_queries)
        query_lsa = self.svd.transform(query_vecs)

        doc_IDs_ordered = []
        for qvec in query_lsa:
            sims = self.doc_vectors @ qvec.T
            ranked = np.argsort(sims)[::-1]
            ranked_ids = [self.docIDs[i] for i in ranked]
            doc_IDs_ordered.append(ranked_ids)
        return doc_IDs_ordered
