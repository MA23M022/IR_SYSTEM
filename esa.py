# esaRetrieval.py
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# class ESARetrieval:

#     def __init__(self):
#         self.doc_ids = []
#         self.docs = []
#         self.vectorizer = TfidfVectorizer()
#         self.doc_concept_matrix = None
        
#     def preprocess(self, docs):
#         return [' '.join(token for sentence in doc for token in sentence) for doc in docs]

#     def buildIndex(self, docs, doc_ids):
#         self.docs = self.preprocess(docs)
#         # print(len(self.docs))  # 1400
#         self.doc_ids = doc_ids
#         self.doc_concept_matrix = self.vectorizer.fit_transform(self.docs)  # shape: (num_docs, num_terms)
#         print(self.doc_concept_matrix.shape)  # (1400, 5179)
        
#     def rank(self, queries):
#         processed_queries = self.preprocess(queries)
#         query_vecs = self.vectorizer.transform(processed_queries)
#         doc_IDs_ordered = []
#         for query_vec in query_vecs:
#             query_scores = cosine_similarity(query_vec, self.doc_concept_matrix).flatten()
#             ranked_indices = np.argsort(query_scores)[::-1]
#             ranked_doc_ids = [self.doc_ids[i] for i in ranked_indices]
#             doc_IDs_ordered.append(ranked_doc_ids)
#         return doc_IDs_ordered



import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ESARetrieval:

    def __init__(self, term_concept_matrix, vocab):
        """
        term_concept_matrix: np.ndarray of shape (num_terms, num_concepts)
        vocab: list of terms corresponding to rows in term_concept_matrix
        """
        self.term_concept_matrix = term_concept_matrix
        self.vocab = vocab
        self.doc_ids = []
        self.docs = []
        self.vectorizer = TfidfVectorizer(vocabulary=vocab)  # fix vocab
        self.doc_concept_matrix = None

    def preprocess(self, docs):
        return [' '.join(token for sentence in doc for token in sentence) for doc in docs]

    def buildIndex(self, docs, doc_ids):
        self.docs = self.preprocess(docs)
        self.doc_ids = doc_ids
        doc_term_matrix = self.vectorizer.fit_transform(self.docs)  # shape: (num_docs, num_terms)
        print(f"doc_term_matrix shape: {doc_term_matrix.shape}")

        # Convert to concept space: doc_term_matrix × term_concept_matrix
        self.doc_concept_matrix = doc_term_matrix.dot(self.term_concept_matrix)
        print(f"doc_concept_matrix shape: {self.doc_concept_matrix.shape}")


    def rank(self, queries):
        processed_queries = self.preprocess(queries)
        query_term_matrix = self.vectorizer.transform(processed_queries)
        query_concept_matrix = query_term_matrix.dot(self.term_concept_matrix)

        doc_IDs_ordered = []
        for query_vec in query_concept_matrix:
            query_vec_2d = query_vec.reshape(1, -1)  # reshape to (1, num_concepts)
            query_scores = cosine_similarity(query_vec_2d, self.doc_concept_matrix).flatten()
            ranked_indices = np.argsort(query_scores)[::-1]
            ranked_doc_ids = [self.doc_ids[i] for i in ranked_indices]
            doc_IDs_ordered.append(ranked_doc_ids)
        return doc_IDs_ordered


    # def rank(self, queries):
    #     processed_queries = self.preprocess(queries)
    #     query_term_matrix = self.vectorizer.transform(processed_queries)
    #     query_concept_matrix = query_term_matrix.dot(self.term_concept_matrix)

    #     doc_IDs_ordered = []
    #     for query_vec in query_concept_matrix:
    #         query_scores = cosine_similarity(query_vec, self.doc_concept_matrix).flatten()
    #         ranked_indices = np.argsort(query_scores)[::-1]
    #         ranked_doc_ids = [self.doc_ids[i] for i in ranked_indices]
    #         doc_IDs_ordered.append(ranked_doc_ids)
    #     return doc_IDs_ordered


