import requests
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class DBESA:
    def __init__(self):
        self.doc_ids = []
        self.vectorizer = CountVectorizer(tokenizer=lambda x: x.split())  # concept vectorizer

    def get_dbpedia_concepts(self, text, confidence=0.5):
        headers = {'Accept': 'application/json'}
        params = {'text': text, 'confidence': confidence}
        try:
            response = requests.get(
                "https://api.dbpedia-spotlight.org/en/annotate",
                headers=headers,
                params=params,
                timeout=5,
                verify=False  # <- This skips SSL verification
            )
            concepts = []
            if response.status_code == 200:
                data = response.json()
                if 'Resources' in data:
                    concepts = [res['@URI'] for res in data['Resources']]
            return concepts
        except:
            return []

    def buildIndex(self, docs, doc_ids):
        self.doc_ids = doc_ids
        concept_docs = []
        for doc in docs:
            flat_text = ' '.join(token for sent in doc for token in sent)
            concepts = self.get_dbpedia_concepts(flat_text)
            # print(concepts)
            concept_docs.append(' '.join(concepts))  # Join concept URIs
        # print(len(concept_docs))  # 1400
        self.doc_concept_matrix = self.vectorizer.fit_transform(concept_docs)
        print(self.doc_concept_matrix.shape)  # (1400, 927)

    def rank(self, queries):
        doc_IDs_ordered = []
        for query in queries:
            query_text = ' '.join(token for sent in query for token in sent)
            concepts = self.get_dbpedia_concepts(query_text)
            concept_str = ' '.join(concepts)
            query_vec = self.vectorizer.transform([concept_str])
            scores = cosine_similarity(query_vec, self.doc_concept_matrix).flatten()
            ranked = np.argsort(scores)[::-1]
            ranked_ids = [self.doc_ids[i] for i in ranked]
            doc_IDs_ordered.append(ranked_ids)
        return doc_IDs_ordered
