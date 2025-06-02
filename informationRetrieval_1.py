from util import *

# Add your import statements here
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lsa import LSA
from esa import ESARetrieval
from dbESA import DBESA


class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docIDs = None
		self.vectorizer = None
		self.doc_vectors = None


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
	

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		#index = None

		#Fill in code here
		self.docIDs = docIDs

		# Flatten documents into strings
		flattened_docs = self.preprocess(docs)

		# Create TF-IDF vectors
		self.vectorizer = TfidfVectorizer()
		self.doc_vectors = self.vectorizer.fit_transform(flattened_docs)

		# Store index
		self.index = self.doc_vectors

		#self.index = index


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		# Preprocess queries into flat text
		flattened_queries = self.preprocess(queries)

		# Transform queries using the trained TF-IDF vectorizer
		query_vectors = self.vectorizer.transform(flattened_queries)

		# Compute cosine similarity between queries and documents
		similarity_matrix = cosine_similarity(query_vectors, self.doc_vectors)
		# print("The similarity matrix is:")
		# print(similarity_matrix)

		# For each query, get the ranking of document IDs based on similarity scores
		for similarities in similarity_matrix:
			ranked_doc_indices = np.argsort(similarities)[::-1]  # descending order
			ranked_docIDs = [self.docIDs[idx] for idx in ranked_doc_indices]
			doc_IDs_ordered.append(ranked_docIDs)
	
		return doc_IDs_ordered


if __name__ == "__main__":
	docs = [
    		[['this', 'is', 'the', 'first', 'document']],
    		[['this', 'document', 'is', 'the', 'second', 'document']],
    		[['and', 'this', 'is', 'the', 'third', 'one']],
    		[['is', 'this', 'the', 'first', 'document']]]
	docIDs = [1, 2, 3, 4]

	queries = [
				[['this', 'is', 'the', 'first']],
				[['third', 'document']]]

	IR_system = InformationRetrieval()
	IR_system.buildIndex(docs, docIDs)
	
	print("The document vectorizer's shape is:")
	print(IR_system.doc_vectors.shape)

	ranks = IR_system.rank(queries)

	print(ranks)

