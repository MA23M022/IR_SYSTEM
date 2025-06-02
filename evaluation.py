from util import *
from collections import defaultdict

# Add your import statements here
import math


class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		retrieved_k = query_doc_IDs_ordered[:k]
		relevant_retrieved = sum([1 for doc_id in retrieved_k if doc_id in true_doc_IDs])
		precision = relevant_retrieved / k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		precisions = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = [int(rel['id']) for rel in qrels if int(rel['query_num']) == query_id]
			precision = self.queryPrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			precisions.append(precision)
		meanPrecision = sum(precisions) / len(precisions)
		return meanPrecision


	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		retrieved_k = query_doc_IDs_ordered[:k]
		relevant_retrieved = sum([1 for doc_id in retrieved_k if doc_id in true_doc_IDs])
		recall = relevant_retrieved / len(true_doc_IDs) if true_doc_IDs else 0.0
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		recalls = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = [int(rel['id']) for rel in qrels if int(rel['query_num']) == query_id]
			recall = self.queryRecall(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			recalls.append(recall)
		meanRecall = sum(recalls) / len(recalls)
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1
		beta = 0.5
		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		if (precision + recall) > 0:
			fscore = ((1+beta**2) * precision * recall) / ((beta**2 * precision) + recall)
		else:
			fscore = 0.0
		return fscore



	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		fscores = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = [int(rel['id']) for rel in qrels if int(rel['query_num']) == query_id]
			fscore = self.queryFscore(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			fscores.append(fscore)
		meanFscore = sum(fscores) / len(fscores)
		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, true_doc_positions, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1
		#print(true_doc_positions)
		#Fill in code here
		retrieved_k = query_doc_IDs_ordered[:k]

		# Compute DCG
		dcg = 0.0
		position_list = []
		for i, doc_id in enumerate(retrieved_k):
			rel = 0
			if doc_id in true_doc_IDs:
				rel = 5 - true_doc_positions[doc_id]
			#if doc_id in true_doc_IDs else 0
			position_list.append(rel)
			dcg += rel / math.log2(i + 2)

		# Compute IDCG
		#ideal_rels = [1] * min(len(true_doc_IDs), k)
		position_list.sort(reverse=True)
		idcg = 0.0
		for i, rel in enumerate(position_list):
			idcg += rel / math.log2(i + 2)

		nDCG = dcg / idcg if idcg > 0 else 0.0
		return nDCG



	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		ndcgs = []
		#print(query_ids)
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = []
			true_doc_positions = defaultdict(int)
			for rel in qrels:
				if int(rel["query_num"]) == query_id:
					ind = int(rel["id"])
					pos_ind = int(rel["position"])
					true_doc_IDs.append(ind)
					true_doc_positions[ind] = pos_ind
			#true_doc_IDs = [int(rel['id']) for rel in qrels if int(rel['query_num']) == query_id]
			ndcg = self.queryNDCG(doc_IDs_ordered[i], query_id, true_doc_IDs, true_doc_positions, k)
			ndcgs.append(ndcg)
		meanNDCG = sum(ndcgs) / len(ndcgs)
		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		retrieved_k = query_doc_IDs_ordered[:k]
		num_relevant = 0
		sum_precisions = 0.0

		for i, doc_id in enumerate(retrieved_k):
			if doc_id in true_doc_IDs:
				num_relevant += 1
				precision_at_i = num_relevant / (i + 1)
				sum_precisions += precision_at_i

		avgPrecision = sum_precisions / len(true_doc_IDs) if true_doc_IDs else 0.0
		return avgPrecision



	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		avg_precisions = []
		for i, query_id in enumerate(query_ids):
			true_doc_IDs = [int(rel['id']) for rel in q_rels if int(rel['query_num']) == query_id]
			avg_precision = self.queryAveragePrecision(doc_IDs_ordered[i], query_id, true_doc_IDs, k)
			avg_precisions.append(avg_precision)
		meanAveragePrecision = sum(avg_precisions) / len(avg_precisions)
		return meanAveragePrecision

