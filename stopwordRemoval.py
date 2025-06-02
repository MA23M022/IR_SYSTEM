from util import *
import os
import json
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction

# Add your import statements here
from nltk.corpus import stopwords

class StopwordRemoval():
	def __init__(self):
		self.stop_words = set(stopwords.words('english'))  					# Load English stopwords

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		#Fill in code here
		stopwordRemovedText = [[word for word in sentence if word.lower() not in self.stop_words] for sentence in text]
		return stopwordRemovedText
