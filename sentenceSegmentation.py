from util import *

# Add your import statements here
from nltk.tokenize import sent_tokenize


class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		#Fill in code here
		segmentedText = re.split(r'(?<=[.!?])\s+', text)
		refinedText = refine_splits(segmentedText)
		return refinedText
												


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		#Fill in code here
		segmentedText = sent_tokenize(text)
		return segmentedText

