from util import *
from sentenceSegmentation import SentenceSegmentation

# Add your import statements here

from nltk.tokenize import word_tokenize, TreebankWordTokenizer


class Tokenization():
	def __init__(self):
		self.tokenizer = TreebankWordTokenizer()

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""
		#Fill in code here
		text = [sentence[:-1] for sentence in text]
		tokenizedText = [sentence.split() for sentence in text]  # Splitting sentences into words
		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		#Fill in code here
		text = [sentence[:-1] for sentence in text]
		tokenizedText = [self.tokenizer.tokenize(sentence) for sentence in text]  # Applying Treebank tokenizer
		return tokenizedText
	
