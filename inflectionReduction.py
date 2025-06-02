from util import *
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
# Add your import statements here
from nltk.stem import PorterStemmer



class InflectionReduction:
	def __init__(self):
		self.stemmer = PorterStemmer()

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		#Fill in code here
		reducedText = [[self.stemmer.stem(word) for word in sentence] for sentence in text]
		return reducedText
