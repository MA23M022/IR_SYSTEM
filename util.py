# Add your import statements here
import nltk
import re
import json
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')  					# Download the punkt tokenizer model
nltk.download('punkt_tab')  				# Ensure punkt_tab is available
nltk.download('stopwords')       	        # Ensure stopwords are downloaded
nltk.download('wordnet')
nltk.download('words')


# Add any utility functions here

abbreviations = ["Dr.", "Mr.", "Mrs.", "Ms.", "U.S.", "etc.", "e.g.", "i.e."]
def refine_splits(sentences):
    for i in range(len(sentences) - 1):
        if any(sentences[i].endswith(abbr) for abbr in abbreviations):
            sentences[i] += " " + sentences[i + 1]
            sentences[i + 1] = ""
    refinedText = [s for s in sentences if s]
    return refinedText
