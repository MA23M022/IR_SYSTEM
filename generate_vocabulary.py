from util import *
import ast
import re
#from nltk.corpus import wordnet as wn
#vocab = set(w.name().split('.')[0] for w in wn.all_synsets())

#from nltk.corpus import words
#vocab = set(word.lower() for word in words.words())

# from nltk.corpus import wordnet as wn
# vocab_1 = set(lemma.name() for synset in wn.all_synsets() for lemma in synset.lemmas())

with open('full_output/tokenized_queries.txt', 'r', encoding='utf-8') as f:
    data_queries_str = f.read()

data_queries = ast.literal_eval(data_queries_str)

vocab_1 = set()
for doc in data_queries:
    for sentence in doc:
        for word in sentence:
            vocab_1.add(word)

with open('full_output/tokenized_docs.txt', 'r', encoding='utf-8') as f:
    data_docs_str = f.read()

data_docs = ast.literal_eval(data_docs_str)

vocab_2 = set()
for doc in data_docs:
    for sentence in doc:
        for word in sentence:
            vocab_2.add(word)

combined_vocab = vocab_1 | vocab_2
combined_vocab_list = list(combined_vocab)
cleaned_list = [re.sub(r'[^a-zA-Z-]', '', word) for word in combined_vocab_list]

# Filter only words that are in vocabulary
#correct_word_list = [word for word in cleaned_list if word.lower() in vocab]
#correct_words = list(combined_vocab)
#correct_word_list = [word.lower() for word in correct_words]
correct_word_list = [word.lower() for word in cleaned_list]

print(type(correct_word_list))
count = 0
with open('vocab_words.txt', 'w') as f:
    for word in correct_word_list:
        if word:
            f.write(word + '\n')
            count += 1
            
print(f"Saved {count} words to vocab_words.txt")

