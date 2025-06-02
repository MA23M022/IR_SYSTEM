from util import *
import ast
import pickle
from collections import Counter, defaultdict

# Load Cranfield dataset
with open('full_output/tokenized_queries.txt', 'r', encoding='utf-8') as f:
    data_str = f.read()

datas = ast.literal_eval(data_str)

# now 'data' is a Python list of lists of lists
documents = []
for data in datas:
    for ele in data:
        documents.append(ele)

#print(type(documents))
#print(documents[0])
#print(len(documents))

# Initialize counters
word_counts = Counter()
cooc_counts = defaultdict(Counter)  # cooc_counts[w][c] = count

window_size = 3

# Build counts
for doc in documents:
    for idx, w in enumerate(doc):
        word_counts[w] += 1
        # context window
        left = max(0, idx - window_size)
        right = min(len(doc), idx + window_size + 1)
        context_words = doc[left:idx] + doc[idx+1:right]
        for c in context_words:
            cooc_counts[w][c] += 1

# Compute probabilities
total_words = sum(word_counts.values())

P_w = {w: count / total_words for w, count in word_counts.items()}

P_c_given_w = {}
for w, counter in cooc_counts.items():
    total_cooc = sum(counter.values())
    P_c_given_w[w] = {c: count / total_cooc for c, count in counter.items()}

#print(f"Probability of word w : {P_w}")

#print(f"Probability of c given w : {P_c_given_w}")

with open("p_w.pkl", "wb") as f:
    pickle.dump(P_w, f)

with open("p_c_given_w.pkl", "wb") as f:
    pickle.dump(P_c_given_w, f)

print("Probability calculation has been done")