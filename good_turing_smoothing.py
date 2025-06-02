from util import *
import ast
import pickle
from collections import Counter, defaultdict

# Load Cranfield dataset
with open('full_output/tokenized_queries.txt', 'r', encoding='utf-8') as f:
    data_str = f.read()

datas = ast.literal_eval(data_str)

documents = []
for data in datas:
    for ele in data:
        documents.append(ele)

# Initialize counters
word_counts = Counter()
cooc_counts = defaultdict(Counter)
window_size = 3

# Build counts
for doc in documents:
    for idx, w in enumerate(doc):
        word_counts[w] += 1
        left = max(0, idx - window_size)
        right = min(len(doc), idx + window_size + 1)
        context_words = doc[left:idx] + doc[idx+1:right]
        for c in context_words:
            cooc_counts[w][c] += 1

# Frequency of frequencies (N_c)
freq_of_freq_w = Counter(word_counts.values())
freq_of_freq_c = defaultdict(Counter)
for w, counter in cooc_counts.items():
    freq_of_freq_c[w] = Counter(counter.values())

# Apply Good-Turing smoothing for P(w)
total_words = sum(word_counts.values())
P_w_gt = {}
for w, c in word_counts.items():
    Nc = freq_of_freq_w[c]
    Nc1 = freq_of_freq_w.get(c + 1, 0)
    if Nc1 > 0:
        c_star = (c + 1) * Nc1 / Nc
    else:
        c_star = c  # fallback to raw count
    P_w_gt[w] = c_star / total_words

# Probability mass for unseen words
N1_w = freq_of_freq_w.get(1, 0)
P_unseen_w = N1_w / total_words

# Apply Good-Turing smoothing for P(c|w)
P_c_given_w_gt = {}
for w, counter in cooc_counts.items():
    Nw = sum(counter.values())
    freq_of_freq = freq_of_freq_c[w]
    P_c_given_w_gt[w] = {}
    N1_c = freq_of_freq.get(1, 0)
    for c, c_count in counter.items():
        Nc = freq_of_freq[c_count]
        Nc1 = freq_of_freq.get(c_count + 1, 0)
        if Nc1 > 0:
            c_star = (c_count + 1) * Nc1 / Nc
        else:
            c_star = c_count
        P_c_given_w_gt[w][c] = c_star / Nw

# Save Good-Turing probabilities
with open("p_w_gt.pkl", "wb") as f:
    pickle.dump(P_w_gt, f)

with open("p_c_given_w_gt.pkl", "wb") as f:
    pickle.dump(P_c_given_w_gt, f)

# Save unseen probability as a fallback
with open("p_unseen_w.pkl", "wb") as f:
    pickle.dump(P_unseen_w, f)

print("Good-Turing probability calculation and saving completed.")
