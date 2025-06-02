from util import *
from collections import defaultdict
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# def build_ngram_vocabulary(words, n=2):
#     """Extract all unique ngrams from vocabulary"""
#     ngram_set = set()
#     for word in words:
#         ngrams = [word[i:i+n] for i in range(len(word)-n+1)]
#         ngram_set.update(ngrams)
#     return sorted(list(ngram_set))

# def vectorize_word(word, ngrams_list, n=2):
#     """Convert a word into a binary vector based on ngrams"""
#     word_ngrams = set([word[i:i+n] for i in range(len(word)-n+1)])
#     return np.array([1 if ng in word_ngrams else 0 for ng in ngrams_list])

# def build_vocab_matrix(vocab, ngrams_list, n=2):
#     """Build matrix where each row is a word vector"""
#     return np.vstack([vectorize_word(word, ngrams_list, n) for word in vocab])

# def find_candidates(wrong_word, vocab_matrix, vocab, ngrams_list, threshold=0.5):
#     """Find candidates with cosine similarity above threshold"""
#     wrong_vec = vectorize_word(wrong_word, ngrams_list, ngrams_list[0].__len__())
#     sims = cosine_similarity([wrong_vec], vocab_matrix)[0]
#     candidates = [(word, sim) for word, sim in zip(vocab, sims) if sim >= threshold]
#     return sorted(candidates, key=lambda x: -x[1])


# if __name__ == "__main__":
#     vocab = None
#     n = 2

#     # Load vocabulary from file
#     with open('vocab_words.txt', 'r') as f:
#         vocab = [line.strip() for line in f]

#     ngrams_list = build_ngram_vocabulary(vocab, n)
#     print("All ngrams:", ngrams_list)

#     vocab_matrix = build_vocab_matrix(vocab, ngrams_list, n)
#     #print("The vocab matrix is")
#     #print(vocab_matrix)
#     #wrong_word = "acple"

#     # After building vocab_matrix, ngrams_list
#     save_model(vocab_matrix, vocab, ngrams_list)
#     print("Model saved!")







def build_ngram_vocabulary(words, n_list=[1, 2]):
    """Extract all unique ngrams from vocabulary for all n values"""
    ngram_set = set()
    for word in words:
        for n in n_list:
            ngrams = [word[i:i+n] for i in range(len(word)-n+1)]
            ngram_set.update(ngrams)
    return sorted(list(ngram_set))

def vectorize_word(word, ngrams_list, n_list=[1, 2]):
    """Convert a word into a binary vector based on all ngrams"""
    word_ngrams = set()
    for n in n_list:
        word_ngrams.update([word[i:i+n] for i in range(len(word)-n+1)])
    return np.array([1 if ng in word_ngrams else 0 for ng in ngrams_list])

def build_vocab_matrix(vocab, ngrams_list, n_list=[1, 2]):
    """Build matrix where each row is a word vector with combined ngrams"""
    return np.vstack([vectorize_word(word, ngrams_list, n_list) for word in vocab])

def find_candidates(wrong_word, vocab_matrix, vocab, ngrams_list, n_list=[1, 2], threshold=0.5):
    """Find candidates with cosine similarity above threshold"""
    wrong_vec = vectorize_word(wrong_word, ngrams_list, n_list)
    sims = cosine_similarity([wrong_vec], vocab_matrix)[0]
    candidates = [(word, sim) for word, sim in zip(vocab, sims) if sim >= threshold]
    return sorted(candidates, key=lambda x: -x[1])

def save_model(vocab_matrix, vocab_list, ngrams_list, filename_prefix='spellcheck_ngram'):
    np.save(f"{filename_prefix}_matrix.npy", vocab_matrix)
    with open(f"{filename_prefix}_meta.pkl", "wb") as f:
        pickle.dump({'vocab': vocab_list, 'ngrams': ngrams_list}, f)

def load_model(filename_prefix='spellcheck_ngram'):
    vocab_matrix = np.load(f"{filename_prefix}_matrix.npy")
    with open(f"{filename_prefix}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_list = meta['vocab']
    ngrams_list = meta['ngrams']
    return vocab_matrix, vocab_list, ngrams_list


if __name__ == "__main__":
    vocab = None
    n_list = [1, 2]  # use unigrams + bigrams

    with open('vocab_words.txt', 'r') as f:
        vocab = [line.strip() for line in f]

    ngrams_list = build_ngram_vocabulary(vocab, n_list)
    print("All ngrams:", ngrams_list)

    vocab_matrix = build_vocab_matrix(vocab, ngrams_list, n_list)
    save_model(vocab_matrix, vocab, ngrams_list)
    print("Model saved!")

