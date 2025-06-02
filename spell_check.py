from util import *
from collections import defaultdict
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from n_gram import load_model, find_candidates
from edit_distance_calculation import filter_candidates_by_distance
from nltk.corpus import stopwords


stop_words = set(stopwords.words('english'))
vocab_list = None
with open('vocab_words.txt', 'r') as f:
    vocab_list = [line.strip() for line in f]

# Optional: convert to set for faster lookup
vocab_set = set(vocab_list)
print(f"Loaded {len(vocab_set)} words from vocab_words.txt")

# In another script / session
vocab_matrix, vocab, ngrams_list = load_model()
print("Model loaded!")

P_w = None
with open("p_w.pkl", "rb") as f:
    P_w = pickle.load(f)

P_c_given_w = None
with open("p_c_given_w.pkl", "rb") as f:
    P_c_given_w = pickle.load(f)

print("Probabilities loaded")


P_w_gt = None
with open("p_w_gt.pkl", "rb") as f:
    P_w_gt = pickle.load(f)

P_c_given_w_gt = None
with open("p_c_given_w_gt.pkl", "rb") as f:
    P_c_given_w_gt = pickle.load(f)

P_unseen_w = None
with open("p_unseen_w.pkl", "rb") as f:
    P_unseen_w = pickle.load(f)

print("Loaded Good-Turing smoothed probabilities.")



def score_candidate(w, context_words, P_w, P_c_given_w, smoothing=1e-8):
    # P(w): probability of candidate
    prob_w = P_w.get(w, smoothing)
    score = prob_w
    
    for c in context_words:
        prob_c_given_w = P_c_given_w.get(w, {}).get(c, smoothing)
        score *= prob_c_given_w
    return score

def filtering_context_word(context_words):
    # when counting context words:
    filtered_context = [word for word in context_words if word.lower() not in stop_words]
    return filtered_context



#sentence = "what papers are avalable on the bucling of empty cylndrical shells"
#sentence = "how do lsrge charges in new mass raio quantitatively affect wing-flutter boundaries"
#sentence = "what is the effact of tha shape of the drugs polar of a lifting spcevraft"
#sentence = "teh"

def spell_check_function(sentence):
    tokens = sentence.split()
    print(f"The sentence tokens are {tokens}")


    modified_sentence = []
    window = 3
    for i, token in enumerate(tokens):
        if token in vocab_set:
            modified_sentence.append(token)
            print("Valid word!")
        else:
            print(f"Invalid word : '{token}' , Searching for candidate words")
            candidates = find_candidates(token, vocab_matrix, vocab, ngrams_list, threshold=0.3)
            rough_candidate_words = [candidate_word[0] for candidate_word in candidates]

            actual_candidate_words = []
            if rough_candidate_words:
                actual_candidate_words = filter_candidates_by_distance(rough_candidate_words,
                                                                        token, max_distance=2)
                # print(f"The rough candidate words are {rough_candidate_words}")
                # print(f"The length of rough candidate list is {len(rough_candidate_words)}")

            print(f"The actual candidate words for '{token}' is {actual_candidate_words}")

            if len(actual_candidate_words) == 0:
                modified_sentence.append(token)
            elif(len(actual_candidate_words) == 1):
                modified_sentence.append(actual_candidate_words[0])
            else:
                error_idx = i  # "improtant"
                context_words = []
                for j in range(error_idx - window, error_idx + window + 1):
                    if j != error_idx and 0 <= j < len(tokens):
                        context_words.append(tokens[j])
                
                #context_words = filtering_context_word(context_words)
                print(f"Context words for '{token}':", context_words)
                scores = {}
                for candidate in actual_candidate_words:
                    # Simple smoothing
                    scores[candidate] = score_candidate(candidate, context_words, P_w, P_c_given_w)

                    # Good turing smoothing
                    #scores[candidate] = score_candidate(candidate, context_words, P_w_gt, P_c_given_w_gt, smoothing=P_unseen_w)
                                                                # pass fallback for unseen words
                # sort by score
                ranked_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for word, score in ranked_candidates:
                    print(f"Candidate: '{word}' , Score: {score}")

                modified_sentence.append(ranked_candidates[0][0])


    print("The modified sentence is:")
    print(modified_sentence)
    return modified_sentence


#print(f"The actual candidate words are: {actual_candidate_words}")
#print(f"The length of actual candiadte list is {len(actual_candidate_words)}")



# candidates = find_candidates(wrong_word, vocab_matrix, vocab, ngrams_list, threshold=0.3)
# print("Candidates:", candidates)
