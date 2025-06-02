from util import *
from nltk.metrics.distance import edit_distance

def filter_candidates_by_distance(candidates, error_word, max_distance=2):
    return [w for w in candidates if edit_distance(w, error_word) <= max_distance]
