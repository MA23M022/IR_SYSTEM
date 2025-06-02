import json
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from spell_check import spell_check_function

nltk.download('punkt')


class PhraseTrieNode:
    def __init__(self):
        self.children = defaultdict(PhraseTrieNode)
        self.is_end = False
        self.freq = 0


class AutocompleteSystem:
    def __init__(self, query_data):
        self.query_data = query_data
        self.query_texts = [q["query"].lower().strip() for q in query_data]
        self.root = PhraseTrieNode()
        self._build_phrase_trie()
        self._build_inverted_index()
        self._build_tfidf_bm25()
        self._build_ngram_model()

    def _build_phrase_trie(self):
        for query in self.query_texts:
            tokens = query.split()
            node = self.root
            for token in tokens:
                node = node.children[token]
            node.is_end = True
            node.freq += 1

    def _dfs_trie(self, node, path, results):
        if node.is_end:
            results.append((" ".join(path), node.freq))
        for word, child in node.children.items():
            self._dfs_trie(child, path + [word], results)

    def _get_phrase_completions(self, prefix):
        tokens = prefix.lower().strip().split()
        node = self.root
        for token in tokens:
            if token not in node.children:
                return []
            node = node.children[token]
        results = []
        self._dfs_trie(node, tokens, results)
        return results

    def _build_inverted_index(self):
        self.inverted_index = defaultdict(set)
        for i, query in enumerate(self.query_texts):
            for word in query.split():
                self.inverted_index[word].add(i)

    def _get_infix_completions(self, word):
        indices = self.inverted_index.get(word, set())
        return [(self.query_texts[i], 1) for i in indices]

    def _build_tfidf_bm25(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.query_texts)

        tokenized_corpus = [word_tokenize(q) for q in self.query_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _score_tfidf_bm25(self, prefix, suggestions):
        tfidf_vec = self.tfidf_vectorizer.transform([prefix])
        bm25_scores = self.bm25.get_scores(word_tokenize(prefix))
        # Normalize BM25 scores to range [0, 1]
        # max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        max_bm25 = np.max(bm25_scores) if bm25_scores.size > 0 else 1.0

        normalized_bm25_scores = bm25_scores / max_bm25
        scores = []
        for s, _ in suggestions:
            try:
                idx = self.query_texts.index(s)
                tfidf_score = np.dot(self.tfidf_matrix[idx].toarray(), tfidf_vec.toarray().T)[0][0]
                bm25_score = normalized_bm25_scores[idx]
                combined = 0.5 * tfidf_score + 0.5 * bm25_score
                scores.append((s, combined))
            except ValueError:
                scores.append((s, 0.0))
        return sorted(scores, key=lambda x: -x[1])[:5]

    def _build_ngram_model(self):
        all_tokens = []
        for query in self.query_texts:
            tokens = word_tokenize(query)
            all_tokens.extend(tokens)
        self.bigram_counts = Counter(ngrams(all_tokens, 2))

    def _predict_next_words(self, prefix):
        tokens = prefix.lower().strip().split()
        if not tokens:
            return []
        last = tokens[-1]
        candidates = [(b[1], count) for b, count in self.bigram_counts.items() if b[0] == last]
        return sorted(candidates, key=lambda x: -x[1])[:3]

    def autocomplete(self, prefix):
        completions = self._get_phrase_completions(prefix)
        last_word = prefix.strip().split()[-1] if prefix.strip() else ""
        infix_matches = self._get_infix_completions(last_word) if last_word else []

        # Merge and deduplicate suggestions
        all_candidates_dict = {c[0]: c for c in completions + infix_matches}
        all_candidates = list(all_candidates_dict.values())

        ranked = self._score_tfidf_bm25(prefix, all_candidates)
        predicted_next = self._predict_next_words(prefix)

        print("\n Next Word Predictions:")
        for word, freq in predicted_next:
            print(f" - {word} (freq: {freq})")

        return ranked


def load_queries_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def correct_sentence_function(sentence):
    modified_sentence = spell_check_function(sentence)
    correct_sentence = " ".join(word for word in modified_sentence)
    return correct_sentence

def autoCompletion_function(query):
    queries = load_queries_from_json("cranfield/cran_queries.json")
    system = AutocompleteSystem(queries)
    
    corrected_user_input = correct_sentence_function(query)
    print(f"Correct sentence : {corrected_user_input}")
    suggestions = system.autocomplete(corrected_user_input)
    print("\n Suggestions:")
    sorted_suggestions = sorted(suggestions, key = lambda x : x[1], reverse=True)
    for suggestion, score in sorted_suggestions:
        print(f" - {suggestion} (score: {score:.4f})")

    return sorted_suggestions[0][0]


if __name__ == "__main__":
    queries = load_queries_from_json("cranfield/cran_queries.json")
    system = AutocompleteSystem(queries)

    print(" Query Autocompletion (type 'exit' to quit):")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break   
        corrected_user_input = correct_sentence_function(user_input)
        print(f"Correct sentence : {corrected_user_input}")
        suggestions = system.autocomplete(corrected_user_input)
        print("\n Suggestions:")
        sorted_suggestions = sorted(suggestions, key = lambda x : x[1], reverse=True)
        for suggestion, score in sorted_suggestions:
            print(f" - {suggestion} (score: {score:.4f})")




