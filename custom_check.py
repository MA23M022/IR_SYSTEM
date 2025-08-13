# vocab_list = None
# with open('vocab_words.txt', 'r') as f:
#     vocab_list = [line.strip() for line in f]

# # Optional: convert to set for faster lookup
# vocab_set = set(vocab_list)
# print(f"Loaded {len(vocab_set)} words from vocab_words.txt")

# word = "the"
# if word in vocab_list:
#     print("Present")
# else:
#     print("Not present")

# import pickle

# ESA_MODEL = None
# prefix='backend/model/cranfield_esa_model'
# with open(f"{prefix}_vocab.pkl", 'rb') as f:
#         ESA_MODEL = pickle.load(f)
#         print("Model loaded")


# from backend.constants.n_gram import load_model

# vocab_matrix, vocab, ngrams_list = load_model()
# print(f"The vocab matrix is {vocab_matrix}")
# print(f"The vocab matrix shape is {vocab_matrix.shape}")

# print(f"The vocab is {vocab}")

# print(f"The ngram list is {ngrams_list}")

# print("Model loaded!")


import time
from backend.model.search_engine import SearchEngine

def get_top_doc_ids(query):
    class Args:
        dataset = "cranfield/"
        out_folder = "output/"
        segmenter = "punkt"
        tokenizer = "ptb"
        custom = True
        method = "lsa"

    args = Args()
    print("I am going into the backend/model file")
    searchEngine = SearchEngine(args)
    searchEngine.set_custom_query(query)  
    corrected_query, doc_ids = searchEngine.handleCustomQuery()

    print(corrected_query, doc_ids[:5])
    return corrected_query, doc_ids[:5]


if __name__ == "__main__":
    query = "what papers are avalable on the buckling of emty cylindrical shells"
    get_top_doc_ids(query)


# if ESA_MODEL:
#      print("Model loaded")
# else:
#      print("Unable to load")



# from backend.utils.concept_matrix_computation import load_esa_model

# term_concept_matrix, vocab = load_esa_model(prefix='cranfield_esa_model')

# vocab_set = set(vocab)

# if "apple" in vocab_set:
#     print("Article presents")
# else:
#     print("Article does not present")

# print("The vocabulary for concept is : ")
# for i in range(500):
#     print(vocab[i])
