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

from concept_matrix_computation import load_esa_model

term_concept_matrix, vocab = load_esa_model(prefix='cranfield_esa_model')

vocab_set = set(vocab)

if "apple" in vocab_set:
    print("Article presents")
else:
    print("Article does not present")

# print("The vocabulary for concept is : ")
# for i in range(500):
#     print(vocab[i])
