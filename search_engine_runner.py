import time
from search_engine import SearchEngine  

def get_top_doc_ids(query):
    class Args:
        dataset = "cranfield/"
        out_folder = "output/"
        segmenter = "punkt"
        tokenizer = "ptb"
        custom = True
        method = "lsa"

    args = Args()
    searchEngine = SearchEngine(args)
    searchEngine.set_custom_query(query)  
    corrected_query, doc_ids = searchEngine.handleCustomQuery()

    return corrected_query, doc_ids[:5]
