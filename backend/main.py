import time
from fastapi import FastAPI
from backend.model.search_engine import SearchEngine
from backend.schema.user_input import UserInput
from backend.schema.model_ouput import ModelOutput
from fastapi.responses import JSONResponse



app = FastAPI()


@app.get("/")
def welcome():
    return {"message" : "welcome to Information Retrival App"}


@app.get("/health")
def health_check():
    return {
        "status" : "OK",
        "model_loaded" : SearchEngine is not None
    }

@app.post("/relevent_docs", response_model = ModelOutput)
def search_relevent_docs(data : UserInput):
    class Args:
        dataset = "cranfield/"
        out_folder = "output/"
        segmenter = data.segmenter
        tokenizer = data.tokenizer
        method = data.method

    args = Args()

    try:
        searchEngine = SearchEngine(args)
        searchEngine.set_custom_query(data.query)
        response = searchEngine.handleCustomQuery()
        return JSONResponse(status_code=200, content={"original_query" : response["original_query"],
                                                      "corrected_query" : response["corrected_query"],
                                                      "doc_ids" : response["doc_ids"]})
    except Exception as e:
        return JSONResponse(status_code=500, content=str(e))



# def get_top_doc_ids(query):
#     class Args:
#         dataset = "cranfield/"
#         out_folder = "output/"
#         segmenter = "punkt"
#         tokenizer = "ptb"
#         method = "lsa"

#     args = Args()
#     searchEngine = SearchEngine(args)
#     searchEngine.set_custom_query(query)  
#     corrected_query, doc_ids = searchEngine.handleCustomQuery()

#     print(corrected_query, doc_ids[:5])
#     print(f"Type of doc ids : {type(doc_ids[0])}")
#     return corrected_query, doc_ids[:5]


# if __name__ == "__main__":
#     query = "what papers are avalable on the buckling of emty cylindrical shells"
#     get_top_doc_ids(query)