from pydantic import BaseModel, Field
from typing import Annotated, Literal


class UserInput(BaseModel):
    segmenter : Annotated[Literal["naive", "punkt"], Field(..., description="Sentece segmentation technique", examples=["punkt"])]
    tokenizer : Annotated[Literal["naive", "ptb"], Field(..., description="Tokenization technique", examples=["ptb"])]
    method : Annotated[Literal["tfidf", "lsa", "esa", "dbesa"],
                        Field(..., description="Information Retrival techniques", examples=["lsa"])]
    
    query : Annotated[str, Field(..., description="Give the query",
                                  examples=["what papers are avalable on the buckling of emty cylindrical shells"])]
