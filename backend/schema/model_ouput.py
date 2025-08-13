from pydantic import BaseModel, Field
from typing import Annotated, List


class ModelOutput(BaseModel):
    original_query : Annotated[str, Field(..., description="The Query given by user",
                                 examples = ["what papers are avalable on the buckling of emty cylindrical shells"])]
    corrected_query : Annotated[str, Field(..., description="The Query corrected by spell check and auto completion",
                examples = ["what papers are available on the buckling of empty cylindrical shells under non-uniform pressure ."])]
    doc_ids : Annotated[List[int], Field(..., description="Top 5 relevent documents", examples=[[739, 887, 897, 1068, 740]])]

