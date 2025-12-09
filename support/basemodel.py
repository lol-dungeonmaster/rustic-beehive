from pydantic import BaseModel, field_validator
from typing import Optional

class ChromaDBResult(BaseModel):
    docs: str
    dist: Optional[float] # requires query
    meta: Optional[dict]  # requires get or query
    store_id: str