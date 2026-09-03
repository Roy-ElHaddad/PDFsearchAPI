from pydantic import BaseModel, Field

from app.config import settings


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Free-text search query.")
    top_k: int = Field(
        default=settings.default_top_k,
        ge=1,
        le=settings.max_top_k,
        description="Number of results to return.",
    )


class SearchResult(BaseModel):
    document_name: str
    page_number: int
    chunk_index: int
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
