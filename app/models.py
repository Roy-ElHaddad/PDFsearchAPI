"""Shared data structures used by both the ingestion pipeline and the API."""

from dataclasses import dataclass, asdict


@dataclass
class Chunk:
    """A single unit of retrievable text, plus the metadata needed to trace
    it back to its source document and location within that document."""

    document_name: str
    page_number: int
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            document_name=data["document_name"],
            page_number=data["page_number"],
            chunk_index=data["chunk_index"],
            text=data["text"],
        )
