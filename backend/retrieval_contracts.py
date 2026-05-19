from __future__ import annotations

from pydantic import BaseModel, Field


_RETRIEVAL_DOCUMENT_SCHEMA_Q1A = {
    "version": "q1a.v1",
    "tiers": ["memory", "episode", "chapter", "arc", "saga"],
    "fields": [
        "schema_version",
        "document_id",
        "tier",
        "source_type",
        "source_subtype",
        "source_row_id",
        "source_table",
        "chat_session_id",
        "from_turn",
        "to_turn",
        "turn_index",
        "title",
        "text",
        "similarity",
        "created_at",
        "query_matched",
        "metadata",
    ],
}


class RetrievalDocumentQ1A(BaseModel):
    schema_version: str = _RETRIEVAL_DOCUMENT_SCHEMA_Q1A["version"]
    document_id: str
    tier: str
    source_type: str
    source_subtype: str | None = None
    source_row_id: int | None = None
    source_table: str | None = None
    chat_session_id: str = ""
    from_turn: int | None = None
    to_turn: int | None = None
    turn_index: int | None = None
    title: str | None = None
    text: str = ""
    similarity: float | None = None
    created_at: str | None = None
    query_matched: bool = True
    metadata: dict = Field(default_factory=dict)
