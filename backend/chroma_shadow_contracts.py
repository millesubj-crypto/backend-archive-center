from __future__ import annotations

from pydantic import BaseModel, Field


class ChromaShadowBackfillDryRunRequest(BaseModel):
    chat_session_id: str = ""
    sample_limit_per_tier: int = 2


class ChromaShadowBackfillBatchRequest(BaseModel):
    chat_session_id: str = ""
    batch_size_per_tier: int = 25
    checkpoint: dict = Field(default_factory=dict)
    retry_rows: list[dict] = Field(default_factory=list)


class ChromaShadowReembedAuditRequest(BaseModel):
    chat_session_id: str = ""
    queue_limit_per_tier: int = 25


class ChromaShadowHealthProbeRequest(BaseModel):
    chat_session_id: str = ""


class ChromaShadowFallbackRunbookRequest(BaseModel):
    chat_session_id: str = ""


class ChromaShadowRebuildDrillRequest(BaseModel):
    chat_session_id: str = ""
    batch_size_per_tier: int = 25
    checkpoint: dict = Field(default_factory=dict)
    retry_rows: list[dict] = Field(default_factory=list)
    execute_rebuild_then_rollback: bool = False


class ChromaShadowAdoptionGateRequest(BaseModel):
    chat_session_id: str = ""
    operator_evidence: dict = Field(default_factory=dict)


class ChromaShadowReleaseHygieneRequest(BaseModel):
    chat_session_id: str = ""
    adoption_operator_evidence: dict = Field(default_factory=dict)
    release_evidence: dict = Field(default_factory=dict)


class ChromaShadowVisibilityGuardRequest(BaseModel):
    chat_session_id: str = ""
    sample_limit: int = 20
