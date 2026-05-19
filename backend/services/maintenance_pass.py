"""L-1a/L-2 maintenance pass contracts."""

from pydantic import BaseModel, Field


__all__ = [
    "DriftSignal",
    "MaintenanceEnqueueRequest",
    "MaintenanceEnqueueResponse",
    "MaintenancePassRequest",
    "MaintenancePassResult",
]


class MaintenancePassRequest(BaseModel):
    """Request body for POST /maintenance-pass/{session_id}.

    Request contract for the maintenance-pass endpoint; result and drift-signal
    contracts live alongside it in this module.
    """

    turn_index: int = -1
    shadow_only: bool = True
    assistant_response: str = ""
    recent_responses: list[str] = Field(default_factory=list)
    supervisor_result: dict | None = None


class DriftSignal(BaseModel):
    """Single drift signal observed during a maintenance pass."""

    signal_type: str
    severity: str = "medium"
    detail: str = ""
    correction_hint: str = ""
    source: str = ""


class MaintenancePassResult(BaseModel):
    """Response body for POST /maintenance-pass/{session_id}."""

    status: str = "ok"
    chat_session_id: str
    executed_at: str
    turn_index: int = -1
    shadow_only: bool = True
    state_status: str = "no_state"
    observations: list[str] = Field(default_factory=list)
    drift_signals: list[DriftSignal] = Field(default_factory=list)
    refresh_suggestions: list[str] = Field(default_factory=list)
    correction_hints: list[str] = Field(default_factory=list)
    patch_suggestions: list[str] = Field(default_factory=list)
    story_plan_refreshed: bool = False
    director_refreshed: bool = False


class MaintenanceEnqueueRequest(BaseModel):
    """Request body for POST /maintenance/enqueue."""

    chat_session_id: str
    turn_index: int = -1
    shadow_only: bool = True
    assistant_response: str = ""
    recent_responses: list[str] = Field(default_factory=list)
    supervisor_result: dict | None = None
    summary_refresh: dict | None = None


class MaintenanceEnqueueResponse(BaseModel):
    """Response body for POST /maintenance/enqueue."""

    status: str
    chat_session_id: str
    turn_index: int
    queue_depth: int = 0
