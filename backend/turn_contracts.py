from __future__ import annotations

from pydantic import BaseModel, Field


class CompleteTurnRequest(BaseModel):
    turn_index: int
    turn_content: str
    context: list[dict] = Field(default_factory=list)
    chat_session_id: str = ""
    output_language_override: dict | None = None


class M4CompleteTurnRequest(BaseModel):
    chat_session_id: str
    turn_index: int
    user_input: str = ""
    assistant_content: str = ""
    context_messages: list[dict] = Field(default_factory=list)
    improvement_trace: dict | None = None
    output_language_override: dict | None = None
    request_type: str = "model"
    client_meta: dict = Field(default_factory=dict)


class M4CompleteTurnResponse(BaseModel):
    status: str = "ok"
    chat_session_id: str = ""
    turn_index: int = 0
    generated_at: str = ""
    save_ok: bool = False
    save_error: str | None = None
    critic_triggered: bool = False
    critic_result: dict | None = None
    episode_result: dict | None = None
    chapter_result: dict | None = None
    summary_fallback: dict | None = None
    maintenance_enqueued: bool = False
    fail_reasons: list[str] = Field(default_factory=list)
    trace_handoff: dict | None = None
    warnings: list[str] = Field(default_factory=list)


class PrepareTurnSettings(BaseModel):
    guide_mode: str = "off"
    narrative_stance: str = "balanced"
    apply_mode: str = "shadow"
    takeover_mode: str = "off"
    injection_enabled: bool = True
    input_context_enabled: bool = True
    max_injection_chars: int = 3000
    max_input_context_chars: int = 800
    episode_interval_turns: int = 10
    supervisor_enabled: bool = True
    top_k: int = 5


class PrepareTurnRequest(BaseModel):
    chat_session_id: str
    request_type: str = "model"
    raw_user_input: str = ""
    messages: list[dict] = Field(default_factory=list)
    continuity_trigger_mode: str = "none"
    continuity_query: str = ""
    turn_index: int | None = None
    settings: PrepareTurnSettings = Field(default_factory=PrepareTurnSettings)
    client_meta: dict = Field(default_factory=dict)


class GenerationPacket(BaseModel):
    """O-1a generation packet returned from backend to plugin orchestration.

    Fail-open contract: if the packet is missing, off, or degraded, the
    plugin keeps its local path and chat flow must not stop.
    """
    packet_mode: str = "off"
    effective_user_input: str | None = None
    injection_text: str | None = None
    guidance_metadata: dict = Field(default_factory=dict)
    safety_metadata: dict = Field(default_factory=dict)
    degraded: bool = False
    fallback_reason: str = ""
    trace_summary: dict = Field(default_factory=dict)
    shadow_compare_record: dict = Field(default_factory=dict)


class PrepareTurnResponse(BaseModel):
    """M-1a POST /prepare-turn response contract.

    The skeleton source returns the stable response shape without computed
    bundle fields. Later phases may fill fields incrementally while preserving
    fail-open fallback semantics.
    """
    status: str = "ok"
    source: str = "skeleton"
    chat_session_id: str = ""
    generated_at: str = ""
    effective_user_input: str | None = None
    injection_text: str | None = None
    input_context_text: str | None = None
    supervisor_result: dict | None = None
    narrative_control_summary: dict | None = None
    autonomy_plan: dict | None = None
    progression_ledger: dict | None = None
    micro_beat_proposal: dict | None = None
    scene_step_proposal: dict | None = None
    combined_proposal: dict | None = None
    generation_packet: GenerationPacket | None = None
    writeback_preview: dict | None = None
    trace_preview: dict | None = None
    session_state: dict | None = None
    narrative_control: dict | None = None
    continuity_pack: dict | None = None
    resume_pack: dict | None = None
    canonical_ledger: dict | None = None
    recall_result: dict | None = None
    supervisor_input_pack: dict | None = None
    injection_pack: dict | None = None
    packet_composition: dict | None = None
    long_session_health: dict | None = None
    fallback_reason: str = "skeleton_only"
    warnings: list[str] = Field(default_factory=list)
