from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean
from sqlalchemy.orm import synonym
from sqlalchemy.sql import func

from backend.database import Base

DEFAULT_SESSION_ID = "default"


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, default=DEFAULT_SESSION_ID, index=True)
    turn_index = Column(Integer, nullable=False, index=True)
    role = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class EffectiveInputLog(Base):
    __tablename__ = "effective_input_logs"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, default=DEFAULT_SESSION_ID, index=True)
    turn_index = Column(Integer, nullable=False, index=True)
    effective_input = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, default=DEFAULT_SESSION_ID, index=True)
    turn_index = Column(Integer, nullable=False, index=True)
    summary_json = Column(Text, nullable=True)   # JSON 문자열
    embedding = Column(Text, nullable=True)      # JSON 직렬화된 float 배열
    embedding_model = Column(String(255), nullable=True)  # Sprint 3-C-3: embedding provenance
    importance = Column(Float, nullable=True)
    # Phase 5-2: 감정 기반 importance boost 기록 (debug/trace 용)
    emotional_boost = Column(Float, nullable=True)      # 적용된 boost 값 (None=미적용, 1.0, 2.0)
    # Phase 5-1: Critic 확장 필드
    evidence = Column(Text, nullable=True)              # JSON 직렬화된 evidence_excerpts 배열
    emotional_intensity = Column(Float, nullable=True)   # 0.0~1.0
    narrative_significance = Column(Float, nullable=True) # 0.0~1.0
    library_wing = Column("place_wing", String(255), nullable=True)
    library_room = Column("place_room", String(255), nullable=True)
    archive_wing = synonym("library_wing")
    archive_room = synonym("library_room")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


class DirectEvidenceRecord(Base):
    __tablename__ = "direct_evidence_records"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, default=DEFAULT_SESSION_ID, index=True)
    evidence_kind = Column(String(100), nullable=False, default="fact_event")
    evidence_text = Column(Text, nullable=False)
    source_turn_start = Column(Integer, nullable=False, index=True)
    source_turn_end = Column(Integer, nullable=False, index=True)
    turn_anchor = Column(Integer, nullable=True, index=True)
    source_message_ids_json = Column(Text, nullable=True)
    source_hash = Column(String(255), nullable=True)
    archive_state = Column(String(50), nullable=False, default="pending_capture", index=True)
    capture_stage = Column(String(50), nullable=False, default="critic_extract")
    capture_verification = Column(String(50), nullable=False, default="pending")
    committed_gate = Column(String(50), nullable=True)
    lineage_json = Column(Text, nullable=True)
    repair_needed = Column(Boolean, nullable=False, default=False)
    tombstoned = Column(Boolean, nullable=False, default=False)
    superseded_by_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)


class KGTriple(Base):
    __tablename__ = "kg_triples"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, default=DEFAULT_SESSION_ID, index=True)
    subject = Column(String(255), nullable=False)
    predicate = Column(String(255), nullable=False)
    object = Column(String(255), nullable=False)
    valid_from = Column(Integer, nullable=True)   # turn_index 기준
    valid_to = Column(Integer, nullable=True)     # null = 현재 유효
    source_turn = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)


# Sprint 3-E-3: Audit Trail
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    chat_session_id = Column(String(255), nullable=True, index=True)
    target_type = Column(String(100), nullable=True)  # memory, kg_triple, session, etc.
    target_id = Column(Integer, nullable=True)
    summary = Column(Text, nullable=True)
    details_json = Column(Text, nullable=True)  # JSON 문자열
    source = Column(String(50), nullable=True, default="api")  # manual_ui, auto_rollback, api, batch


# Sprint 3-E-4: Critic Feedback
class CriticFeedback(Base):
    __tablename__ = "critic_feedback"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    chat_session_id = Column(String(255), nullable=False, index=True)
    target_type = Column(String(50), nullable=False)  # memory, kg_triple
    target_id = Column(Integer, nullable=False)
    feedback_value = Column(String(20), nullable=False)  # up, down
    feedback_note = Column(Text, nullable=True)
    source = Column(String(50), nullable=True, default="manual_ui")


# Phase 2-1: Active State
class ActiveState(Base):
    __tablename__ = "active_states"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, index=True)
    state_type = Column(String(100), nullable=False, index=True)  # scene_state, relationship_state, unresolved_threads
    content = Column(Text, nullable=False)  # JSON 문자열
    turn_index = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)


# HS-1a: Canonical State Layer (parallel to active_states)
class CanonicalStateLayer(Base):
    __tablename__ = "canonical_state_layers"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, index=True)
    layer_type = Column(String(100), nullable=False, index=True)  # scene_state, relationship_state, unresolved_threads
    content = Column(Text, nullable=False)  # JSON 문자열
    source_state_type = Column(String(100), nullable=True)
    turn_index = Column(Integer, nullable=False, index=True)
    source_turn = Column(Integer, nullable=True)
    source_record = Column(Integer, nullable=True)
    last_verified_turn = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)


# Phase 3-1: Episode Summary
class EpisodeSummary(Base):
    __tablename__ = "episode_summaries"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, index=True)
    from_turn = Column(Integer, nullable=False)
    to_turn = Column(Integer, nullable=False)
    summary_text = Column(Text, nullable=False)
    key_entities = Column(Text, nullable=True)       # JSON 문자열
    key_events = Column(Text, nullable=True)         # JSON 문자열
    open_loops_json = Column(Text, nullable=True)    # JSON 문자열
    relationship_changes_json = Column(Text, nullable=True)  # JSON 문자열
    embedding_vector = Column(Text, nullable=True)   # JSON 직렬화된 float 배열
    embedding_model = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)


# Step 9 P-1a: Chapter Summary
class ChapterSummary(Base):
    __tablename__ = "chapter_summaries"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, index=True)
    from_turn = Column(Integer, nullable=False)
    to_turn = Column(Integer, nullable=False)
    chapter_index = Column(Integer, nullable=True)
    chapter_title = Column(String(255), nullable=True)
    summary_text = Column(Text, nullable=False)
    open_loops_json = Column(Text, nullable=True)          # JSON 문자열
    relationship_changes_json = Column(Text, nullable=True)  # JSON 문자열
    world_changes_json = Column(Text, nullable=True)       # JSON 문자열
    callback_candidates_json = Column(Text, nullable=True) # JSON 문자열
    resume_text = Column(Text, nullable=True)
    embedding_vector = Column(Text, nullable=True)         # JSON 직렬화된 float 배열
    embedding_model = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)


# Step 9 P-3a: Arc Summary
class ArcSummary(Base):
    __tablename__ = "arc_summaries"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, index=True)
    from_turn = Column(Integer, nullable=False)
    to_turn = Column(Integer, nullable=False)
    arc_index = Column(Integer, nullable=True)
    arc_name = Column(String(255), nullable=True)
    arc_status = Column(String(50), nullable=False, default="active")
    core_conflict = Column(Text, nullable=True)
    key_turning_points_json = Column(Text, nullable=True)      # JSON 문자열
    active_promises_json = Column(Text, nullable=True)         # JSON 문자열
    unresolved_debts_json = Column(Text, nullable=True)        # JSON 문자열
    resolved_payoffs_json = Column(Text, nullable=True)        # JSON 문자열
    callback_candidates_json = Column(Text, nullable=True)     # JSON 문자열
    irreversible_turns_json = Column(Text, nullable=True)      # JSON 문자열
    callback_debts_json = Column(Text, nullable=True)          # JSON 문자열
    relationship_pivots_json = Column(Text, nullable=True)     # JSON 문자열
    future_payoff_candidates_json = Column(Text, nullable=True)  # JSON 문자열
    arc_resume_text = Column(Text, nullable=True)
    embedding_vector = Column(Text, nullable=True)             # JSON 직렬화된 float 배열
    embedding_model = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)


# Step 9 P-4a: Saga Digest
class SagaDigest(Base):
    __tablename__ = "saga_digests"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, index=True)
    from_turn = Column(Integer, nullable=False)
    to_turn = Column(Integer, nullable=False)
    era_label = Column(String(255), nullable=True)
    saga_summary = Column(Text, nullable=False)
    persistent_facts_json = Column(Text, nullable=True)        # JSON 문자열
    never_drop_candidates_json = Column(Text, nullable=True)   # JSON 문자열
    resume_pack_text = Column(Text, nullable=True)
    embedding_vector = Column(Text, nullable=True)             # JSON 직렬화된 float 배열
    embedding_model = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)


# ------------------------------------------------------------------ #
#  Phase E-1: Storyline Registry
# ------------------------------------------------------------------ #

class Storyline(Base):
    __tablename__ = "storylines"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, default=DEFAULT_SESSION_ID, index=True)
    name = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="active")  # active, paused, resolved
    entities_json = Column(Text, nullable=True)           # JSON: 참여 캐릭터 배열
    current_context = Column(Text, nullable=True)         # 현재 스토리라인 상황 요약
    key_points_json = Column(Text, nullable=True)         # JSON: 핵심 사건 배열
    ongoing_tensions_json = Column(Text, nullable=True)   # JSON: 진행 중 긴장 배열
    confidence = Column(Float, nullable=True)             # H-2a: storyline 품질 신뢰도 (0.0~1.0)
    evidence_count = Column(Integer, nullable=True)       # H-2a: 누적 evidence 개수
    last_evidence_turn = Column(Integer, nullable=True)   # H-2a: 마지막 evidence가 관측된 턴
    first_turn = Column(Integer, nullable=True)
    last_turn = Column(Integer, nullable=True)
    # I-3a: Memory Trust Controls
    pinned = Column(Boolean, nullable=False, default=False)           # injection budget 경쟁 시 우선 유지
    suppressed = Column(Boolean, nullable=False, default=False)       # injection에서 완전 제외
    user_corrected = Column(Boolean, nullable=False, default=False)   # 사용자 직접 수정 항목 (AI 자동 덧쓰기 제어)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


# ------------------------------------------------------------------ #
#  Phase E-2: Character State Tracker
# ------------------------------------------------------------------ #

class CharacterState(Base):
    __tablename__ = "character_states"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, default=DEFAULT_SESSION_ID, index=True)
    character_name = Column(String(255), nullable=False, index=True)
    appearance_json = Column(Text, nullable=True)       # JSON: 외형 정보
    personality_json = Column(Text, nullable=True)      # JSON: 성격 특성
    status_json = Column(Text, nullable=True)           # JSON: 위치, 감정, 건강 등
    relationships_json = Column(Text, nullable=True)    # JSON: 대상별 sentiment, affection, tension
    speech_style_json = Column(Text, nullable=True)     # JSON: 말투 (E-3에서 활용)
    turn_index = Column(Integer, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


class CharacterEvent(Base):
    __tablename__ = "character_events"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, default=DEFAULT_SESSION_ID, index=True)
    character_name = Column(String(255), nullable=False, index=True)
    turn_index = Column(Integer, nullable=True)
    event_type = Column(String(100), nullable=False)    # relationship_shift, personality_change, status_change, appearance_change
    details_json = Column(Text, nullable=True)          # JSON: 이벤트 상세
    created_at = Column(DateTime, server_default=func.now(), nullable=False, index=True)


# ------------------------------------------------------------------ #
# Phase E-4 — World Rules Registry
# ------------------------------------------------------------------ #

class WorldRule(Base):
    __tablename__ = "world_rules"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, default=DEFAULT_SESSION_ID, index=True)
    scope = Column(String(255), nullable=False, default="root")        # root / region / location / faction / system
    # I-4a: scope 인스턴스 이름 (root는 None, 나머지는 e.g. "Northern Highlands")
    scope_name = Column(String(500), nullable=True)
    category = Column(String(100), nullable=False, default="custom")   # exists / systems / physics / custom
    key = Column(String(500), nullable=False)
    value_json = Column(Text, nullable=True)            # JSON: 규칙 상세
    genre = Column(String(100), nullable=True)          # 장르 힌트
    source_turn = Column(Integer, nullable=True)        # 최초 감지 턴
    # I-3a: Memory Trust Controls
    pinned = Column(Boolean, nullable=False, default=False)
    suppressed = Column(Boolean, nullable=False, default=False)
    user_corrected = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


# ------------------------------------------------------------------ #
#  H-5: Continuity Hook (Relationship + Promise Tracker)
# ------------------------------------------------------------------ #

class PendingThread(Base):
    """
    약속, 미해결 목표, 열린 질문, 감정 빚 등 서사에서 사라지면 안 되는 open loop를 추적한다.
    MO의 기존 session-scoped row 패턴과 character_events / storylines 구조에 맞춰 설계한 테이블이다.
    """
    __tablename__ = "pending_threads"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, default=DEFAULT_SESSION_ID, index=True)

    # hook 유형: promise / unresolved_goal / open_question / risk / emotional_debt
    thread_type = Column(String(50), nullable=False, index=True)

    # 한 줄 제목 — 검색 및 injection 요약용
    title = Column(String(500), nullable=False)

    # 상태: open / paused / resolved
    status = Column(String(20), nullable=False, default="open", index=True)

    # 약속/목표의 주체 캐릭터 (없으면 null)
    owner = Column(String(255), nullable=True)

    # 약속/목표의 대상 캐릭터 또는 사물 (없으면 null)
    target = Column(String(255), nullable=True)

    # hook이 처음 등장한 턴
    source_turn = Column(Integer, nullable=True)

    # 가장 최근 이 hook이 대화에서 관측된 턴
    last_seen_turn = Column(Integer, nullable=True)

    # 0.0~1.0 신뢰도 — critic이 추출했을 때 얼마나 명시적이었는지
    confidence = Column(Float, nullable=True)

    # 상세 내용 (JSON 문자열) — 추가 컨텍스트, 조건, 부가 정보
    details_json = Column(Text, nullable=True)

    # resolved 시 어떻게 해결되었는지 한 줄 메모
    resolution_note = Column(Text, nullable=True)

    # I-3a: Memory Trust Controls
    pinned = Column(Boolean, nullable=False, default=False)
    suppressed = Column(Boolean, nullable=False, default=False)
    user_corrected = Column(Boolean, nullable=False, default=False)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


# ------------------------------------------------------------------ #
#  I-4c: Session Active Scope Cache
# ------------------------------------------------------------------ #

class SessionActiveScopeCache(Base):
    """세션별 현재 활성 scope를 저장한다.

    supervisor가 장면 전환을 감지하거나 사용자가 명시적으로 지정할 때 갱신된다.
    GET /world-rules/{id}/inherited 경로의 기본값으로 사용된다.
    """
    __tablename__ = "session_active_scopes"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, unique=True, index=True)
    # 현재 활성 scope level: root / region / location / faction / system
    active_scope = Column(String(50), nullable=False, default="root")
    # 해당 scope의 인스턴스 이름 (root는 None)
    scope_name = Column(String(500), nullable=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


# ------------------------------------------------------------------ #
#  K-2a: Guidance Plan State (Step 7 — Persistent Story Plan)
# ------------------------------------------------------------------ #

class GuidancePlanState(Base):
    """
    session-scoped persistent guidance state.
    GET /narrative-control/ 가 매 턴 heuristic 재조립하는 것을 방지하기 위해
    story_plan / director 결과를 캐시하고 conservative update 규칙으로 갱신한다.

    state_status:
      "empty"     — 아직 초기화되지 않음
      "heuristic" — DB 자산(storylines/hooks/world_rules)에서 heuristic 조립
      "llm_updated" — K-4 이후 LLM 보강 (현재는 미사용)

    freshness_policy:
      last_turn 기준으로 FRESHNESS_TURNS 이내이면 cached 상태 재사용.
      이후에는 heuristic re-evaluate를 실행해 conservative update.
    """
    __tablename__ = "guidance_plan_states"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, unique=True, index=True)

    # StoryPlanState JSON 직렬화 (current_arc, narrative_goal, active_tensions, next_beats, ...)
    story_plan_json = Column(Text, nullable=True)

    # DirectorState JSON 직렬화 (scene_mandate, required_outcomes, forbidden_moves, ...)
    director_json = Column(Text, nullable=True)

    # 상태 수준: "empty" | "heuristic" | "llm_updated"
    state_status = Column(String(50), nullable=False, default="empty")

    # 이 state가 빌드/갱신된 시점의 턴 인덱스 (-1 = 미기록)
    last_turn = Column(Integer, nullable=False, default=-1)

    # 경고 메시지 배열 JSON
    warnings_json = Column(Text, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


# ------------------------------------------------------------------ #
#  K-5a: Guidance Compact Record (Step 7 — Resolution / Compaction)
# ------------------------------------------------------------------ #

class GuidanceCompactRecord(Base):
    """
    해결/소멸된 guidance 항목의 compact history 레코드.

    resolved arc, resolved required_outcome, expired forbidden_move,
    resolved checklist item을 즉시 삭제하거나 무한 유지하는 대신
    이 테이블로 이동해 축약된 형태로 보존한다.

    record_type:
      "resolved_arc"       - 완료/해결된 arc (current_arc, storyline 등)
      "resolved_outcome"   - director.resolved_outcomes 에서 이동된 항목
      "expired_forbidden"  - director.expired_forbidden 에서 이동된 항목
      "resolved_checklist" - director.execution_checklist 완료 항목

    origin_type:
      "storyline"          - Storyline 테이블 행
      "director_outcome"   - GuidancePlanState.director_json 내 required_outcomes
      "director_forbidden" - GuidancePlanState.director_json 내 forbidden_moves
      "checklist"          - GuidancePlanState.director_json 내 execution_checklist
      "hook"               - PendingThread 테이블 행

    emotional_weight (K-5f):
      None  = 미측정
      0.0   = 감정 중요도 낮음 (빠른 압축 허용)
      1.0   = 감정 중요도 높음 (더 오래 유지)
      supervisor 응답의 pressure_level / active_tensions를 proxy로 사용하는
      heuristic 값이며, LLM 기반 체크는 추후 추가한다.
    """
    __tablename__ = "guidance_compact_records"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, index=True)

    # 레코드 유형 (위 docstring 참조)
    record_type = Column(String(50), nullable=False, index=True)

    # 한 줄 제목 — trace / continuity 조회용
    title = Column(String(500), nullable=False)

    # 연속성 참조용 압축 텍스트 (prompt에 직접 삽입할 수 있는 한 줄 요약)
    compact_summary = Column(Text, nullable=True)

    # 원본 내용 JSON (compaction 이전 상태 보존, debug 및 복구용)
    original_content_json = Column(Text, nullable=True)

    # 원본 엔티티 유형 (위 docstring 참조)
    origin_type = Column(String(50), nullable=True)

    # 원본 엔티티 ID (Storyline.id, PendingThread.id 등; director 내부 항목은 null)
    origin_id = Column(Integer, nullable=True)

    # K-5f: 감정 강도 기반 중요도 (0.0~1.0, None=미측정)
    # pressure_level/active_tensions 기반 heuristic으로 초기화
    emotional_weight = Column(Float, nullable=True)

    # 이 항목이 처음 active 상태였던 턴
    source_turn = Column(Integer, nullable=True)

    # compact 처리된 턴 (active -> compact 전환 시점)
    compacted_turn = Column(Integer, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)


# ------------------------------------------------------------------ #
#  L-1a: Maintenance Pass State (Step 7 — Turn Maintenance Pass)
# ------------------------------------------------------------------ #

class MaintenancePassState(Base):
    """
    매 턴 onAfterRequest 완료 직후에 실행되는 maintenance pass의 결과를 저장한다.

    shadow_only=True 동안은 guidance state를 직접 변경하지 않고,
    관찰(observations)과 drift_signals만 기록한다.
    L-1d에서 shadow_only=False로 전환되면 결과가 GuidancePlanState 갱신에 반영된다.

    status:
      "ok"       — maintenance pass 완료 (shadow 또는 applied)
      "skipped"  — 실행 조건 미충족 (no state, 너무 최근 실행 등)
      "error"    — pass 중 예외 발생 (non-fatal)

    shadow_only:
      True  — 관찰만, guidance state 미변경 (L-1a/L-1b 기본값)
      False — 결과를 guidance state에 반영 (L-1d 이후)
    """
    __tablename__ = "maintenance_pass_states"

    id = Column(Integer, primary_key=True, index=True)
    chat_session_id = Column(String(255), nullable=False, index=True)

    # 이 maintenance pass를 트리거한 턴 인덱스
    turn_index = Column(Integer, nullable=False, default=-1)

    # pass 결과 상태
    status = Column(String(50), nullable=False, default="ok")

    # shadow-only 여부 (True = 관찰만, False = 실제 적용)
    shadow_only = Column(Boolean, nullable=False, default=True)

    # 관찰 내용 JSON (list[str] — 읽기 좋은 한 줄 요약들)
    observations_json = Column(Text, nullable=True)

    # 탐지된 drift signal JSON (list[dict]) — L-2에서 채워짐
    drift_signals_json = Column(Text, nullable=True)

    # 갱신 제안 JSON (list[str]) — L-1c에서 채워짐
    refresh_suggestions_json = Column(Text, nullable=True)

    # story plan이 이 pass에서 갱신되었는지 (shadow=True 시 항상 False)
    story_plan_refreshed = Column(Boolean, nullable=False, default=False)

    # director가 이 pass에서 갱신되었는지 (shadow=True 시 항상 False)
    director_refreshed = Column(Boolean, nullable=False, default=False)

    executed_at = Column(DateTime, server_default=func.now(), nullable=False)

