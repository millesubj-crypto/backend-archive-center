"""Session service module.

Priority 4-d-2: Session export and compare routes extraction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def export_session(
    palace,
    Utf8JSONResponse,
    append_audit_log,
    chat_session_id: str,
):
    """Export a single session as JSON (Explorer UI)."""
    try:
        sid = (chat_session_id or "").strip()
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required."},
                status_code=400,
            )

        result = palace.export_session(sid)

        if result["ok"]:
            append_audit_log(
                event_type="session_export",
                chat_session_id=sid,
                target_type="session",
                summary=f"Exported session {sid}",
                source="manual_ui",
            )
            return {"status": "ok", **result["data"]}
        else:
            return Utf8JSONResponse(
                content={"status": "error", "detail": result["detail"]},
                status_code=404,
            )
    except Exception as exc:
        logger.error("GET /sessions/%s/export failed: %s", chat_session_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )


def compare_sessions(
    SessionLocal,
    func,
    ChatLog,
    Memory,
    KGTriple,
    AuditLog,
    CriticFeedback,
    to_kst,
    Utf8JSONResponse,
    session_ids: str = "",
    preview_limit: int = 10,
):
    """Compare 2 sessions side by side (Explorer UI)."""
    try:
        ids = [s.strip() for s in (session_ids or "").split(",") if s.strip()]
        if len(ids) < 2:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "At least 2 session_ids are required."},
                status_code=400,
            )
        # Safety cap: max 2 sessions
        ids = ids[:2]
        plimit = max(1, min(preview_limit, 30))

        db = SessionLocal()
        try:
            result = {}
            for sid in ids:
                # counts
                logs_count = db.query(func.count(ChatLog.id)).filter(ChatLog.chat_session_id == sid).scalar() or 0
                mem_count = db.query(func.count(Memory.id)).filter(Memory.chat_session_id == sid).scalar() or 0
                kg_count = db.query(func.count(KGTriple.id)).filter(KGTriple.chat_session_id == sid).scalar() or 0
                audit_count = db.query(func.count(AuditLog.id)).filter(AuditLog.chat_session_id == sid).scalar() or 0
                fb_up = db.query(func.count(CriticFeedback.id)).filter(
                    CriticFeedback.chat_session_id == sid, CriticFeedback.feedback_value == "up"
                ).scalar() or 0
                fb_down = db.query(func.count(CriticFeedback.id)).filter(
                    CriticFeedback.chat_session_id == sid, CriticFeedback.feedback_value == "down"
                ).scalar() or 0

                last_log = db.query(func.max(ChatLog.created_at)).filter(ChatLog.chat_session_id == sid).scalar()

                # recent chat_logs
                recent_logs = (
                    db.query(ChatLog)
                    .filter(ChatLog.chat_session_id == sid)
                    .order_by(ChatLog.turn_index.desc(), ChatLog.id.desc())
                    .limit(plimit * 2)  # user+assistant pairs
                    .all()
                )
                logs_preview = [{
                    "id": r.id, "turn_index": r.turn_index, "role": r.role,
                    "content": r.content[:200] if r.content else "",
                    "created_at": to_kst(r.created_at),
                } for r in reversed(recent_logs)]

                # recent memories
                recent_mems = (
                    db.query(Memory)
                    .filter(Memory.chat_session_id == sid)
                    .order_by(Memory.id.desc())
                    .limit(plimit)
                    .all()
                )
                mem_preview = [{
                    "id": r.id,
                    "summary_json": r.summary_json[:200] if r.summary_json else "",
                    "importance": r.importance,
                    "created_at": to_kst(r.created_at),
                } for r in recent_mems]

                # recent kg_triples
                recent_kg = (
                    db.query(KGTriple)
                    .filter(KGTriple.chat_session_id == sid)
                    .order_by(KGTriple.id.desc())
                    .limit(plimit)
                    .all()
                )
                kg_preview = [{
                    "id": r.id,
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "created_at": to_kst(r.created_at),
                } for r in recent_kg]

                result[sid] = {
                    "counts": {
                        "chat_logs": logs_count,
                        "memories": mem_count,
                        "kg_triples": kg_count,
                        "audit_logs": audit_count,
                        "feedback_up": fb_up,
                        "feedback_down": fb_down,
                    },
                    "last_activity": to_kst(last_log) if last_log else None,
                    "logs_preview": logs_preview,
                    "memories_preview": mem_preview,
                    "kg_triples": kg_preview,
                }

            return {"status": "ok", "sessions": result}
        finally:
            db.close()
    except Exception as exc:
        logger.error("GET /sessions/compare failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )


# --------------------------------------------------------------------------- #
# Response models for guidance snapshot
# --------------------------------------------------------------------------- #

class GuidanceSnapshotCompactRecord(BaseModel):
    """Guidance compact record model (used in snapshot inclusion)."""
    record_type: str
    title: str
    compact_summary: str | None = None
    origin_type: str | None = None
    source_turn: int | None = None
    compacted_turn: int | None = None
    emotional_weight: float | None = None


class GuidanceSnapshotMaintenanceLast(BaseModel):
    """Last maintenance pass result model (used in snapshot inclusion)."""
    turn_index: int
    status: str
    shadow_only: bool
    observations_count: int
    drift_signals_count: int
    executed_at: str | None = None


class GuidanceSnapshotResponse(BaseModel):
    """L-3a: guidance-inclusive snapshot contract.

    GET /sessions/{chat_session_id}/guidance-snapshot response structure.
    """
    status: str = "ok"
    chat_session_id: str
    state_status: str = "no_state"
    last_turn: int = -1
    story_plan: dict = Field(default_factory=dict)
    director: dict = Field(default_factory=dict)
    compact_records: list[GuidanceSnapshotCompactRecord] = Field(default_factory=list)
    maintenance_last: GuidanceSnapshotMaintenanceLast | None = None
    generated_at: str = ""
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Response models for step7 health
# --------------------------------------------------------------------------- #

class Step7HealthGuidanceState(BaseModel):
    """Guidance state summary and arc persistence metrics."""
    status: str = "no_state"
    last_built_turn: int = -1
    arc_age_turns: int = 0
    active_tensions: int = 0
    next_beats: int = 0
    open_required: int = 0
    forbidden_count: int = 0


class Step7HealthDriftSummary(BaseModel):
    """Drift signal summary for last N maintenance passes."""
    passes_analyzed: int = 0
    total_signals: int = 0
    high_severity: int = 0
    by_type: dict = Field(default_factory=dict)


class Step7HealthCompactionSummary(BaseModel):
    """Compact record summary and compaction transition."""
    total_records: int = 0
    by_type: dict = Field(default_factory=dict)
    avg_emotional_weight: float = 0.0
    latest_compaction_turn: int = -1


class Step7HealthMaintenanceSummary(BaseModel):
    """Maintenance pass summary and suggestion count."""
    total_passes: int = 0
    ok_count: int = 0
    error_count: int = 0
    ok_rate: float = 0.0
    last_suggestions: list[str] = Field(default_factory=list)


class Step7HealthRegressionChecks(BaseModel):
    """Regression checklist pass/warn/fail determination.

    Criteria:
      guidance_persistence : pass if last_built_turn >= 0 and status not in (no_state, empty)
                             warn if status == empty
                             fail if status == no_state
      arc_stability        : pass if arc_age_turns >= 3
                             warn if arc_age_turns >= 1
                             else fail (arc not persisted before guidance build)
      compaction_health    : pass if total_records >= 1
                             warn if total_records == 0 and guidance != no_state
                             else skip (session too short, compaction not started)
      maintenance_effect   : pass if ok_rate >= 0.8
                             warn if ok_rate >= 0.5
                             skip if total_passes == 0
                             else fail
    """
    guidance_persistence: str = "skip"
    arc_stability: str = "skip"
    compaction_health: str = "skip"
    maintenance_effect: str = "skip"
    notes: list[str] = Field(default_factory=list)


class Step7HealthResponse(BaseModel):
    """L-5a: Step 7 long-session health probe response contract."""
    status: str = "ok"
    chat_session_id: str
    total_turns: int = 0
    guidance_state: Step7HealthGuidanceState = Field(default_factory=Step7HealthGuidanceState)
    drift_summary: Step7HealthDriftSummary = Field(default_factory=Step7HealthDriftSummary)
    compaction_summary: Step7HealthCompactionSummary = Field(default_factory=Step7HealthCompactionSummary)
    maintenance_summary: Step7HealthMaintenanceSummary = Field(default_factory=Step7HealthMaintenanceSummary)
    regression_checks: Step7HealthRegressionChecks = Field(default_factory=Step7HealthRegressionChecks)
    generated_at: str = ""
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _build_step7_regression_checks(
    gs: Step7HealthGuidanceState,
    ds: Step7HealthDriftSummary,
    cs: Step7HealthCompactionSummary,
    ms: Step7HealthMaintenanceSummary,
) -> Step7HealthRegressionChecks:
    """Regression check determination logic (pure function, no DB access)."""
    notes: list[str] = []

    # guidance_persistence
    if gs.status == "no_state":
        gp = "fail"
        notes.append("guidance state is missing - check NC endpoint call")
    elif gs.status == "empty":
        gp = "warn"
        notes.append("guidance state is empty - rebuild scheduled on next NC call")
    elif gs.last_built_turn >= 0:
        gp = "pass"
    else:
        gp = "warn"
        notes.append("guidance state exists but last built turn is unknown")

    # arc_stability
    if gs.arc_age_turns >= 3:
        arc = "pass"
    elif gs.arc_age_turns >= 1:
        arc = "warn"
        notes.append(f"arc_age_turns={gs.arc_age_turns} - check for 3+ turns stability")
    elif gs.status == "no_state":
        arc = "skip"
    else:
        arc = "fail"
        notes.append("arc_age_turns=0 - arc not persisted before guidance build")

    # compaction_health
    if cs.total_records >= 1:
        comp = "pass"
    elif gs.status not in ("no_state", "empty"):
        comp = "warn"
        notes.append("compact record count is 0 - session may be unresolved")
    else:
        comp = "skip"

    # maintenance_effect
    if ms.total_passes == 0:
        maint = "skip"
    elif ms.ok_rate >= 0.8:
        maint = "pass"
    elif ms.ok_rate >= 0.5:
        maint = "warn"
        notes.append(f"maintenance ok_rate={ms.ok_rate:.2f} - below pass threshold")
    else:
        maint = "fail"
        notes.append(f"maintenance ok_rate={ms.ok_rate:.2f} - maintenance intervention needed")

    return Step7HealthRegressionChecks(
        guidance_persistence=gp,
        arc_stability=arc,
        compaction_health=comp,
        maintenance_effect=maint,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Handlers for guidance snapshot and step7 health
# --------------------------------------------------------------------------- #

def get_guidance_snapshot(
    SessionLocal,
    GuidancePlanState,
    GuidanceCompactRecord,
    MaintenancePassState,
    GuidanceSnapshotResponse,
    GuidanceSnapshotCompactRecord,
    GuidanceSnapshotMaintenanceLast,
    Utf8JSONResponse,
    chat_session_id: str,
):
    """Return session guidance state as a portable snapshot (Explorer UI).

    L-3a: read-only. Does not modify guidance state.
    Includes: GuidancePlanState (story_plan + director), GuidanceCompactRecord
    last 10 items, and the most recent MaintenancePassState summary.
    """
    import json as _json

    sid = (chat_session_id or "").strip()
    if not sid:
        return Utf8JSONResponse(
            content={"status": "error", "detail": "chat_session_id is required"},
            status_code=400,
        )

    generated_at = datetime.now(timezone.utc).isoformat()
    warnings: list[str] = []

    db = SessionLocal()
    try:
        # Step 1. GuidancePlanState query
        gps = (
            db.query(GuidancePlanState)
            .filter(GuidancePlanState.chat_session_id == sid)
            .first()
        )

        if gps is None:
            return GuidanceSnapshotResponse(
                status="ok",
                chat_session_id=sid,
                state_status="no_state",
                last_turn=-1,
                generated_at=generated_at,
                warnings=["GuidancePlanState not found for session"],
            )

        # story_plan parse
        story_plan: dict = {}
        if gps.story_plan_json:
            try:
                story_plan = _json.loads(gps.story_plan_json)
            except Exception as exc:
                warnings.append(f"story_plan_json parse error: {exc}")

        # director parse
        director: dict = {}
        if gps.director_json:
            try:
                director = _json.loads(gps.director_json)
            except Exception as exc:
                warnings.append(f"director_json parse error: {exc}")

        # Step 2. Last 10 GuidanceCompactRecord items
        compact_rows = (
            db.query(GuidanceCompactRecord)
            .filter(GuidanceCompactRecord.chat_session_id == sid)
            .order_by(
                GuidanceCompactRecord.compacted_turn.desc().nullslast(),
                GuidanceCompactRecord.id.desc(),
            )
            .limit(10)
            .all()
        )
        compact_records: list[GuidanceSnapshotCompactRecord] = []
        for row in compact_rows:
            compact_records.append(GuidanceSnapshotCompactRecord(
                record_type=row.record_type or "",
                title=row.title or "",
                compact_summary=row.compact_summary,
                origin_type=row.origin_type,
                source_turn=row.source_turn,
                compacted_turn=row.compacted_turn,
                emotional_weight=row.emotional_weight,
            ))

        # Step 3. Most recent MaintenancePassState
        mnt_row = (
            db.query(MaintenancePassState)
            .filter(MaintenancePassState.chat_session_id == sid)
            .order_by(MaintenancePassState.turn_index.desc(), MaintenancePassState.id.desc())
            .first()
        )
        maintenance_last: GuidanceSnapshotMaintenanceLast | None = None
        if mnt_row:
            obs_list: list = []
            drift_list: list = []
            executed_at_str: str | None = None
            try:
                obs_list = _json.loads(mnt_row.observations_json or "[]")
                if not isinstance(obs_list, list):
                    obs_list = []
            except Exception as exc:
                warnings.append(f"MaintenancePassState observations parse error: {exc}")
            try:
                drift_list = _json.loads(mnt_row.drift_signals_json or "[]")
                if not isinstance(drift_list, list):
                    drift_list = []
            except Exception as exc:
                warnings.append(f"MaintenancePassState drift_signals parse error: {exc}")
            try:
                executed_at_str = (
                    mnt_row.created_at.isoformat() if mnt_row.created_at else None
                )
            except Exception:
                executed_at_str = None
            maintenance_last = GuidanceSnapshotMaintenanceLast(
                turn_index=mnt_row.turn_index,
                status=mnt_row.status or "ok",
                shadow_only=bool(mnt_row.shadow_only),
                observations_count=len(obs_list),
                drift_signals_count=len(drift_list),
                executed_at=executed_at_str,
            )

        # L-3d: state_status="empty" rebuild pending flag
        effective_status = gps.state_status or "empty"
        if effective_status == "empty":
            warnings.append(
                "guidance state is empty - rebuild will be triggered by next GET /narrative-control call"
            )

        return GuidanceSnapshotResponse(
            status="ok",
            chat_session_id=sid,
            state_status=effective_status,
            last_turn=gps.last_turn if gps.last_turn is not None else -1,
            story_plan=story_plan,
            director=director,
            compact_records=compact_records,
            maintenance_last=maintenance_last,
            generated_at=generated_at,
            warnings=warnings,
        )

    except Exception as exc:
        logger.error("GET /sessions/%s/guidance-snapshot failed: %s", chat_session_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )
    finally:
        db.close()


def get_step7_health(
    SessionLocal,
    ChatLog,
    GuidancePlanState,
    GuidanceCompactRecord,
    MaintenancePassState,
    Step7HealthResponse,
    Step7HealthGuidanceState,
    Step7HealthDriftSummary,
    Step7HealthCompactionSummary,
    Step7HealthMaintenanceSummary,
    Step7HealthRegressionChecks,
    Utf8JSONResponse,
    chat_session_id: str,
    passes: int = 10,
):
    """L-5a: Step 7 long-session health probe.

    Parameters:
      passes  - number of recent maintenance passes to analyze (default 10, min 1, max 50)
    """
    import json as _json

    db = SessionLocal()
    try:
        sid = (chat_session_id or "").strip()
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required"},
                status_code=400,
            )

        passes = max(1, min(passes, 50))
        generated_at = datetime.now(timezone.utc).isoformat()
        warnings: list[str] = []

        # 1) total_turns
        total_turns: int = db.query(ChatLog).filter(
            ChatLog.chat_session_id == sid
        ).count()

        # 2) guidance_state
        gps = db.query(GuidancePlanState).filter(GuidancePlanState.chat_session_id == sid).first()
        gs = Step7HealthGuidanceState()
        if gps:
            gs.status = gps.state_status or "empty"
            gs.last_built_turn = gps.last_turn if gps.last_turn is not None else -1
            if gs.last_built_turn >= 0 and total_turns > 0:
                gs.arc_age_turns = max(0, total_turns - gs.last_built_turn)
            try:
                sp = _json.loads(gps.story_plan_json or "{}") if gps.story_plan_json else {}
                if isinstance(sp, dict):
                    gs.active_tensions = len(sp.get("active_tensions") or [])
                    gs.next_beats = len(sp.get("next_beats") or [])
            except Exception:
                warnings.append("story_plan_json parse failed - guidance_state partial read")
            try:
                dr = _json.loads(gps.director_json or "{}") if gps.director_json else {}
                if isinstance(dr, dict):
                    gs.open_required = len(dr.get("required_outcomes") or [])
                    gs.forbidden_count = len(dr.get("forbidden_moves") or [])
            except Exception:
                warnings.append("director_json parse failed - guidance_state partial read")

        # 3) compaction_summary
        records = db.query(GuidanceCompactRecord).filter(GuidanceCompactRecord.chat_session_id == sid).all()
        cs = Step7HealthCompactionSummary()
        cs.total_records = len(records)
        if records:
            by_type: dict[str, int] = {}
            weights: list[float] = []
            latest_ct = -1
            for r in records:
                by_type[r.record_type] = by_type.get(r.record_type, 0) + 1
                if r.emotional_weight is not None:
                    weights.append(float(r.emotional_weight))
                ct = r.compacted_turn if r.compacted_turn is not None else -1
                if ct > latest_ct:
                    latest_ct = ct
            cs.by_type = by_type
            cs.avg_emotional_weight = round(sum(weights) / len(weights), 3) if weights else 0.0
            cs.latest_compaction_turn = latest_ct

        # 4) drift_summary + maintenance_summary
        mps_rows = (
            db.query(MaintenancePassState)
            .filter(MaintenancePassState.chat_session_id == sid)
            .order_by(MaintenancePassState.turn_index.desc())
            .limit(passes)
            .all()
        )
        ds = Step7HealthDriftSummary()
        ms = Step7HealthMaintenanceSummary()
        if mps_rows:
            ds.passes_analyzed = len(mps_rows)
            ms.total_passes = len(mps_rows)
            by_signal: dict[str, int] = {}
            for row in mps_rows:
                if row.status == "ok":
                    ms.ok_count += 1
                elif row.status == "error":
                    ms.error_count += 1
                try:
                    sigs = _json.loads(row.drift_signals_json or "[]")
                    if isinstance(sigs, list):
                        for sig in sigs:
                            if isinstance(sig, dict):
                                stype = sig.get("signal_type", "unknown")
                                by_signal[stype] = by_signal.get(stype, 0) + 1
                                ds.total_signals += 1
                                if sig.get("severity") == "high":
                                    ds.high_severity += 1
                except Exception:
                    pass
            ds.by_type = by_signal
            ms.ok_rate = round(ms.ok_count / ms.total_passes, 3)
            latest_mps = mps_rows[0]
            try:
                ms.last_suggestions = _json.loads(latest_mps.refresh_suggestions_json or "[]") or []
            except Exception:
                pass

        # 5) regression_checks
        rc = _build_step7_regression_checks(gs, ds, cs, ms)

        return Step7HealthResponse(
            status="ok",
            chat_session_id=sid,
            total_turns=total_turns,
            guidance_state=gs,
            drift_summary=ds,
            compaction_summary=cs,
            maintenance_summary=ms,
            regression_checks=rc,
            generated_at=generated_at,
            warnings=warnings,
        )

    except Exception as exc:
        logger.error("GET /sessions/%s/step7-health failed: %s", chat_session_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )
    finally:
        db.close()

