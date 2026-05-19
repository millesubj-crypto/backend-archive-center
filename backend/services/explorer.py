"""Explorer/Admin/Session service module.

Priority 4-a: Explorer read routes + request models.
All read routes are thin wrappers around palace.* methods.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Request Models (used by both read and edit routes)
# ---------------------------------------------------------------------------

class PatchMemoryRequest(BaseModel):
    chat_session_id: str
    summary_json: str | None = None
    importance: float | None = None
    archive_wing: str | None = None
    archive_room: str | None = None


class PatchKGTripleRequest(BaseModel):
    chat_session_id: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    valid_from: int | str | None = None
    valid_to: int | str | None = None


class PatchDirectEvidenceReviewRequest(BaseModel):
    chat_session_id: str
    capture_verification: str | None = None
    repair_needed: bool | None = None
    archive_state: str | None = None
    committed_gate: str | None = None
    review_note: str | None = None


class DirectEvidenceRevalidateRequest(BaseModel):
    chat_session_id: str
    review_note: str | None = None


class PatchDirectEvidenceTombstoneRequest(BaseModel):
    chat_session_id: str
    tombstoned: bool = True
    review_note: str | None = None


class PatchDirectEvidenceSupersedeRequest(BaseModel):
    chat_session_id: str
    superseded_by_id: int
    review_note: str | None = None


class ReindexRequest(BaseModel):
    chat_session_id: str
    force: bool = False
    batch_size: int = 20
    max_items: int = 200


class RescanRequest(BaseModel):
    chat_session_id: str
    max_items: int = 50
    turn_indices: list[int] = Field(default_factory=list)


class SessionMigrateRequest(BaseModel):
    source_session_id: str
    target_session_id: str
    dry_run: bool = False
    gate_status: str | None = None
    gate_reason: str | None = None
    gate_confidence: float | None = None
    gate_reads: int | None = None


class HypaImportSummary(BaseModel):
    text: str
    is_important: bool = False
    tags: list[str] = Field(default_factory=list)
    category: str = ""


class HypaImportRequest(BaseModel):
    chat_session_id: str
    summaries: list[HypaImportSummary]


class FeedbackRequest(BaseModel):
    chat_session_id: str
    target_type: str  # memory, kg_triple
    target_id: int
    feedback_value: str  # up, down
    feedback_note: str | None = None


# ---------------------------------------------------------------------------
#  Explorer Read Route Handlers
# ---------------------------------------------------------------------------

def list_sessions(palace, Utf8JSONResponse):
    """Return a list of all sessions (Explorer UI)."""
    try:
        sessions = palace.list_sessions()
        return {"status": "ok", "sessions": sessions, "count": len(sessions)}
    except Exception as exc:
        logger.error("GET /sessions failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "sessions": [], "count": 0, "detail": str(exc)},
            status_code=500,
        )


def explorer_chat_logs(palace, Utf8JSONResponse, chat_session_id: str = "", limit: int = 30, offset: int = 0):
    """Explorer UI: read chat_logs."""
    try:
        result = palace.get_chat_logs(
            chat_session_id=chat_session_id or None,
            limit=limit,
            offset=offset,
        )
        return {"status": "ok", **result}
    except Exception as exc:
        logger.error("GET /explorer/chat_logs failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "items": [], "total": 0, "detail": str(exc)},
            status_code=500,
        )


def explorer_memories(palace, Utf8JSONResponse, chat_session_id: str = "", limit: int = 30, offset: int = 0):
    """Explorer UI: read memories."""
    try:
        result = palace.get_memories(
            chat_session_id=chat_session_id or None,
            limit=limit,
            offset=offset,
        )
        return {"status": "ok", **result}
    except Exception as exc:
        logger.error("GET /explorer/memories failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "items": [], "total": 0, "detail": str(exc)},
            status_code=500,
        )


def explorer_direct_evidence(
    palace,
    Utf8JSONResponse,
    chat_session_id: str = "",
    limit: int = 30,
    offset: int = 0,
    include_tombstoned: bool = True,
):
    """Explorer UI: read direct_evidence_records."""
    try:
        result = palace.get_direct_evidence_records(
            chat_session_id=chat_session_id or None,
            limit=limit,
            offset=offset,
            include_tombstoned=include_tombstoned,
        )
        return {"status": "ok", **result}
    except Exception as exc:
        logger.error("GET /explorer/direct-evidence failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "items": [], "total": 0, "detail": str(exc)},
            status_code=500,
        )


def explorer_kg_triples(palace, Utf8JSONResponse, chat_session_id: str = "", limit: int = 30, offset: int = 0):
    """Explorer UI: read kg_triples."""
    try:
        result = palace.get_kg_triples(
            chat_session_id=chat_session_id or None,
            limit=limit,
            offset=offset,
        )
        return {"status": "ok", **result}
    except Exception as exc:
        logger.error("GET /explorer/kg_triples failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "items": [], "total": 0, "detail": str(exc)},
            status_code=500,
        )


def explorer_chapter_summaries(
    palace, Utf8JSONResponse, chat_session_id: str = "", limit: int = 30, offset: int = 0
):
    """Explorer UI: read chapter_summaries."""
    try:
        result = palace.get_chapter_summaries(
            chat_session_id=chat_session_id or None,
            limit=limit,
            offset=offset,
        )
        return {"status": "ok", **result}
    except Exception as exc:
        logger.error("GET /explorer/chapter_summaries failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "items": [], "total": 0, "detail": str(exc)},
            status_code=500,
        )


# ---------------------------------------------------------------------------
#  Explorer Edit Route Handlers (Priority 4-b)
# ---------------------------------------------------------------------------

def patch_memory(palace, Utf8JSONResponse, append_audit_log, memory_id: int, req: PatchMemoryRequest):
    """Edit a memory row (session-scoped)."""
    try:
        fields = {}
        if req.summary_json is not None:
            fields["summary_json"] = req.summary_json
        if req.importance is not None:
            fields["importance"] = req.importance
        if req.archive_wing is not None:
            fields["archive_wing"] = req.archive_wing
        if req.archive_room is not None:
            fields["archive_room"] = req.archive_room

        if not fields:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "No fields to update."},
                status_code=400,
            )

        result = palace.update_memory(memory_id, req.chat_session_id, fields)

        if result["ok"]:
            append_audit_log(
                event_type="memory_edit",
                chat_session_id=req.chat_session_id,
                target_type="memory",
                target_id=memory_id,
                summary=f"Edited memory {memory_id}: {list(fields.keys())}",
                details={"fields": fields},
                source="manual_ui",
            )
            return {"status": "ok", "detail": result["detail"], "updated_fields": result["updated_fields"]}
        else:
            return Utf8JSONResponse(
                content={"status": "error", "detail": result["detail"], "updated_fields": []},
                status_code=400,
            )
    except Exception as exc:
        logger.error("PATCH /explorer/memories/%d failed: %s", memory_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc), "updated_fields": []},
            status_code=500,
        )


def patch_kg_triple(palace, Utf8JSONResponse, append_audit_log, triple_id: int, req: PatchKGTripleRequest):
    """Edit a kg_triple row (session-scoped)."""
    try:
        fields = {}
        if req.subject is not None:
            fields["subject"] = req.subject
        if req.predicate is not None:
            fields["predicate"] = req.predicate
        if req.object is not None:
            fields["object"] = req.object
        if req.valid_from is not None:
            fields["valid_from"] = req.valid_from
        if req.valid_to is not None:
            fields["valid_to"] = req.valid_to

        if not fields:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "No fields to update."},
                status_code=400,
            )

        result = palace.update_kg_triple(triple_id, req.chat_session_id, fields)

        if result["ok"]:
            append_audit_log(
                event_type="kg_edit",
                chat_session_id=req.chat_session_id,
                target_type="kg_triple",
                target_id=triple_id,
                summary=f"Edited kg_triple {triple_id}: {list(fields.keys())}",
                details={"fields": fields},
                source="manual_ui",
            )
            return {"status": "ok", "detail": result["detail"], "updated_fields": result["updated_fields"]}
        else:
            return Utf8JSONResponse(
                content={"status": "error", "detail": result["detail"], "updated_fields": []},
                status_code=400,
            )
    except Exception as exc:
        logger.error("PATCH /explorer/kg_triples/%d failed: %s", triple_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc), "updated_fields": []},
            status_code=500,
        )


def patch_direct_evidence_review(
    palace,
    Utf8JSONResponse,
    append_audit_log,
    SessionLocal,
    DirectEvidenceRecord,
    normalize_capture_verification,
    normalize_archive_state,
    normalize_committed_gate,
    DIRECT_EVIDENCE_CAPTURE_VERIFICATIONS,
    DIRECT_EVIDENCE_ARCHIVE_STATES,
    DIRECT_EVIDENCE_COMMITTED_GATES,
    perf_counter,
    record_id: int,
    req: PatchDirectEvidenceReviewRequest,
):
    """Review a direct evidence row and update verification/archive/gate."""
    try:
        started_at = perf_counter()
        sid = (req.chat_session_id or "").strip()
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required.", "updated_fields": []},
                status_code=400,
            )

        if (
            req.capture_verification is None
            and req.repair_needed is None
            and req.archive_state is None
            and req.committed_gate is None
        ):
            return Utf8JSONResponse(
                content={"status": "error", "detail": "No fields to update.", "updated_fields": []},
                status_code=400,
            )

        normalized_capture_verification = None
        if req.capture_verification is not None:
            normalized_capture_verification = normalize_capture_verification(req.capture_verification)
            if normalized_capture_verification not in DIRECT_EVIDENCE_CAPTURE_VERIFICATIONS:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": f"invalid capture_verification: {req.capture_verification!r}", "updated_fields": []},
                    status_code=400,
                )

        normalized_archive_state = None
        if req.archive_state is not None:
            normalized_archive_state = normalize_archive_state(req.archive_state)
            if normalized_archive_state not in DIRECT_EVIDENCE_ARCHIVE_STATES:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": f"invalid archive_state: {req.archive_state!r}", "updated_fields": []},
                    status_code=400,
                )

        normalized_committed_gate = None
        if req.committed_gate is not None:
            normalized_committed_gate = normalize_committed_gate(req.committed_gate)
            if normalized_committed_gate not in DIRECT_EVIDENCE_COMMITTED_GATES:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": f"invalid committed_gate: {req.committed_gate!r}", "updated_fields": []},
                    status_code=400,
                )

        db = SessionLocal()
        try:
            row = db.query(DirectEvidenceRecord).filter(DirectEvidenceRecord.id == record_id).first()
            if not row:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": f"direct evidence {record_id} not found", "updated_fields": []},
                    status_code=404,
                )
            if row.chat_session_id != sid:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": "Session mismatch: cannot edit row from a different session.", "updated_fields": []},
                    status_code=400,
                )
            if bool(row.tombstoned):
                return Utf8JSONResponse(
                    content={"status": "error", "detail": "Cannot review a tombstoned direct evidence.", "updated_fields": []},
                    status_code=400,
                )
            if row.superseded_by_id is not None:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": "Cannot review a superseded direct evidence.", "updated_fields": []},
                    status_code=400,
                )

            current_capture_verification = normalize_capture_verification(row.capture_verification)
            next_capture_verification = normalized_capture_verification or current_capture_verification

            if req.repair_needed is not None:
                next_repair_needed = bool(req.repair_needed)
            elif normalized_capture_verification == "verified":
                next_repair_needed = False
            elif normalized_capture_verification in {"rejected", "needs_review"}:
                next_repair_needed = True
            else:
                next_repair_needed = bool(row.repair_needed)

            if normalized_archive_state is not None:
                next_archive_state = normalized_archive_state
            elif next_capture_verification == "verified" and not next_repair_needed:
                next_archive_state = "verified_direct"
            elif next_repair_needed or next_capture_verification in {"rejected", "needs_review"}:
                next_archive_state = "repair_queue"
            else:
                next_archive_state = normalize_archive_state(row.archive_state)

            next_committed_gate = normalized_committed_gate or "manual"

            row.capture_verification = next_capture_verification
            row.repair_needed = next_repair_needed
            row.archive_state = next_archive_state
            row.committed_gate = next_committed_gate
            db.commit()
            db.refresh(row)

            updated_fields = ["capture_verification", "repair_needed", "archive_state", "committed_gate"]
            latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
            cost_measurement = {
                "policy_version": "lc1a.v1",
                "operation": "direct_evidence_review",
                "latency_ms": latency_ms,
                "rows_touched": 1,
                "repair_queue_target": bool(row.repair_needed),
            }
            append_audit_log(
                event_type="direct_evidence_review",
                chat_session_id=sid,
                target_type="direct_evidence",
                target_id=record_id,
                summary=f"Reviewed direct evidence {record_id}",
                details={
                    "capture_verification": row.capture_verification,
                    "repair_needed": bool(row.repair_needed),
                    "archive_state": row.archive_state,
                    "committed_gate": row.committed_gate,
                    "review_note": (req.review_note or "").strip() or None,
                    "cost_measurement": cost_measurement,
                },
                source="manual_ui",
            )

            return {
                "status": "ok",
                "record_id": row.id,
                "capture_verification": row.capture_verification,
                "repair_needed": bool(row.repair_needed),
                "archive_state": row.archive_state,
                "committed_gate": row.committed_gate,
                "updated_fields": updated_fields,
                "cost_measurement": cost_measurement,
            }
        finally:
            db.close()
    except Exception as exc:
        logger.error("PATCH /explorer/direct-evidence/%d/review failed: %s", record_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc), "updated_fields": []},
            status_code=500,
        )


def patch_direct_evidence_revalidate(
    palace,
    Utf8JSONResponse,
    append_audit_log,
    SessionLocal,
    DirectEvidenceRecord,
    normalize_capture_verification,
    normalize_archive_state,
    perf_counter,
    record_id: int,
    req: DirectEvidenceRevalidateRequest,
):
    """Revalidate a direct evidence in repair queue to finalize state."""
    try:
        started_at = perf_counter()
        sid = (req.chat_session_id or "").strip()
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required.", "updated_fields": []},
                status_code=400,
            )

        db = SessionLocal()
        try:
            row = db.query(DirectEvidenceRecord).filter(DirectEvidenceRecord.id == record_id).first()
            if not row:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": f"direct evidence {record_id} not found", "updated_fields": []},
                    status_code=404,
                )
            if row.chat_session_id != sid:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": "Session mismatch: cannot edit row from a different session.", "updated_fields": []},
                    status_code=400,
                )
            if bool(row.tombstoned):
                return Utf8JSONResponse(
                    content={"status": "error", "detail": "Cannot revalidate a tombstoned direct evidence.", "updated_fields": []},
                    status_code=400,
                )
            if row.superseded_by_id is not None:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": "Cannot revalidate a superseded direct evidence.", "updated_fields": []},
                    status_code=400,
                )

            current_capture_verification = normalize_capture_verification(row.capture_verification)
            current_archive_state = normalize_archive_state(row.archive_state)
            current_repair_needed = bool(row.repair_needed)

            if not current_repair_needed and current_capture_verification not in {"needs_review", "rejected"} and current_archive_state != "repair_queue":
                return Utf8JSONResponse(
                    content={
                        "status": "error",
                        "detail": "Revalidation not allowed: only direct evidence in repair queue or needs_review/rejected state can be revalidated.",
                        "updated_fields": [],
                    },
                    status_code=400,
                )

            row.capture_verification = "verified"
            row.repair_needed = False
            row.archive_state = "verified_direct"
            row.committed_gate = "finalize"
            db.commit()
            db.refresh(row)

            updated_fields = ["capture_verification", "repair_needed", "archive_state", "committed_gate"]
            latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
            cost_measurement = {
                "policy_version": "lc1a.v1",
                "operation": "direct_evidence_revalidate",
                "latency_ms": latency_ms,
                "rows_touched": 1,
                "repair_queue_exit": True,
            }
            append_audit_log(
                event_type="direct_evidence_revalidate",
                chat_session_id=sid,
                target_type="direct_evidence",
                target_id=record_id,
                summary=f"Revalidated direct evidence {record_id}",
                details={
                    "capture_verification": row.capture_verification,
                    "repair_needed": bool(row.repair_needed),
                    "archive_state": row.archive_state,
                    "committed_gate": row.committed_gate,
                    "review_note": (req.review_note or "").strip() or None,
                    "cost_measurement": cost_measurement,
                },
                source="manual_ui",
            )

            return {
                "status": "ok",
                "record_id": row.id,
                "capture_verification": row.capture_verification,
                "repair_needed": bool(row.repair_needed),
                "archive_state": row.archive_state,
                "committed_gate": row.committed_gate,
                "updated_fields": updated_fields,
                "cost_measurement": cost_measurement,
            }
        finally:
            db.close()
    except Exception as exc:
        logger.error("PATCH /explorer/direct-evidence/%d/revalidate failed: %s", record_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc), "updated_fields": []},
            status_code=500,
        )


def patch_direct_evidence_tombstone(
    palace,
    Utf8JSONResponse,
    append_audit_log,
    SessionLocal,
    DirectEvidenceRecord,
    DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS,
    record_id: int,
    req: PatchDirectEvidenceTombstoneRequest,
):
    """Set or unset tombstone flag on a direct evidence row."""
    try:
        sid = (req.chat_session_id or "").strip()
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required.", "updated_fields": []},
                status_code=400,
            )

        db = SessionLocal()
        try:
            row = db.query(DirectEvidenceRecord).filter(DirectEvidenceRecord.id == record_id).first()
            if not row:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": f"direct evidence {record_id} not found", "updated_fields": []},
                    status_code=404,
                )
            if row.chat_session_id != sid:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": "Session mismatch: cannot edit row from a different session.", "updated_fields": []},
                    status_code=400,
                )

            row.tombstoned = bool(req.tombstoned)
            db.commit()
            db.refresh(row)

            append_audit_log(
                event_type="direct_evidence_tombstone",
                chat_session_id=sid,
                target_type="direct_evidence",
                target_id=record_id,
                summary=f"Tombstone {'enabled' if row.tombstoned else 'disabled'} for direct evidence {record_id}",
                details={
                    "tombstoned": bool(row.tombstoned),
                    "superseded_by_id": row.superseded_by_id,
                    "review_note": (req.review_note or "").strip() or None,
                    "deleted_turn_tombstone_retention_window_turns": DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS,
                },
                source="manual_ui",
            )

            return {
                "status": "ok",
                "record_id": row.id,
                "tombstoned": bool(row.tombstoned),
                "superseded_by_id": row.superseded_by_id,
                "deleted_turn_tombstone_retention_window_turns": DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS,
                "updated_fields": ["tombstoned"],
            }
        finally:
            db.close()
    except Exception as exc:
        logger.error("PATCH /explorer/direct-evidence/%d/tombstone failed: %s", record_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc), "updated_fields": []},
            status_code=500,
        )


def patch_direct_evidence_supersede(
    palace,
    Utf8JSONResponse,
    append_audit_log,
    SessionLocal,
    DirectEvidenceRecord,
    record_id: int,
    req: PatchDirectEvidenceSupersedeRequest,
):
    """Mark a direct evidence row as superseded by a newer record."""
    try:
        sid = (req.chat_session_id or "").strip()
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required.", "updated_fields": []},
                status_code=400,
            )

        if req.superseded_by_id <= 0:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "superseded_by_id must be a positive integer.", "updated_fields": []},
                status_code=400,
            )
        if req.superseded_by_id == record_id:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "Cannot supersede a record with itself.", "updated_fields": []},
                status_code=400,
            )

        db = SessionLocal()
        try:
            row = db.query(DirectEvidenceRecord).filter(DirectEvidenceRecord.id == record_id).first()
            if not row:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": f"direct evidence {record_id} not found", "updated_fields": []},
                    status_code=404,
                )
            if row.chat_session_id != sid:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": "Session mismatch: cannot edit row from a different session.", "updated_fields": []},
                    status_code=400,
                )

            replacement = (
                db.query(DirectEvidenceRecord)
                .filter(DirectEvidenceRecord.id == req.superseded_by_id)
                .first()
            )
            if not replacement:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": f"superseded_by target {req.superseded_by_id} not found", "updated_fields": []},
                    status_code=404,
                )
            if replacement.chat_session_id != sid:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": "Session mismatch: supersede target must belong to the same session.", "updated_fields": []},
                    status_code=400,
                )

            row.superseded_by_id = int(req.superseded_by_id)
            db.commit()
            db.refresh(row)

            append_audit_log(
                event_type="direct_evidence_supersede",
                chat_session_id=sid,
                target_type="direct_evidence",
                target_id=record_id,
                summary=f"Superseded direct evidence {record_id} by {req.superseded_by_id}",
                details={
                    "superseded_by_id": row.superseded_by_id,
                    "tombstoned": bool(row.tombstoned),
                    "review_note": (req.review_note or "").strip() or None,
                },
                source="manual_ui",
            )

            return {
                "status": "ok",
                "record_id": row.id,
                "superseded_by_id": row.superseded_by_id,
                "tombstoned": bool(row.tombstoned),
                "updated_fields": ["superseded_by_id"],
            }
        finally:
            db.close()
    except Exception as exc:
        logger.error("PATCH /explorer/direct-evidence/%d/supersede failed: %s", record_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc), "updated_fields": []},
            status_code=500,
        )


def regenerate_memory(
    palace,
    Utf8JSONResponse,
    append_audit_log,
    SessionLocal,
    ChatLog,
    critic,
    _looks_like_ooc_text_main,
    body: dict,
    background_tasks,
):
    """Regenerate memory for a specific turn by re-running Critic."""
    try:
        sid = (body.get("chat_session_id") or "").strip()
        turn_idx = body.get("turn_index")
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required."},
                status_code=400,
            )
        if turn_idx is None:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "turn_index is required."},
                status_code=400,
            )
        try:
            turn_idx = int(turn_idx)
        except (ValueError, TypeError):
            return Utf8JSONResponse(
                content={"status": "error", "detail": "turn_index must be an integer."},
                status_code=400,
            )

        db = SessionLocal()
        try:
            logs = (
                db.query(ChatLog)
                .filter(ChatLog.chat_session_id == sid, ChatLog.turn_index == turn_idx)
                .order_by(ChatLog.id.asc())
                .all()
            )
            if not logs:
                return Utf8JSONResponse(
                    content={"status": "error", "detail": f"No ChatLog found for turn {turn_idx}."},
                    status_code=404,
                )

            user_text = ""
            assistant_text = ""
            for log in logs:
                if log.role == "user":
                    user_text = (log.content or "")[:500]
                elif log.role == "assistant":
                    assistant_text = (log.content or "")[:500]
            turn_content = f"{{{{user}}}}: {user_text}\n{{{{char}}}}: {assistant_text}"
            if _looks_like_ooc_text_main(user_text):
                return {
                    "status": "ok",
                    "detail": f"Turn {turn_idx} is OOC, skipping regeneration.",
                    "skipped": True,
                    "skip_reason": "ooc_turn_guard",
                }

            context_logs = (
                db.query(ChatLog)
                .filter(
                    ChatLog.chat_session_id == sid,
                    ChatLog.turn_index < turn_idx,
                    ChatLog.role.in_(["user", "assistant"]),
                )
                .order_by(ChatLog.turn_index.desc(), ChatLog.id.desc())
                .limit(6)
                .all()
            )
            context = [
                {"role": cl.role, "content": (cl.content or "")[:300]}
                for cl in reversed(context_logs)
            ]
        finally:
            db.close()

        background_tasks.add_task(
            critic.run,
            turn_idx,
            turn_content,
            context,
            sid,
        )

        append_audit_log(
            event_type="memory_regenerate",
            chat_session_id=sid,
            target_type="memory",
            target_id=None,
            summary=f"Regenerate memory for turn {turn_idx}",
            source="explorer_ui",
        )

        logger.info("POST /explorer/memories/regenerate accepted (turn=%d, session=%s)", turn_idx, sid)
        return {"status": "ok", "detail": f"Memory regeneration started for turn {turn_idx}."}

    except Exception as exc:
        logger.exception("regenerate_memory failed")
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )


def delete_memory(palace, Utf8JSONResponse, append_audit_log, memory_id: int, chat_session_id: str = ""):
    """Hard delete a memory row (session-scoped)."""
    try:
        sid = (chat_session_id or "").strip()
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required."},
                status_code=400,
            )

        result = palace.delete_memory(memory_id, sid)

        if result["ok"]:
            append_audit_log(
                event_type="memory_delete",
                chat_session_id=sid,
                target_type="memory",
                target_id=memory_id,
                summary=f"Deleted memory {memory_id}",
                source="manual_ui",
            )
            return {
                "status": "ok",
                "detail": result["detail"],
                "deleted_id": result["deleted_id"],
                "deleted_type": result["deleted_type"],
                "session_id": sid,
            }
        else:
            return Utf8JSONResponse(
                content={
                    "status": "error",
                    "detail": result["detail"],
                    "deleted_id": result["deleted_id"],
                    "deleted_type": result["deleted_type"],
                    "session_id": sid,
                },
                status_code=400,
            )
    except Exception as exc:
        logger.error("DELETE /explorer/memories/%d failed: %s", memory_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc), "deleted_id": None, "deleted_type": "memory", "session_id": chat_session_id},
            status_code=500,
        )


def delete_kg_triple(palace, Utf8JSONResponse, append_audit_log, triple_id: int, chat_session_id: str = ""):
    """Hard delete a kg_triple row (session-scoped)."""
    try:
        sid = (chat_session_id or "").strip()
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required."},
                status_code=400,
            )

        result = palace.delete_kg_triple(triple_id, sid)

        if result["ok"]:
            append_audit_log(
                event_type="kg_delete",
                chat_session_id=sid,
                target_type="kg_triple",
                target_id=triple_id,
                summary=f"Deleted kg_triple {triple_id}",
                source="manual_ui",
            )
            return {
                "status": "ok",
                "detail": result["detail"],
                "deleted_id": result["deleted_id"],
                "deleted_type": result["deleted_type"],
                "session_id": sid,
            }
        else:
            return Utf8JSONResponse(
                content={
                    "status": "error",
                    "detail": result["detail"],
                    "deleted_id": result["deleted_id"],
                    "deleted_type": result["deleted_type"],
                    "session_id": sid,
                },
                status_code=400,
            )
    except Exception as exc:
        logger.error("DELETE /explorer/kg_triples/%d failed: %s", triple_id, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc), "deleted_id": None, "deleted_type": "kg_triple", "session_id": chat_session_id},
            status_code=500,
        )
