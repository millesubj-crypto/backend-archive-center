"""Admin service module.

Priority 4-c: Admin utility routes extraction.
All admin routes are thin wrappers around palace.* methods.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def admin_reindex(
    palace,
    Utf8JSONResponse,
    append_audit_log,
    retrieval_index_registry,
    _mark_retrieval_index_dirty_q1c,
    req,
):
    """Reindex session memory embeddings with current model."""
    try:
        sid = (req.chat_session_id or "").strip()
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required."},
                status_code=400,
            )

        result = palace.reindex_session(
            chat_session_id=sid,
            force=req.force,
            batch_size=req.batch_size,
            max_items=req.max_items,
        )

        if result.get("ok"):
            schedule = result.get("reembed_schedule") if isinstance(result.get("reembed_schedule"), dict) else {}
            embedding_update = schedule.get("embedding_model_update") if isinstance(schedule.get("embedding_model_update"), dict) else {}
            retrieval_refresh = schedule.get("retrieval_index_refresh_handoff") if isinstance(schedule.get("retrieval_index_refresh_handoff"), dict) else {}
            updated_rows = int(embedding_update.get("updated_rows") or 0)
            updated_turn_range = embedding_update.get("updated_turn_range") if isinstance(embedding_update.get("updated_turn_range"), list) else None
            dirty_turn = None
            if isinstance(updated_turn_range, list) and len(updated_turn_range) == 2 and isinstance(updated_turn_range[1], int):
                dirty_turn = int(updated_turn_range[1])

            retrieval_index_handoff = {
                "requested": False,
                "status": "no_row_updates",
                "dirty_reason": retrieval_refresh.get("dirty_reason") or "reembed_provenance_refresh",
                "refresh_trigger": retrieval_refresh.get("refresh_trigger") or "next_prepare_turn",
                "refresh_mode": retrieval_refresh.get("refresh_mode") or "mark_dirty_then_next_prepare_turn_refresh",
                "index_snapshot": retrieval_index_registry.snapshot(sid),
            }

            if updated_rows > 0:
                index_snapshot = _mark_retrieval_index_dirty_q1c(
                    sid,
                    reason=str(retrieval_index_handoff["dirty_reason"]),
                    turn_index=dirty_turn,
                )
                if isinstance(index_snapshot, dict):
                    if index_snapshot.get("status") == "off":
                        handoff_status = "shadow_mode_off"
                    elif index_snapshot.get("dirty") is True:
                        handoff_status = "dirty_marked"
                    else:
                        handoff_status = "snapshot_returned"
                else:
                    handoff_status = "dirty_mark_failed"

                retrieval_index_handoff = {
                    "requested": True,
                    "status": handoff_status,
                    "dirty_reason": retrieval_refresh.get("dirty_reason") or "reembed_provenance_refresh",
                    "refresh_trigger": retrieval_refresh.get("refresh_trigger") or "next_prepare_turn",
                    "refresh_mode": retrieval_refresh.get("refresh_mode") or "mark_dirty_then_next_prepare_turn_refresh",
                    "index_snapshot": index_snapshot,
                }

            result["retrieval_index_handoff"] = retrieval_index_handoff
            append_audit_log(
                event_type="reindex",
                chat_session_id=sid,
                target_type="session",
                summary=f"Reindex session {sid}: {result.get('succeeded', 0)} updated",
                details={k: v for k, v in result.items() if k != "ok"},
                source="batch",
            )
            return {"status": "ok", **result}
        else:
            return Utf8JSONResponse(
                content={"status": "error", "detail": result.get("detail", "unknown error")},
                status_code=400,
            )
    except Exception as exc:
        logger.error("POST /admin/reindex failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )


def admin_rescan(
    palace,
    Utf8JSONResponse,
    append_audit_log,
    req,
):
    """Rescan session chat_logs and re-run Critic for missing memories."""
    try:
        sid = (req.chat_session_id or "").strip()
        if not sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id is required."},
                status_code=400,
            )

        result = palace.rescan_session(
            chat_session_id=sid,
            max_items=req.max_items,
            turn_indices=req.turn_indices or None,
        )

        if result.get("ok"):
            append_audit_log(
                event_type="rescan",
                chat_session_id=sid,
                target_type="session",
                summary=f"Rescan session {sid}: {result.get('succeeded', 0)} succeeded",
                details={k: v for k, v in result.items() if k != "ok"},
                source="batch",
            )
            return {"status": "ok", **result}
        else:
            return Utf8JSONResponse(
                content={"status": "error", "detail": result.get("detail", "unknown error")},
                status_code=400,
            )
    except Exception as exc:
        logger.error("POST /admin/rescan failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )


def admin_session_migrate(
    palace,
    Utf8JSONResponse,
    append_audit_log,
    req,
):
    """Migrate compat alias session rows to primary session."""
    try:
        source_sid = (req.source_session_id or "").strip()
        target_sid = (req.target_session_id or "").strip()

        if not source_sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "source_session_id is required."},
                status_code=400,
            )
        if not target_sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "target_session_id is required."},
                status_code=400,
            )
        if source_sid == target_sid:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "source_session_id and target_session_id must be different."},
                status_code=400,
            )

        result = palace.migrate_session_data(
            source_session_id=source_sid,
            target_session_id=target_sid,
            dry_run=bool(req.dry_run),
            gate_status=req.gate_status,
        )

        if result.get("ok"):
            event_type = "session_migrate_dry_run" if req.dry_run else "session_migrate"
            append_audit_log(
                event_type=event_type,
                chat_session_id=target_sid,
                target_type="session",
                summary=(
                    f"Session migrate {'dry-run' if req.dry_run else 'apply'}: "
                    f"{source_sid} -> {target_sid}"
                ),
                details={
                    "source_session_id": source_sid,
                    "target_session_id": target_sid,
                    "dry_run": bool(req.dry_run),
                    "moved_rows": int(result.get("moved_rows") or 0),
                    "dropped_rows": int(result.get("dropped_rows") or 0),
                    "source_rows_remaining": int(result.get("source_rows_remaining") or 0),
                    "tables_with_source_rows": int(result.get("tables_with_source_rows") or 0),
                    "gate_status": req.gate_status,
                    "gate_reason": req.gate_reason,
                    "gate_confidence": req.gate_confidence,
                    "gate_reads": req.gate_reads,
                },
                source="batch",
            )
            return {"status": "ok", **result}

        return Utf8JSONResponse(
            content={"status": "error", "detail": result.get("detail", "unknown error")},
            status_code=400,
        )
    except Exception as exc:
        logger.error("POST /admin/session-migrate failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )


# ---------------------------------------------------------------------------
#  Priority 4-d-1: Audit + Feedback Routes
# ---------------------------------------------------------------------------

def get_audit_logs(
    SessionLocal,
    AuditLog,
    to_kst,
    Utf8JSONResponse,
    chat_session_id: str = "",
    event_type: str = "",
    limit: int = 20,
    offset: int = 0,
):
    """Return audit logs ordered by newest first."""
    try:
        db = SessionLocal()
        try:
            q = db.query(AuditLog).order_by(AuditLog.created_at.desc())

            sid = (chat_session_id or "").strip()
            if sid:
                q = q.filter(AuditLog.chat_session_id == sid)

            evt = (event_type or "").strip()
            if evt:
                q = q.filter(AuditLog.event_type == evt)

            clamped_limit = max(1, min(limit, 200))
            clamped_offset = max(0, offset)

            total = q.count()
            rows = q.offset(clamped_offset).limit(clamped_limit).all()

            items = []
            for r in rows:
                items.append({
                    "id": r.id,
                    "created_at": to_kst(r.created_at),
                    "event_type": r.event_type,
                    "chat_session_id": r.chat_session_id,
                    "target_type": r.target_type,
                    "target_id": r.target_id,
                    "summary": r.summary,
                    "details_json": r.details_json,
                    "source": r.source,
                })

            return {
                "status": "ok",
                "items": items,
                "total": total,
                "limit": clamped_limit,
                "offset": clamped_offset,
            }
        finally:
            db.close()
    except Exception as exc:
        logger.error("GET /audit failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )


def post_feedback(
    save_critic_feedback,
    append_audit_log,
    Utf8JSONResponse,
    req,
):
    """Submit feedback on a Critic output (memory / kg_triple)."""
    try:
        result = save_critic_feedback(
            chat_session_id=req.chat_session_id,
            target_type=req.target_type,
            target_id=req.target_id,
            feedback_value=req.feedback_value,
            feedback_note=req.feedback_note,
            source="manual_ui",
        )

        if result["ok"]:
            append_audit_log(
                event_type="critic_feedback",
                chat_session_id=req.chat_session_id,
                target_type=req.target_type,
                target_id=req.target_id,
                summary=f"Feedback {req.feedback_value} on {req.target_type} #{req.target_id}",
                details={"feedback_value": req.feedback_value, "feedback_note": req.feedback_note},
                source="manual_ui",
            )
            return {"status": "ok", **result}
        else:
            return Utf8JSONResponse(
                content={"status": "error", "detail": result["detail"]},
                status_code=400,
            )
    except Exception as exc:
        logger.error("POST /feedback failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )


def get_feedback_latest(
    get_latest_feedback,
    Utf8JSONResponse,
    chat_session_id: str = "",
    target_type: str = "",
    target_ids: str = "",
):
    """Return latest feedback for given targets.

    target_ids is a comma-separated string of integers: "1,2,3"
    """
    try:
        sid = (chat_session_id or "").strip()
        tt = (target_type or "").strip()
        if not sid or not tt:
            return Utf8JSONResponse(
                content={"status": "error", "detail": "chat_session_id and target_type are required."},
                status_code=400,
            )

        ids = []
        for part in (target_ids or "").split(","):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))

        if not ids:
            return {"status": "ok", "feedbacks": {}}

        feedbacks = get_latest_feedback(
            chat_session_id=sid,
            target_type=tt,
            target_ids=ids,
        )

        # keys converted to strings for JSON safety
        return {
            "status": "ok",
            "feedbacks": {str(k): v for k, v in feedbacks.items()},
        }
    except Exception as exc:
        logger.error("GET /feedback/latest failed: %s", exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )
