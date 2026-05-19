from __future__ import annotations

import logging
import threading as _threading
from typing import Callable

from sqlalchemy import func

from backend.database import SessionLocal
from backend.models import ChatLog
from backend.services.maintenance_pass import (
    MaintenanceEnqueueRequest,
    MaintenanceEnqueueResponse,
    MaintenancePassRequest,
)

logger = logging.getLogger(__name__)

# ================================================================== #
#  N-1b/c: Background Maintenance Queue                              #
#  - per-session single-writer                                       #
#  - pending slot: newest job only (stale discard on replace)        #
#  - N-1c: turn_index stale check + rollback invalidation           #
# ================================================================== #

# module-level queue state: {session_id: {"running": bool, "pending": dict|None}}
_mq: dict[str, dict] = {}
_mq_lock = _threading.Lock()

# R-3d: per-session long_session_health snapshot cache: {session_id: snapshot dict}
_r3d_health_cache: dict[str, dict] = {}
_r3d_health_lock = _threading.Lock()

# Injected callbacks from main.py to avoid circular imports.
_run_maintenance_pass: Callable[[str, MaintenancePassRequest], None] | None = None
_run_summary_refresh: Callable[..., dict] | None = None


def bind_maintenance_callbacks(
    *,
    run_maintenance_pass: Callable[[str, MaintenancePassRequest], None],
    run_summary_refresh: Callable[..., dict],
) -> None:
    """Bind runtime callbacks required by the queue worker.

    Must be called once during application startup before any enqueue.
    """
    global _run_maintenance_pass, _run_summary_refresh
    _run_maintenance_pass = run_maintenance_pass
    _run_summary_refresh = run_summary_refresh


def _mq_discard_session(session_id: str) -> None:
    """N-1c: discard pending maintenance job for a session after rollback.

    Running jobs are shadow-only and cannot be cancelled mid-flight;
    only the pending slot is cleared so stale jobs do not execute after rollback.
    """
    with _mq_lock:
        state = _mq.get(session_id)
        if state:
            dropped = state.get("pending") is not None
            state["pending"] = None
            if dropped:
                logger.debug("[N-1c] mq pending discarded on rollback: session=%s", session_id)


def _mq_execute(session_id: str, job: dict) -> None:
    """N-1b: execute a per-session maintenance job.

    If a pending job exists after completion, it is started immediately.
    Always runs inside a daemon thread (isolated from request lifecycle).
    """
    _job_turn = job.get("turn_index", -1)
    _stale_discard = False

    # N-1c: turn_index stale check (post-rollback guard)
    if _job_turn > 0:
        try:
            _sdb = SessionLocal()
            try:
                _max_turn = (
                    _sdb.query(func.max(ChatLog.turn_index))
                    .filter(ChatLog.chat_session_id == session_id)
                    .scalar()
                )
                _max_turn = _max_turn if _max_turn is not None else -1
                if _max_turn >= 0 and _job_turn > _max_turn + 1:
                    logger.debug(
                        "[N-1c] mq stale discard session=%s job_turn=%d max_turn=%d",
                        session_id,
                        _job_turn,
                        _max_turn,
                    )
                    _stale_discard = True
            finally:
                _sdb.close()
        except Exception as _v_err:
            logger.debug("[N-1c] mq turn check error (non-fatal): %s", _v_err)

    try:
        if _stale_discard:
            return

        if _run_maintenance_pass is None:
            logger.warning("[N-1b] mq execute skipped: run_maintenance_pass callback not bound")
            return

        _req = MaintenancePassRequest(
            turn_index=_job_turn,
            shadow_only=True,
            assistant_response=job.get("assistant_response", ""),
            recent_responses=job.get("recent_responses", []),
            supervisor_result=job.get("supervisor_result"),
        )
        _run_maintenance_pass(session_id, _req)

        # R-3c: run hierarchy summary refresh opportunistically from the queue.
        if _run_summary_refresh is not None:
            _refresh_result = _run_summary_refresh(
                session_id,
                turn_index=_job_turn,
                summary_refresh=(job.get("summary_refresh") or {}),
            )
            if (_refresh_result or {}).get("triggered_layer_count", 0) > 0:
                logger.info(
                    "[R-3c] maintenance refresh executed session=%s turn=%d status=%s triggered=%d",
                    session_id,
                    _job_turn,
                    _refresh_result.get("status"),
                    _refresh_result.get("triggered_layer_count", 0),
                )
    except Exception as _e:
        logger.warning("[N-1b] mq execute error session=%s: %s", session_id, _e)
    finally:
        # After completion: if pending exists, start it immediately; otherwise go idle.
        with _mq_lock:
            _state = _mq.get(session_id)
            if _state:
                _pending = _state.get("pending")
                if _pending:
                    _state["pending"] = None
                    t = _threading.Thread(
                        target=_mq_execute, args=(session_id, _pending), daemon=True
                    )
                    t.start()
                else:
                    _state["running"] = False


def enqueue_maintenance_job(req: MaintenanceEnqueueRequest) -> MaintenanceEnqueueResponse:
    """N-1b: enqueue a maintenance job.

    - single-writer per session
    - pending slot holds only the newest job (stale discard)
    - shadow_only=True invariant (N-1 baseline policy)
    - actual execution happens in a background daemon thread
    """
    sid = (req.chat_session_id or "").strip()
    if not sid:
        return MaintenanceEnqueueResponse(
            status="skipped",
            chat_session_id="",
            turn_index=-1,
            queue_depth=0,
        )

    job = {
        "chat_session_id": sid,
        "turn_index": req.turn_index,
        "shadow_only": True,
        "assistant_response": (req.assistant_response or "")[:400],
        "recent_responses": (req.recent_responses or [])[-5:],
        "supervisor_result": req.supervisor_result,
        "summary_refresh": req.summary_refresh,
    }

    with _mq_lock:
        state = _mq.setdefault(sid, {"running": False, "pending": None})

        if not state["running"]:
            state["running"] = True
            status = "enqueued"
            queue_depth = 1
            t = _threading.Thread(target=_mq_execute, args=(sid, job), daemon=True)
            t.start()
        else:
            was_pending = state["pending"] is not None
            state["pending"] = job
            status = "replaced_pending" if was_pending else "queued_pending"
            queue_depth = 2

    logger.debug("[N-1b] mq enqueue session=%s turn=%d status=%s", sid, req.turn_index, status)
    return MaintenanceEnqueueResponse(
        status=status,
        chat_session_id=sid,
        turn_index=req.turn_index,
        queue_depth=queue_depth,
    )


def get_maintenance_queue_status(session_id: str = "") -> dict:
    """N-1d: return maintenance queue status.

    If session_id is provided, return that session only.
    If empty, return all active sessions.
    """
    with _mq_lock:
        if session_id:
            sid = session_id.strip()
            state = _mq.get(sid)
            if state is None:
                health_snap = _r3d_get_health_snapshot(sid)
                return {
                    "sessions": [],
                    "total_running": 0,
                    "total_pending": 0,
                    "shadow_only": True,
                    "long_session_health_snapshot": health_snap,
                    "note": "session not found in queue",
                }
            has_pending = state.get("pending") is not None
            running = bool(state.get("running"))
            health_snap = _r3d_get_health_snapshot(sid)
            return {
                "sessions": [{
                    "session_id": sid,
                    "running": running,
                    "has_pending": has_pending,
                    "queue_depth": (1 if running else 0) + (1 if has_pending else 0),
                }],
                "total_running": 1 if running else 0,
                "total_pending": 1 if has_pending else 0,
                "shadow_only": True,
                "long_session_health_snapshot": health_snap,
            }
        else:
            sessions = []
            total_running = 0
            total_pending = 0
            for _sid, _state in _mq.items():
                _running = bool(_state.get("running"))
                _has_pending = _state.get("pending") is not None
                if _running or _has_pending:
                    sessions.append({
                        "session_id": _sid,
                        "running": _running,
                        "has_pending": _has_pending,
                        "queue_depth": (1 if _running else 0) + (1 if _has_pending else 0),
                    })
                    if _running:
                        total_running += 1
                    if _has_pending:
                        total_pending += 1
            with _r3d_health_lock:
                health_session_count = len(_r3d_health_cache)
            return {
                "sessions": sessions,
                "total_running": total_running,
                "total_pending": total_pending,
                "shadow_only": True,
                "long_session_health_session_count": health_session_count,
            }


def _r3d_get_health_snapshot(session_id: str) -> dict | None:
    """R-3d: read a per-session long_session_health snapshot from the transient cache."""
    if not session_id:
        return None
    with _r3d_health_lock:
        return dict(_r3d_health_cache[session_id]) if session_id in _r3d_health_cache else None


def _r3d_set_health_snapshot(session_id: str, snapshot: dict) -> None:
    """R-3d: write a per-session long_session_health snapshot into the transient cache."""
    if not session_id or not isinstance(snapshot, dict):
        return
    with _r3d_health_lock:
        _r3d_health_cache[session_id] = snapshot


def _r3d_clear_health_snapshot(session_id: str) -> None:
    """R-3d: remove a per-session long_session_health snapshot from the transient cache."""
    if not session_id:
        return
    with _r3d_health_lock:
        _r3d_health_cache.pop(session_id, None)
