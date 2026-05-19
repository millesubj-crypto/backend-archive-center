"""backend/services/prepare_turn_momentum.py

Prepare-turn momentum packet builder.
Extracted from backend/main.py as part of Step 21.5 accelerated slice.

This module MUST NOT import backend.main.
"""

import json as _json_mp
from datetime import datetime, timezone

from backend.database import SessionLocal
from backend.models import Storyline, PendingThread


def fetch_momentum_packet(sid: str, logger) -> dict | None:
    """M-2c: momentum packet from DB query (no HTTP endpoint equivalent).
    Aligns with get_momentum_packet HTTP endpoint query semantics.
    On failure returns None (supervisor works without momentum).
    """
    try:
        db = SessionLocal()
        try:
            active_sl = (
                db.query(Storyline)
                .filter(
                    Storyline.chat_session_id == sid,
                    Storyline.status == "active",
                    Storyline.suppressed == False,  # noqa: E712
                )
                .order_by(
                    Storyline.pinned.desc(),
                    Storyline.last_turn.desc().nullslast(),
                    Storyline.id.desc(),
                )
                .limit(10)
                .all()
            )
            open_hooks = (
                db.query(PendingThread)
                .filter(
                    PendingThread.chat_session_id == sid,
                    PendingThread.status.in_(["open", "paused"]),
                    PendingThread.suppressed == False,  # noqa: E712
                )
                .order_by(
                    PendingThread.pinned.desc(),
                    PendingThread.last_seen_turn.asc().nullsfirst(),
                    PendingThread.id.asc(),
                )
                .limit(10)
                .all()
            )
        finally:
            db.close()

        if not active_sl and not open_hooks:
            return None

        def _pjl(raw: str | None) -> list:
            if not raw:
                return []
            try:
                v = _json_mp.loads(raw)
                return v if isinstance(v, list) else []
            except Exception:
                return []

        next_pressure: list[dict] = []
        for sl in active_sl:
            tens = _pjl(sl.ongoing_tensions_json)
            if tens:
                label = f"{sl.name}: {tens[0]}"
                next_pressure.append({
                    "label": label, "source_type": "storyline",
                    "source_id": sl.id, "source_name": sl.name, "priority": 0.8,
                })
            if len(next_pressure) >= 3:
                break
        if not next_pressure:
            for sl in active_sl[:2]:
                next_pressure.append({
                    "label": sl.name, "source_type": "storyline",
                    "source_id": sl.id, "source_name": sl.name, "priority": 0.5,
                })

        payoff_candidates: list[dict] = []
        sorted_sl = sorted(active_sl, key=lambda s: (-(s.confidence or 0.0), -(s.evidence_count or 0)))
        for sl in sorted_sl[:2]:
            payoff_candidates.append({
                "label": f"Close: {sl.name}", "source_type": "storyline",
                "source_id": sl.id, "source_name": sl.name, "priority": 0.7,
            })
        for hk in open_hooks[:1]:
            payoff_candidates.append({
                "label": f"Payoff: {hk.title}", "source_type": "hook",
                "source_id": hk.id, "source_name": hk.title, "priority": 0.6,
            })

        tension_to_reuse: list[dict] = []
        for hk in open_hooks[:2]:
            tension_to_reuse.append({
                "label": f"Unresolved: {hk.title}", "source_type": "hook",
                "source_id": hk.id, "source_name": hk.title, "priority": 0.5,
            })

        packet_status = "ready" if (next_pressure or payoff_candidates) else "partial"
        return {
            "packet_status": packet_status,
            "next_pressure": next_pressure,
            "payoff_candidates": payoff_candidates,
            "tension_to_reuse": tension_to_reuse,
            "beats_to_avoid": [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        logger.warning("[prepare-turn] fetch_momentum_packet failed: %s", exc)
        return None
