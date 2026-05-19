"""Deterministic repair sweep for already-written truth-memory rows.

This module is intentionally named by behavior rather than project phase so its
purpose remains obvious even without the surrounding progress/context docs.
It repairs two historical data-integrity problems:

1. Active KG triples that were left open in parallel for the same
   `(chat_session_id, subject, predicate)` key.
2. Stale direct-evidence rows that were captured as pending before the runtime
   auto-verification path existed.
"""

from __future__ import annotations

import json
from collections import defaultdict

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.models import AuditLog, ChatLog, DirectEvidenceRecord, KGTriple


_KG_POLICY_VERSION = "gap1a.v1"
_EVIDENCE_POLICY_VERSION = "gap1b.v1"
_REPAIR_SOURCE = "historical_truth_memory_repair"
_MIN_VERIFICATION_DELAY = 2


def _coerce_turn(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _resolve_kg_turn(row: KGTriple) -> int | None:
    return _coerce_turn(row.source_turn) or _coerce_turn(row.valid_from)


def _collect_session_ids(db: Session, chat_session_id: str | None = None) -> list[str]:
    if isinstance(chat_session_id, str) and chat_session_id.strip():
        return [chat_session_id.strip()]

    session_ids: set[str] = set()
    for model in (ChatLog, KGTriple, DirectEvidenceRecord):
        rows = db.query(model.chat_session_id).distinct().all()
        for (session_id,) in rows:
            sid = str(session_id or "").strip()
            if sid:
                session_ids.add(sid)
    return sorted(session_ids)


def _resolve_latest_turn(db: Session, chat_session_id: str) -> int | None:
    candidates: list[int] = []
    for model, field in (
        (ChatLog, ChatLog.turn_index),
        (DirectEvidenceRecord, DirectEvidenceRecord.source_turn_end),
        (KGTriple, KGTriple.source_turn),
    ):
        value = (
            db.query(func.max(field))
            .filter(model.chat_session_id == chat_session_id)
            .scalar()
        )
        turn = _coerce_turn(value)
        if turn is not None:
            candidates.append(turn)
    if not candidates:
        return None
    return max(candidates)


def _repair_session_kg_triples(db: Session, chat_session_id: str) -> dict:
    result = {
        "groups": 0,
        "rows_closed": 0,
        "audits_written": 0,
        "survivors": 0,
    }

    active_rows = (
        db.query(KGTriple)
        .filter(KGTriple.chat_session_id == chat_session_id)
        .filter(KGTriple.valid_to.is_(None))
        .order_by(KGTriple.subject.asc(), KGTriple.predicate.asc(), KGTriple.id.asc())
        .all()
    )

    grouped: dict[tuple[str, str], list[KGTriple]] = defaultdict(list)
    for row in active_rows:
        grouped[(str(row.subject or ""), str(row.predicate or ""))].append(row)

    for (subject, predicate), rows in grouped.items():
        if len(rows) <= 1:
            continue

        ordered_rows = sorted(
            rows,
            key=lambda item: (
                _resolve_kg_turn(item) if _resolve_kg_turn(item) is not None else -1,
                int(item.id or 0),
            ),
        )
        result["groups"] += 1
        result["survivors"] += 1

        for index, row in enumerate(ordered_rows[:-1]):
            next_row = ordered_rows[index + 1]
            close_turn = _resolve_kg_turn(next_row)
            if close_turn is None:
                close_turn = _resolve_kg_turn(row)
            if close_turn is None:
                continue

            resolution = "duplicate_active_cleanup"
            if str(row.object or "") != str(next_row.object or ""):
                resolution = "historical_conflict_cleanup"

            row.valid_to = close_turn
            db.add(
                AuditLog(
                    event_type="kg_triple_superseded",
                    chat_session_id=chat_session_id,
                    target_type="kg_triple",
                    target_id=row.id,
                    summary=f"Historical truth-memory repair superseded KG triple {subject}:{predicate}",
                    details_json=json.dumps(
                        {
                            "policy_version": _KG_POLICY_VERSION,
                            "repair_mode": "historical_sweep",
                            "resolution": resolution,
                            "subject": subject,
                            "predicate": predicate,
                            "previous_object": row.object,
                            "next_object": next_row.object,
                            "source_turn": close_turn,
                        },
                        ensure_ascii=False,
                    ),
                    source=_REPAIR_SOURCE,
                )
            )
            result["rows_closed"] += 1
            result["audits_written"] += 1

    return result


def _repair_session_direct_evidence(db: Session, chat_session_id: str) -> dict:
    result = {
        "latest_turn": None,
        "reviewed": 0,
        "verified": 0,
        "audits_written": 0,
    }

    latest_turn = _resolve_latest_turn(db, chat_session_id)
    if latest_turn is None or latest_turn < 0:
        return result

    result["latest_turn"] = latest_turn

    candidates = (
        db.query(DirectEvidenceRecord)
        .filter(DirectEvidenceRecord.chat_session_id == chat_session_id)
        .filter(DirectEvidenceRecord.capture_verification == "pending")
        .filter(DirectEvidenceRecord.tombstoned.is_(False))
        .filter(DirectEvidenceRecord.repair_needed.is_(False))
        .filter(DirectEvidenceRecord.superseded_by_id.is_(None))
        .filter(DirectEvidenceRecord.source_turn_end.is_not(None))
        .filter(DirectEvidenceRecord.source_turn_end <= (latest_turn - _MIN_VERIFICATION_DELAY))
        .all()
    )

    for row in candidates:
        result["reviewed"] += 1
        row.capture_verification = "verified"
        row.archive_state = "verified_direct"
        row.repair_needed = False
        row.committed_gate = "finalize"
        db.add(
            AuditLog(
                event_type="evidence_auto_verification",
                chat_session_id=chat_session_id,
                target_type="direct_evidence_record",
                target_id=row.id,
                summary="Historical truth-memory repair auto-verified stale direct evidence",
                details_json=json.dumps(
                    {
                        "policy_version": _EVIDENCE_POLICY_VERSION,
                        "mode": "historical_sweep",
                        "source_turn_end": row.source_turn_end,
                        "verified_at_turn": latest_turn,
                    },
                    ensure_ascii=False,
                ),
                source=_REPAIR_SOURCE,
            )
        )
        result["verified"] += 1
        result["audits_written"] += 1

    return result


def repair_historical_truth_memory(db: Session, *, chat_session_id: str | None = None) -> dict:
    result = {
        "policy_versions": {
            "kg": _KG_POLICY_VERSION,
            "direct_evidence": _EVIDENCE_POLICY_VERSION,
        },
        "source": _REPAIR_SOURCE,
        "sessions_scanned": 0,
        "kg_groups_repaired": 0,
        "kg_rows_closed": 0,
        "direct_evidence_reviewed": 0,
        "direct_evidence_verified": 0,
        "audit_rows_written": 0,
        "session_summaries": [],
    }

    session_ids = _collect_session_ids(db, chat_session_id=chat_session_id)
    for session_id in session_ids:
        kg_result = _repair_session_kg_triples(db, session_id)
        evidence_result = _repair_session_direct_evidence(db, session_id)

        result["sessions_scanned"] += 1
        result["kg_groups_repaired"] += int(kg_result["groups"])
        result["kg_rows_closed"] += int(kg_result["rows_closed"])
        result["direct_evidence_reviewed"] += int(evidence_result["reviewed"])
        result["direct_evidence_verified"] += int(evidence_result["verified"])
        result["audit_rows_written"] += int(kg_result["audits_written"]) + int(evidence_result["audits_written"])

        if kg_result["groups"] or evidence_result["reviewed"]:
            result["session_summaries"].append(
                {
                    "chat_session_id": session_id,
                    "kg_groups_repaired": kg_result["groups"],
                    "kg_rows_closed": kg_result["rows_closed"],
                    "direct_evidence_reviewed": evidence_result["reviewed"],
                    "direct_evidence_verified": evidence_result["verified"],
                    "latest_turn": evidence_result["latest_turn"],
                }
            )

    return result


def run_historical_truth_memory_repair(*, chat_session_id: str | None = None) -> dict:
    db: Session = SessionLocal()
    try:
        result = repair_historical_truth_memory(db, chat_session_id=chat_session_id)
        db.commit()
        return result
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    result = run_historical_truth_memory_repair()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()