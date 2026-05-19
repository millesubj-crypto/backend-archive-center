"""LC1 extracted metric builders (lc1d, lc1g, lc1h, lc1m, lc1c, lc1e, lc1f, lc1i, lc1j, lc1o, lc1n).

Extracted from backend/main.py as part of Step 21.5 Priority 3-c Phase A through Phase E.
These builders avoid prepare_turn bundle ownership; the progression ledger
callback is bound by backend.main at runtime.

Current composition:
- Phase A: lc1d, lc1g, lc1m
- Phase B: lc1h (depends on lc1d + lc1g)
- Phase C unlocker: lc1c (enables lc1e/lc1f movement)
- Phase C: lc1e (depends on lc1c), lc1f (depends on lc1d + lc1e)
- Phase D: lc1i (depends on progression ledger callback), lc1j (depends on lc1i + lc1h)
- Phase E: lc1o (AuditLog-only, no GuidancePlanState dependency), lc1n (ChatLog + EffectiveInputLog + canonical/dense surfaces)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import func

from backend.direct_evidence import (
    is_gate_approved_current_truth,
    is_within_retention_window,
    resolve_retention_importance_tier,
)
from backend.models import AuditLog, CanonicalStateLayer, ChatLog, DirectEvidenceRecord, EffectiveInputLog, EpisodeSummary, ChapterSummary, ArcSummary, SagaDigest, GuidancePlanState, Memory
from backend.turn_contracts import PrepareTurnSettings
from backend.palace.bridge import get_all_canonical_state_layers_for_session

_LC1D_POLICY_VERSION = "lc1d.v1"
_LC1D_NON_SIMILARITY_MAX = 0.2
_LC1D_LONG_TURN_THRESHOLD = 300
_LC1D_ULTRA_LONG_TURN_THRESHOLD = 600
_LC1G_POLICY_VERSION = "lc1g.v1"
_LC1G_STALE_TURN_GAP = 300
_LC1G_LOW_CONFIDENCE_MAX = 0.55
_LC1G_MISMATCH_OVERLAP_MIN = 0.08
_LC1F_POLICY_VERSION = "lc1f.v1"
_LC1F_SHORT_TURN_WINDOW = 120
_LC1F_MID_TURN_WINDOW = 300
_LC1C_POLICY_VERSION = "lc1c.v1"
_LC1C_TURN_WINDOW = 300
_LC1E_POLICY_VERSION = "lc1e.v1"
_LC1H_POLICY_VERSION = "lc1h.v1"
_LC1M_POLICY_VERSION = "lc1m.v1"
_LC1M_MAX_SPLIT_LATENCY_MULTIPLIER = 1.25
_LC1M_SINGLE_ERROR_TOLERANCE = 0.02
_LC1M_SINGLE_ACCURACY_TOLERANCE = 0.01
_LC1I_POLICY_VERSION = "lc1i.v1"
_LC1I_OVERLAP_MIN = 0.2
_LC1J_POLICY_VERSION = "lc1j.v1"
_LC1J_MAX_COVERAGE_DROP = 0.02
_LC1J_MAX_FALSE_NEGATIVE_INCREASE = 0.01
_LC1J_MAX_FALSE_POSITIVE_INCREASE = 0.01

_LC1O_POLICY_VERSION = "lc1o.v1"
_LC1O_MIN_PREVIEW_COVERAGE_RATIO = 0.6
_LC1O_MIN_CALL_SAVED_RATIO = 0.05
_LC1O_MIN_LATENCY_SAVED_RATIO = 0.03

_progression_ledger_builder = None


def bind_lc1_builder_callbacks(*, progression_ledger_builder):
    global _progression_ledger_builder
    _progression_ledger_builder = progression_ledger_builder


def _build_progression_ledger_for_lc1(**kwargs) -> dict:
    if _progression_ledger_builder is None:
        raise RuntimeError("LC1 progression ledger callback is not bound")
    return _progression_ledger_builder(**kwargs)


def _lc1c_text_len(value: object) -> int:
    text = str(value or "")
    return len(text) if text else 0


def _build_lc1c_memory_footprint(*, db, chat_session_id: str) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1C_POLICY_VERSION,
            "turn_window": _LC1C_TURN_WINDOW,
            "status": "off",
            "reason": "session_missing",
            "latest_turn_index": None,
            "canonical_state_chars": 0,
            "dense_summary_chars": 0,
            "live_ledger_chars": 0,
            "total_chars": 0,
        }

    latest_turn = (
        db.query(func.max(ChatLog.turn_index))
        .filter(ChatLog.chat_session_id == sid)
        .scalar()
    )
    latest_turn_index = int(latest_turn) if latest_turn is not None else None
    window_start = (
        max(1, int(latest_turn_index) - _LC1C_TURN_WINDOW + 1)
        if latest_turn_index is not None
        else 1
    )

    canonical_layers = get_all_canonical_state_layers_for_session(chat_session_id=sid)
    canonical_chars = sum(_lc1c_text_len(item.get("content")) for item in canonical_layers)

    episode_rows = (
        db.query(EpisodeSummary)
        .filter(EpisodeSummary.chat_session_id == sid)
        .filter(EpisodeSummary.to_turn >= window_start)
        .all()
    )
    chapter_rows = (
        db.query(ChapterSummary)
        .filter(ChapterSummary.chat_session_id == sid)
        .filter(ChapterSummary.to_turn >= window_start)
        .all()
    )
    arc_rows = (
        db.query(ArcSummary)
        .filter(ArcSummary.chat_session_id == sid)
        .filter(ArcSummary.to_turn >= window_start)
        .all()
    )
    saga_rows = (
        db.query(SagaDigest)
        .filter(SagaDigest.chat_session_id == sid)
        .filter(SagaDigest.to_turn >= window_start)
        .all()
    )

    dense_chars = 0
    for row in episode_rows:
        dense_chars += _lc1c_text_len(row.summary_text)
    for row in chapter_rows:
        dense_chars += _lc1c_text_len(row.summary_text)
        dense_chars += _lc1c_text_len(row.resume_text)
    for row in arc_rows:
        dense_chars += _lc1c_text_len(row.core_conflict)
        dense_chars += _lc1c_text_len(row.arc_resume_text)
    for row in saga_rows:
        dense_chars += _lc1c_text_len(row.saga_summary)
        dense_chars += _lc1c_text_len(row.resume_pack_text)

    cached = (
        db.query(GuidancePlanState)
        .filter(GuidancePlanState.chat_session_id == sid)
        .first()
    )
    if cached and cached.story_plan_json and cached.director_json:
        try:
            sp = json.loads(cached.story_plan_json)
            dr = json.loads(cached.director_json)
        except Exception:
            sp = {}
            dr = {}
        ledger = _build_progression_ledger_for_lc1(
            story_plan_data=sp,
            director_data=dr,
            last_turn=int(cached.last_turn or 0),
            state_status=cached.state_status,
        )
    else:
        ledger = _build_progression_ledger_for_lc1(
            story_plan_data={},
            director_data={},
            last_turn=int(latest_turn_index or 0),
            state_status="empty",
        )

    live_ledger_chars = _lc1c_text_len(json.dumps(ledger, ensure_ascii=False))
    total_chars = canonical_chars + dense_chars + live_ledger_chars

    return {
        "policy_version": _LC1C_POLICY_VERSION,
        "turn_window": _LC1C_TURN_WINDOW,
        "status": "ok",
        "latest_turn_index": latest_turn_index,
        "window_start_turn": window_start,
        "canonical_state_chars": canonical_chars,
        "dense_summary_chars": dense_chars,
        "live_ledger_chars": live_ledger_chars,
        "total_chars": total_chars,
        "counts": {
            "canonical_layers": len(canonical_layers),
            "episodes": len(episode_rows),
            "chapters": len(chapter_rows),
            "arcs": len(arc_rows),
            "sagas": len(saga_rows),
        },
    }


def _build_lc1e_context_budget_comparison(*, db, chat_session_id: str) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1E_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "turn_window": _LC1C_TURN_WINDOW,
            "hybrid_chars": 0,
            "always_on_hypamemory_chars": 0,
            "always_on_recent_window_chars": 0,
            "saved_chars_vs_always_on": 0,
            "saved_ratio_vs_always_on": 0.0,
        }

    hybrid = _build_lc1c_memory_footprint(db=db, chat_session_id=sid)
    hybrid_chars = int(hybrid.get("total_chars") or 0)
    latest_turn_index = hybrid.get("latest_turn_index")
    window_start_turn = hybrid.get("window_start_turn")

    memory_rows = (
        db.query(Memory)
        .filter(Memory.chat_session_id == sid)
        .order_by(Memory.id.asc())
        .all()
    )

    def _memory_row_payload_chars(row: Memory) -> int:
        summary_json = str(row.summary_json or "")
        if summary_json:
            return _lc1c_text_len(summary_json)
        return 0

    always_on_all_chars = sum(_memory_row_payload_chars(row) for row in memory_rows)

    if window_start_turn is None:
        always_on_recent_window_chars = always_on_all_chars
    else:
        always_on_recent_window_chars = sum(
            _memory_row_payload_chars(row)
            for row in memory_rows
            if isinstance(row.turn_index, int) and row.turn_index >= int(window_start_turn)
        )

    saved_chars = max(always_on_all_chars - hybrid_chars, 0)
    saved_ratio = (saved_chars / always_on_all_chars) if always_on_all_chars > 0 else 0.0

    return {
        "policy_version": _LC1E_POLICY_VERSION,
        "status": "ok",
        "turn_window": _LC1C_TURN_WINDOW,
        "latest_turn_index": latest_turn_index,
        "window_start_turn": window_start_turn,
        "hybrid_chars": hybrid_chars,
        "always_on_hypamemory_chars": always_on_all_chars,
        "always_on_recent_window_chars": always_on_recent_window_chars,
        "saved_chars_vs_always_on": saved_chars,
        "saved_ratio_vs_always_on": round(saved_ratio, 4),
        "estimated_tokens": {
            "hybrid": hybrid_chars // 4,
            "always_on_hypamemory": always_on_all_chars // 4,
            "always_on_recent_window": always_on_recent_window_chars // 4,
        },
        "counts": {
            "memory_rows_total": len(memory_rows),
            "memory_rows_recent_window": sum(
                1
                for row in memory_rows
                if window_start_turn is not None
                and isinstance(row.turn_index, int)
                and row.turn_index >= int(window_start_turn)
            ) if window_start_turn is not None else len(memory_rows),
        },
        "comparison_contract": {
            "always_on_mode": "inject_all_memory_summary_json_each_turn",
            "hybrid_mode": "canonical_dense_ledger_char_surface_from_lc1c",
        },
    }


def _lc1d_tokenize(value: object) -> set[str]:
    text = str(value or "").lower()
    tokens = re.findall(r"[a-z0-9가-힣]+", text)
    return {t for t in tokens if len(t) >= 2}


def _lc1d_jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _lc1d_overlap_ratio(candidate_tokens: set[str], layer_tokens: set[str]) -> float:
    if not candidate_tokens or not layer_tokens:
        return 0.0
    return len(candidate_tokens & layer_tokens) / max(len(candidate_tokens), 1)


def _lc1d_scope_label(age_turns: int | None) -> str:
    if age_turns is None:
        return "unknown"
    if age_turns >= _LC1D_ULTRA_LONG_TURN_THRESHOLD:
        return "ultra_long"
    if age_turns >= _LC1D_LONG_TURN_THRESHOLD:
        return "long"
    return "short"


def _lc1d_build_layer_surfaces(*, db, chat_session_id: str, latest_turn_index: int | None) -> dict:
    sid = (chat_session_id or "").strip()

    canonical_layers = get_all_canonical_state_layers_for_session(chat_session_id=sid)
    canonical_text = "\n".join(str(item.get("content") or "") for item in canonical_layers)

    dense_parts: list[str] = []
    for row in db.query(EpisodeSummary).filter(EpisodeSummary.chat_session_id == sid).all():
        dense_parts.append(str(row.summary_text or ""))
    for row in db.query(ChapterSummary).filter(ChapterSummary.chat_session_id == sid).all():
        dense_parts.append(str(row.summary_text or ""))
        dense_parts.append(str(row.resume_text or ""))
    for row in db.query(ArcSummary).filter(ArcSummary.chat_session_id == sid).all():
        dense_parts.append(str(row.core_conflict or ""))
        dense_parts.append(str(row.arc_resume_text or ""))
    for row in db.query(SagaDigest).filter(SagaDigest.chat_session_id == sid).all():
        dense_parts.append(str(row.saga_summary or ""))
        dense_parts.append(str(row.resume_pack_text or ""))
    dense_text = "\n".join(part for part in dense_parts if part)

    cached = (
        db.query(GuidancePlanState)
        .filter(GuidancePlanState.chat_session_id == sid)
        .first()
    )
    if cached and cached.story_plan_json and cached.director_json:
        try:
            sp = json.loads(cached.story_plan_json)
            dr = json.loads(cached.director_json)
        except Exception:
            sp = {}
            dr = {}
        ledger = _build_progression_ledger_for_lc1(
            story_plan_data=sp,
            director_data=dr,
            last_turn=int(cached.last_turn or 0),
            state_status=cached.state_status,
        )
    else:
        ledger = _build_progression_ledger_for_lc1(
            story_plan_data={},
            director_data={},
            last_turn=int(latest_turn_index or 0),
            state_status="empty",
        )
    ledger_text = json.dumps(ledger, ensure_ascii=False)

    return {
        "canonical": {"text": canonical_text, "tokens": _lc1d_tokenize(canonical_text)},
        "dense_summary": {"text": dense_text, "tokens": _lc1d_tokenize(dense_text)},
        "live_ledger": {"text": ledger_text, "tokens": _lc1d_tokenize(ledger_text)},
    }


def _lc1d_is_retained_in_layer(
    *,
    evidence_text: str,
    evidence_tokens: set[str],
    layer_text: str,
    layer_tokens: set[str],
) -> bool:
    normalized_evidence = " ".join(str(evidence_text or "").lower().split())
    normalized_layer = " ".join(str(layer_text or "").lower().split())
    if normalized_evidence and len(normalized_evidence) >= 12 and normalized_evidence in normalized_layer:
        return True
    overlap = _lc1d_overlap_ratio(evidence_tokens, layer_tokens)
    return len(evidence_tokens) >= 2 and overlap >= 0.35


def _build_lc1d_integrity_replay(*, db, chat_session_id: str, replay_query: str | None = None) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1D_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "candidates_total": 0,
            "retained_total": 0,
            "retention_rate": 0.0,
        }

    latest_turn = (
        db.query(func.max(ChatLog.turn_index))
        .filter(ChatLog.chat_session_id == sid)
        .scalar()
    )
    latest_turn_index = int(latest_turn) if latest_turn is not None else None

    query_source = "param"
    query_text = " ".join(str(replay_query or "").split()).strip()
    if not query_text:
        query_source = "latest_user"
        latest_user = (
            db.query(ChatLog)
            .filter(ChatLog.chat_session_id == sid, ChatLog.role == "user")
            .order_by(ChatLog.turn_index.desc(), ChatLog.id.desc())
            .first()
        )
        query_text = " ".join(str((latest_user.content if latest_user else "") or "").split()).strip()
    if not query_text:
        query_source = "empty"

    query_tokens = _lc1d_tokenize(query_text)
    layers = _lc1d_build_layer_surfaces(db=db, chat_session_id=sid, latest_turn_index=latest_turn_index)

    rows = (
        db.query(DirectEvidenceRecord)
        .filter(DirectEvidenceRecord.chat_session_id == sid)
        .order_by(DirectEvidenceRecord.id.desc())
        .all()
    )

    candidates: list[dict] = []
    scanned_rows = 0
    for row in rows:
        scanned_rows += 1
        lineage = {}
        try:
            lineage = json.loads(row.lineage_json or "{}") if row.lineage_json else {}
            if not isinstance(lineage, dict):
                lineage = {}
        except Exception:
            lineage = {}

        if not is_gate_approved_current_truth(
            archive_state=row.archive_state,
            capture_verification=row.capture_verification,
            repair_needed=row.repair_needed,
            committed_gate=row.committed_gate,
            tombstoned=row.tombstoned,
            superseded_by_id=row.superseded_by_id,
        ):
            continue

        if not is_within_retention_window(
            source_turn_end=row.source_turn_end,
            latest_turn_index=latest_turn_index,
            archive_state=row.archive_state,
            capture_verification=row.capture_verification,
            repair_needed=row.repair_needed,
            committed_gate=row.committed_gate,
            tombstoned=row.tombstoned,
            lineage=lineage,
        ):
            continue

        tier = resolve_retention_importance_tier(
            archive_state=row.archive_state,
            capture_verification=row.capture_verification,
            repair_needed=row.repair_needed,
            committed_gate=row.committed_gate,
            tombstoned=row.tombstoned,
            lineage=lineage,
        )
        if tier not in {"critical", "high"}:
            continue

        age_turns = (
            max(int(latest_turn_index) - int(row.source_turn_end), 0)
            if latest_turn_index is not None and row.source_turn_end is not None
            else None
        )
        scope = _lc1d_scope_label(age_turns)
        if scope not in {"long", "ultra_long"}:
            continue

        evidence_text = str(row.evidence_text or "")
        evidence_tokens = _lc1d_tokenize(evidence_text)
        similarity = _lc1d_jaccard_similarity(evidence_tokens, query_tokens)
        non_similar = similarity <= _LC1D_NON_SIMILARITY_MAX
        if not non_similar:
            continue

        retained_canonical = _lc1d_is_retained_in_layer(
            evidence_text=evidence_text,
            evidence_tokens=evidence_tokens,
            layer_text=layers["canonical"]["text"],
            layer_tokens=layers["canonical"]["tokens"],
        )
        retained_dense = _lc1d_is_retained_in_layer(
            evidence_text=evidence_text,
            evidence_tokens=evidence_tokens,
            layer_text=layers["dense_summary"]["text"],
            layer_tokens=layers["dense_summary"]["tokens"],
        )
        retained_ledger = _lc1d_is_retained_in_layer(
            evidence_text=evidence_text,
            evidence_tokens=evidence_tokens,
            layer_text=layers["live_ledger"]["text"],
            layer_tokens=layers["live_ledger"]["tokens"],
        )

        candidates.append({
            "record_id": row.id,
            "source_turn_end": row.source_turn_end,
            "age_turns": age_turns,
            "scope": scope,
            "importance_tier": tier,
            "replay_similarity": round(similarity, 4),
            "retained_in": {
                "canonical": retained_canonical,
                "dense_summary": retained_dense,
                "live_ledger": retained_ledger,
            },
        })

    retained_by_layer = {
        "canonical": sum(1 for c in candidates if c["retained_in"]["canonical"]),
        "dense_summary": sum(1 for c in candidates if c["retained_in"]["dense_summary"]),
        "live_ledger": sum(1 for c in candidates if c["retained_in"]["live_ledger"]),
    }
    retained_total = sum(
        1
        for c in candidates
        if any(c["retained_in"].values())
    )
    candidate_total = len(candidates)
    retention_rate = (retained_total / candidate_total) if candidate_total else 0.0
    scope_counts = {
        "long": sum(1 for c in candidates if c["scope"] == "long"),
        "ultra_long": sum(1 for c in candidates if c["scope"] == "ultra_long"),
    }

    return {
        "policy_version": _LC1D_POLICY_VERSION,
        "status": "ok",
        "latest_turn_index": latest_turn_index,
        "replay_query_source": query_source,
        "replay_query": query_text,
        "replay_non_similarity_max": _LC1D_NON_SIMILARITY_MAX,
        "long_turn_threshold": _LC1D_LONG_TURN_THRESHOLD,
        "ultra_long_turn_threshold": _LC1D_ULTRA_LONG_TURN_THRESHOLD,
        "scanned_direct_evidence_rows": scanned_rows,
        "candidates_total": candidate_total,
        "retained_total": retained_total,
        "gaps_total": max(candidate_total - retained_total, 0),
        "retention_rate": round(retention_rate, 4),
        "retained_by_layer": retained_by_layer,
        "scope_counts": scope_counts,
        "candidate_examples": candidates[:5],
    }


def _build_lc1g_accuracy_replay(*, db, chat_session_id: str) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1G_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "stale_contamination_rate": 0.0,
            "current_state_mismatch_rate": 0.0,
            "unverified_fact_promotion_count": 0,
        }

    latest_turn = (
        db.query(func.max(ChatLog.turn_index))
        .filter(ChatLog.chat_session_id == sid)
        .scalar()
    )
    latest_turn_index = int(latest_turn) if latest_turn is not None else None
    mid_start = (
        max(1, int(latest_turn_index) - _LC1F_MID_TURN_WINDOW + 1)
        if latest_turn_index is not None
        else 1
    )

    canonical_layers = get_all_canonical_state_layers_for_session(chat_session_id=sid)
    canonical_total = len(canonical_layers)

    stale_suspects = 0
    unverified_promotions = 0

    rows = (
        db.query(DirectEvidenceRecord)
        .filter(DirectEvidenceRecord.chat_session_id == sid)
        .filter(DirectEvidenceRecord.source_turn_end >= mid_start)
        .all()
    )
    approved_texts: list[str] = []
    for row in rows:
        lineage = {}
        try:
            lineage = json.loads(row.lineage_json or "{}") if row.lineage_json else {}
            if not isinstance(lineage, dict):
                lineage = {}
        except Exception:
            lineage = {}
        if not is_gate_approved_current_truth(
            archive_state=row.archive_state,
            capture_verification=row.capture_verification,
            repair_needed=row.repair_needed,
            committed_gate=row.committed_gate,
            tombstoned=row.tombstoned,
            superseded_by_id=row.superseded_by_id,
        ):
            continue
        if not is_within_retention_window(
            source_turn_end=row.source_turn_end,
            latest_turn_index=latest_turn_index,
            archive_state=row.archive_state,
            capture_verification=row.capture_verification,
            repair_needed=row.repair_needed,
            committed_gate=row.committed_gate,
            tombstoned=row.tombstoned,
            lineage=lineage,
        ):
            continue
        approved_texts.append(str(row.evidence_text or ""))

    approved_tokens = _lc1d_tokenize("\n".join(approved_texts))
    mismatch_eligible = 0
    mismatch_count = 0

    for layer in canonical_layers:
        source_turn = layer.get("source_turn")
        last_verified_turn = layer.get("last_verified_turn")
        confidence_raw = layer.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except Exception:
            confidence = None

        if last_verified_turn is None or (confidence is not None and confidence < 0.5):
            unverified_promotions += 1

        stale_age = (
            max(int(latest_turn_index) - int(source_turn), 0)
            if latest_turn_index is not None and isinstance(source_turn, int)
            else None
        )
        if stale_age is not None and stale_age >= _LC1G_STALE_TURN_GAP:
            if (confidence is None or confidence <= _LC1G_LOW_CONFIDENCE_MAX) or last_verified_turn is None:
                stale_suspects += 1

        layer_tokens = _lc1d_tokenize(layer.get("content") or "")
        if layer_tokens and approved_tokens:
            mismatch_eligible += 1
            overlap = _lc1d_overlap_ratio(layer_tokens, approved_tokens)
            if overlap < _LC1G_MISMATCH_OVERLAP_MIN:
                mismatch_count += 1

    stale_rate = (stale_suspects / canonical_total) if canonical_total else 0.0
    mismatch_rate = (mismatch_count / mismatch_eligible) if mismatch_eligible else 0.0

    return {
        "policy_version": _LC1G_POLICY_VERSION,
        "status": "ok",
        "latest_turn_index": latest_turn_index,
        "stale_turn_gap": _LC1G_STALE_TURN_GAP,
        "low_confidence_max": _LC1G_LOW_CONFIDENCE_MAX,
        "mismatch_overlap_min": _LC1G_MISMATCH_OVERLAP_MIN,
        "stale_contamination_rate": round(stale_rate, 4),
        "current_state_mismatch_rate": round(mismatch_rate, 4),
        "unverified_fact_promotion_count": unverified_promotions,
        "counts": {
            "canonical_total": canonical_total,
            "stale_suspects": stale_suspects,
            "mismatch_eligible": mismatch_eligible,
            "mismatch_count": mismatch_count,
            "approved_direct_evidence_recent": len(approved_texts),
        },
    }


def _build_lc1h_forgetting_hallucination_replay(*, db, chat_session_id: str) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1H_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "false_negative_rate": 0.0,
            "false_positive_rate": 0.0,
            "important_fact_missing_rate": 0.0,
            "current_fact_hallucination_rate": 0.0,
        }

    integrity = _build_lc1d_integrity_replay(db=db, chat_session_id=sid)
    accuracy = _build_lc1g_accuracy_replay(db=db, chat_session_id=sid)

    candidates_total = int(integrity.get("candidates_total") or 0)
    gaps_total = int(integrity.get("gaps_total") or 0)
    false_negative_rate = (gaps_total / candidates_total) if candidates_total > 0 else 0.0

    canonical_total = int((accuracy.get("counts") or {}).get("canonical_total") or 0)
    mismatch_count = int((accuracy.get("counts") or {}).get("mismatch_count") or 0)
    unverified_promotions = int(accuracy.get("unverified_fact_promotion_count") or 0)
    # Conservative upper-bound for hallucination candidates.
    hallucination_count = min(canonical_total, mismatch_count + unverified_promotions)
    false_positive_rate = (hallucination_count / canonical_total) if canonical_total > 0 else 0.0

    return {
        "policy_version": _LC1H_POLICY_VERSION,
        "status": "ok",
        "important_fact_missing_rate": round(false_negative_rate, 4),
        "current_fact_hallucination_rate": round(false_positive_rate, 4),
        "false_negative_rate": round(false_negative_rate, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "counts": {
            "important_candidates_total": candidates_total,
            "important_missing_total": gaps_total,
            "canonical_total": canonical_total,
            "hallucination_candidate_total": hallucination_count,
            "mismatch_count": mismatch_count,
            "unverified_fact_promotion_count": unverified_promotions,
        },
        "sources": {
            "integrity_replay_policy_version": integrity.get("policy_version"),
            "accuracy_replay_policy_version": accuracy.get("policy_version"),
        },
    }


def _build_lc1m_single_call_vs_split_comparison(
    *,
    db,
    chat_session_id: str,
    candidate_single_call_latency_proxy_ms: float | None = None,
    candidate_single_call_error_rate: float | None = None,
    candidate_single_call_accuracy_rate: float | None = None,
    max_sample_rows: int = 120,
) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1M_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "gate_result": "off",
        }

    limit_rows = max(1, min(int(max_sample_rows or 120), 500))
    audit_rows = (
        db.query(AuditLog)
        .filter(
            AuditLog.chat_session_id == sid,
            AuditLog.event_type == "critic_pipeline_trace",
        )
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        .limit(limit_rows)
        .all()
    )

    if not audit_rows:
        return {
            "policy_version": _LC1M_POLICY_VERSION,
            "status": "off",
            "reason": "pipeline_trace_missing",
            "gate_result": "off",
            "sample_count": 0,
        }

    def _safe_details(raw: str | None) -> dict:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _clamp01(value: float) -> float:
        return min(max(float(value), 0.0), 1.0)

    traces = []
    for row in reversed(audit_rows):
        details = _safe_details(row.details_json)
        if details:
            traces.append(details)

    if not traces:
        return {
            "policy_version": _LC1M_POLICY_VERSION,
            "status": "off",
            "reason": "pipeline_trace_payload_missing",
            "gate_result": "off",
            "sample_count": 0,
        }

    sample_count = len(traces)
    ok_count = 0
    degraded_count = 0
    failed_count = 0
    skipped_count = 0
    partial_count = 0
    warning_total = 0
    component_total_sum = 0
    component_failed_sum = 0
    compactor_deferred_count = 0
    pipeline_versions = set()

    for trace in traces:
        pipeline_versions.add(str(trace.get("pipeline_version") or "unknown"))
        status = str(trace.get("status") or "").strip().lower()
        if status == "ok":
            ok_count += 1
        elif status == "degraded":
            degraded_count += 1
        elif status == "failed":
            failed_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            partial_count += 1

        warnings = trace.get("warnings")
        if isinstance(warnings, list):
            warning_total += len(warnings)

        stages = trace.get("stages") if isinstance(trace.get("stages"), dict) else {}
        reducer = stages.get("deterministic_reducer") if isinstance(stages.get("deterministic_reducer"), dict) else {}
        compactor = stages.get("summary_compactor_background") if isinstance(stages.get("summary_compactor_background"), dict) else {}
        if str(compactor.get("status") or "") == "deferred_background":
            compactor_deferred_count += 1

        component_counts = reducer.get("component_counts") if isinstance(reducer.get("component_counts"), dict) else {}
        component_total_sum += int(component_counts.get("total") or 0)
        component_failed_sum += int(component_counts.get("failed") or 0)

    degraded_or_failed = degraded_count + failed_count
    split_error_rate = _clamp01(degraded_or_failed / max(sample_count, 1))

    base_success = (ok_count + (0.5 * partial_count)) / max(sample_count, 1)
    component_penalty = (component_failed_sum / max(component_total_sum, 1)) * 0.2
    split_accuracy_rate = _clamp01(base_success - component_penalty)

    avg_components = component_total_sum / max(sample_count, 1)
    avg_warning_count = warning_total / max(sample_count, 1)
    compactor_deferred_ratio = compactor_deferred_count / max(sample_count, 1)
    split_latency_proxy_ms = round(
        150.0
        + (avg_components * 14.0)
        + (avg_warning_count * 8.0)
        + (compactor_deferred_ratio * 12.0),
        3,
    )

    coupling_penalty = (
        (component_failed_sum / max(component_total_sum, 1)) * 0.25
        + (compactor_deferred_ratio * 0.08)
    )
    baseline_single_call_error_rate = _clamp01(split_error_rate + coupling_penalty)
    baseline_single_call_accuracy_rate = _clamp01(split_accuracy_rate - coupling_penalty)
    baseline_single_call_latency_proxy_ms = round(
        max(split_latency_proxy_ms * 0.92, 60.0)
        + (avg_components * 6.0)
        + (compactor_deferred_ratio * 20.0),
        3,
    )

    single_latency_proxy_ms = (
        round(max(float(candidate_single_call_latency_proxy_ms), 1.0), 3)
        if candidate_single_call_latency_proxy_ms is not None
        else baseline_single_call_latency_proxy_ms
    )
    single_error_rate = (
        _clamp01(float(candidate_single_call_error_rate))
        if candidate_single_call_error_rate is not None
        else baseline_single_call_error_rate
    )
    single_accuracy_rate = (
        _clamp01(float(candidate_single_call_accuracy_rate))
        if candidate_single_call_accuracy_rate is not None
        else baseline_single_call_accuracy_rate
    )

    checks = [
        {
            "name": "split_error_rate_not_worse_than_single",
            "passed": split_error_rate <= (single_error_rate + _LC1M_SINGLE_ERROR_TOLERANCE),
            "detail": (
                f"split_error_rate={round(split_error_rate, 4)}, "
                f"single_error_rate={round(single_error_rate, 4)}, "
                f"tolerance={_LC1M_SINGLE_ERROR_TOLERANCE}"
            ),
        },
        {
            "name": "split_accuracy_not_worse_than_single",
            "passed": split_accuracy_rate + _LC1M_SINGLE_ACCURACY_TOLERANCE >= single_accuracy_rate,
            "detail": (
                f"split_accuracy_rate={round(split_accuracy_rate, 4)}, "
                f"single_accuracy_rate={round(single_accuracy_rate, 4)}, "
                f"tolerance={_LC1M_SINGLE_ACCURACY_TOLERANCE}"
            ),
        },
        {
            "name": "split_latency_within_single_call_ceiling",
            "passed": split_latency_proxy_ms <= (single_latency_proxy_ms * _LC1M_MAX_SPLIT_LATENCY_MULTIPLIER),
            "detail": (
                f"split_latency_proxy_ms={split_latency_proxy_ms}, "
                f"single_latency_proxy_ms={single_latency_proxy_ms}, "
                f"max_multiplier={_LC1M_MAX_SPLIT_LATENCY_MULTIPLIER}"
            ),
        },
    ]
    failed = [item["name"] for item in checks if not item.get("passed")]
    gate_pass = len(failed) == 0

    return {
        "policy_version": _LC1M_POLICY_VERSION,
        "status": "ok",
        "gate_result": "pass" if gate_pass else "warn",
        "comparison_mode": "split_pipeline_trace_vs_single_call_proxy",
        "sample_count": sample_count,
        "split_pipeline": {
            "pipeline_versions": sorted(pipeline_versions),
            "all_in_single_call_observed": False,
            "latency_proxy_ms": split_latency_proxy_ms,
            "error_rate": round(split_error_rate, 4),
            "accuracy_rate": round(split_accuracy_rate, 4),
            "status_counts": {
                "ok": ok_count,
                "degraded": degraded_count,
                "failed": failed_count,
                "skipped": skipped_count,
                "partial": partial_count,
            },
            "reducer_component_counts": {
                "total": component_total_sum,
                "failed": component_failed_sum,
                "avg_total_per_trace": round(avg_components, 3),
            },
            "warning_count_total": warning_total,
            "compactor_deferred_count": compactor_deferred_count,
        },
        "single_call_proxy": {
            "latency_proxy_ms": round(single_latency_proxy_ms, 3),
            "error_rate": round(single_error_rate, 4),
            "accuracy_rate": round(single_accuracy_rate, 4),
            "candidate_override_applied": {
                "latency_proxy_ms": candidate_single_call_latency_proxy_ms is not None,
                "error_rate": candidate_single_call_error_rate is not None,
                "accuracy_rate": candidate_single_call_accuracy_rate is not None,
            },
            "proxy_note": "single-call values are deterministic proxy estimates derived from split trace coupling signals",
        },
        "deltas": {
            "latency_proxy_ms": round(split_latency_proxy_ms - single_latency_proxy_ms, 3),
            "error_rate": round(split_error_rate - single_error_rate, 4),
            "accuracy_rate": round(split_accuracy_rate - single_accuracy_rate, 4),
        },
        "thresholds": {
            "max_split_latency_multiplier_vs_single": _LC1M_MAX_SPLIT_LATENCY_MULTIPLIER,
            "single_error_tolerance": _LC1M_SINGLE_ERROR_TOLERANCE,
            "single_accuracy_tolerance": _LC1M_SINGLE_ACCURACY_TOLERANCE,
        },
        "checks": checks,
        "failed_checks": failed,
        "sources": {
            "audit_event_type": "critic_pipeline_trace",
            "pipeline_contract_ref": "ea1j.v1",
        },
    }


def _build_lc1f_non_regression_check(*, db, chat_session_id: str) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1F_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
        }

    latest_turn = (
        db.query(func.max(ChatLog.turn_index))
        .filter(ChatLog.chat_session_id == sid)
        .scalar()
    )
    latest_turn_index = int(latest_turn) if latest_turn is not None else None

    short_start = (
        max(1, int(latest_turn_index) - _LC1F_SHORT_TURN_WINDOW + 1)
        if latest_turn_index is not None
        else 1
    )
    mid_start = (
        max(1, int(latest_turn_index) - _LC1F_MID_TURN_WINDOW + 1)
        if latest_turn_index is not None
        else 1
    )

    canonical_short = (
        db.query(func.count(CanonicalStateLayer.id))
        .filter(CanonicalStateLayer.chat_session_id == sid)
        .filter(CanonicalStateLayer.turn_index >= short_start)
        .scalar()
    ) or 0
    canonical_mid = (
        db.query(func.count(CanonicalStateLayer.id))
        .filter(CanonicalStateLayer.chat_session_id == sid)
        .filter(CanonicalStateLayer.turn_index >= mid_start)
        .scalar()
    ) or 0

    dense_short = {
        "episodes": (
            db.query(func.count(EpisodeSummary.id))
            .filter(EpisodeSummary.chat_session_id == sid)
            .filter(EpisodeSummary.to_turn >= short_start)
            .scalar()
        ) or 0,
        "chapters": (
            db.query(func.count(ChapterSummary.id))
            .filter(ChapterSummary.chat_session_id == sid)
            .filter(ChapterSummary.to_turn >= short_start)
            .scalar()
        ) or 0,
        "arcs": (
            db.query(func.count(ArcSummary.id))
            .filter(ArcSummary.chat_session_id == sid)
            .filter(ArcSummary.to_turn >= short_start)
            .scalar()
        ) or 0,
        "sagas": (
            db.query(func.count(SagaDigest.id))
            .filter(SagaDigest.chat_session_id == sid)
            .filter(SagaDigest.to_turn >= short_start)
            .scalar()
        ) or 0,
    }
    dense_mid = {
        "episodes": (
            db.query(func.count(EpisodeSummary.id))
            .filter(EpisodeSummary.chat_session_id == sid)
            .filter(EpisodeSummary.to_turn >= mid_start)
            .scalar()
        ) or 0,
        "chapters": (
            db.query(func.count(ChapterSummary.id))
            .filter(ChapterSummary.chat_session_id == sid)
            .filter(ChapterSummary.to_turn >= mid_start)
            .scalar()
        ) or 0,
        "arcs": (
            db.query(func.count(ArcSummary.id))
            .filter(ArcSummary.chat_session_id == sid)
            .filter(ArcSummary.to_turn >= mid_start)
            .scalar()
        ) or 0,
        "sagas": (
            db.query(func.count(SagaDigest.id))
            .filter(SagaDigest.chat_session_id == sid)
            .filter(SagaDigest.to_turn >= mid_start)
            .scalar()
        ) or 0,
    }

    def _approved_within(turn_start: int) -> int:
        rows = (
            db.query(DirectEvidenceRecord)
            .filter(DirectEvidenceRecord.chat_session_id == sid)
            .filter(DirectEvidenceRecord.source_turn_end >= turn_start)
            .all()
        )
        count = 0
        for row in rows:
            lineage = {}
            try:
                lineage = json.loads(row.lineage_json or "{}") if row.lineage_json else {}
                if not isinstance(lineage, dict):
                    lineage = {}
            except Exception:
                lineage = {}
            if not is_gate_approved_current_truth(
                archive_state=row.archive_state,
                capture_verification=row.capture_verification,
                repair_needed=row.repair_needed,
                committed_gate=row.committed_gate,
                tombstoned=row.tombstoned,
                superseded_by_id=row.superseded_by_id,
            ):
                continue
            if not is_within_retention_window(
                source_turn_end=row.source_turn_end,
                latest_turn_index=latest_turn_index,
                archive_state=row.archive_state,
                capture_verification=row.capture_verification,
                repair_needed=row.repair_needed,
                committed_gate=row.committed_gate,
                tombstoned=row.tombstoned,
                lineage=lineage,
            ):
                continue
            count += 1
        return count

    approved_short = _approved_within(short_start)
    approved_mid = _approved_within(mid_start)

    replay = _build_lc1d_integrity_replay(db=db, chat_session_id=sid)
    budget = _build_lc1e_context_budget_comparison(db=db, chat_session_id=sid)

    checks = [
        {
            "name": "short_canonical_presence",
            "passed": canonical_short > 0,
            "detail": f"canonical_short={canonical_short}",
        },
        {
            "name": "mid_canonical_presence",
            "passed": canonical_mid > 0,
            "detail": f"canonical_mid={canonical_mid}",
        },
        {
            "name": "short_dense_presence",
            "passed": sum(dense_short.values()) > 0,
            "detail": f"dense_short_total={sum(dense_short.values())}",
        },
        {
            "name": "mid_dense_presence",
            "passed": sum(dense_mid.values()) > 0,
            "detail": f"dense_mid_total={sum(dense_mid.values())}",
        },
        {
            "name": "short_direct_evidence_presence",
            "passed": approved_short > 0,
            "detail": f"approved_short={approved_short}",
        },
        {
            "name": "mid_direct_evidence_presence",
            "passed": approved_mid > 0,
            "detail": f"approved_mid={approved_mid}",
        },
        {
            "name": "budget_not_worse_than_always_on",
            "passed": int(budget.get("saved_chars_vs_always_on") or 0) >= 0,
            "detail": f"saved_chars_vs_always_on={int(budget.get('saved_chars_vs_always_on') or 0)}",
        },
        {
            "name": "integrity_replay_not_zero_when_candidates_exist",
            "passed": (
                int(replay.get("candidates_total") or 0) == 0
                or int(replay.get("retained_total") or 0) > 0
            ),
            "detail": (
                f"candidates={int(replay.get('candidates_total') or 0)},"
                f" retained={int(replay.get('retained_total') or 0)}"
            ),
        },
    ]

    failed = [c for c in checks if not c["passed"]]
    overall = "pass" if not failed else "warn"

    return {
        "policy_version": _LC1F_POLICY_VERSION,
        "status": "ok",
        "overall_result": overall,
        "latest_turn_index": latest_turn_index,
        "windows": {
            "short_turns": _LC1F_SHORT_TURN_WINDOW,
            "mid_turns": _LC1F_MID_TURN_WINDOW,
            "short_start_turn": short_start,
            "mid_start_turn": mid_start,
        },
        "signals": {
            "canonical": {
                "short_count": canonical_short,
                "mid_count": canonical_mid,
            },
            "dense_summary": {
                "short": dense_short,
                "mid": dense_mid,
            },
            "direct_evidence": {
                "short_gate_approved": approved_short,
                "mid_gate_approved": approved_mid,
            },
            "integrity_replay": {
                "candidates_total": int(replay.get("candidates_total") or 0),
                "retained_total": int(replay.get("retained_total") or 0),
                "retention_rate": float(replay.get("retention_rate") or 0.0),
            },
            "context_budget": {
                "hybrid_chars": int(budget.get("hybrid_chars") or 0),
                "always_on_hypamemory_chars": int(budget.get("always_on_hypamemory_chars") or 0),
                "saved_chars_vs_always_on": int(budget.get("saved_chars_vs_always_on") or 0),
                "saved_ratio_vs_always_on": float(budget.get("saved_ratio_vs_always_on") or 0.0),
            },
        },
        "checks": checks,
        "failed_checks": [c["name"] for c in failed],
    }


def _build_lc1i_ablation_compare(*, db, chat_session_id: str) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1I_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "full_coverage_rate": 0.0,
            "ablations": {},
        }

    latest_turn = (
        db.query(func.max(ChatLog.turn_index))
        .filter(ChatLog.chat_session_id == sid)
        .scalar()
    )
    latest_turn_index = int(latest_turn) if latest_turn is not None else None

    # relationship-v2 proxy: canonical relationship_state layers
    canonical_layers = get_all_canonical_state_layers_for_session(chat_session_id=sid)
    relationship_layers = [
        layer for layer in canonical_layers
        if str(layer.get("layer_type") or "") == "relationship_state"
    ]
    relationship_text = "\n".join(str(layer.get("content") or "") for layer in relationship_layers)

    # ledger/world pressure proxy from progression_ledger
    cached = (
        db.query(GuidancePlanState)
        .filter(GuidancePlanState.chat_session_id == sid)
        .first()
    )
    if cached and cached.story_plan_json and cached.director_json:
        try:
            sp = json.loads(cached.story_plan_json)
            dr = json.loads(cached.director_json)
        except Exception:
            sp = {}
            dr = {}
        progression = _build_progression_ledger_for_lc1(
            story_plan_data=sp,
            director_data=dr,
            last_turn=int(cached.last_turn or 0),
            state_status=cached.state_status,
        )
    else:
        progression = _build_progression_ledger_for_lc1(
            story_plan_data={},
            director_data={},
            last_turn=int(latest_turn_index or 0),
            state_status="empty",
        )

    ledger_payload = {
        "unresolved_tensions": progression.get("unresolved_tensions") or [],
        "consequences": progression.get("consequences") or [],
        "payoffs": progression.get("payoffs") or [],
    }
    world_pressure_payload = {
        "world_pressure": progression.get("world_pressure") or {},
        "scene_deltas": progression.get("scene_deltas") or [],
    }
    ledger_text = json.dumps(ledger_payload, ensure_ascii=False)
    world_pressure_text = json.dumps(world_pressure_payload, ensure_ascii=False)

    approved_rows = (
        db.query(DirectEvidenceRecord)
        .filter(DirectEvidenceRecord.chat_session_id == sid)
        .order_by(DirectEvidenceRecord.id.desc())
        .all()
    )

    evidence_texts: list[str] = []
    for row in approved_rows:
        lineage = {}
        try:
            lineage = json.loads(row.lineage_json or "{}") if row.lineage_json else {}
            if not isinstance(lineage, dict):
                lineage = {}
        except Exception:
            lineage = {}
        if not is_gate_approved_current_truth(
            archive_state=row.archive_state,
            capture_verification=row.capture_verification,
            repair_needed=row.repair_needed,
            committed_gate=row.committed_gate,
            tombstoned=row.tombstoned,
            superseded_by_id=row.superseded_by_id,
        ):
            continue
        if not is_within_retention_window(
            source_turn_end=row.source_turn_end,
            latest_turn_index=latest_turn_index,
            archive_state=row.archive_state,
            capture_verification=row.capture_verification,
            repair_needed=row.repair_needed,
            committed_gate=row.committed_gate,
            tombstoned=row.tombstoned,
            lineage=lineage,
        ):
            continue
        evidence_texts.append(str(row.evidence_text or ""))

    evidence_tokens = [_lc1d_tokenize(text) for text in evidence_texts if text]

    def _coverage_for(components: list[str]) -> tuple[int, float]:
        combined_text = "\n".join(part for part in components if part)
        combined_tokens = _lc1d_tokenize(combined_text)
        if not evidence_tokens:
            return 0, 0.0
        hit = 0
        for tokens in evidence_tokens:
            if not tokens:
                continue
            if _lc1d_overlap_ratio(tokens, combined_tokens) >= _LC1I_OVERLAP_MIN:
                hit += 1
        rate = (hit / len(evidence_tokens)) if evidence_tokens else 0.0
        return hit, rate

    full_hit, full_rate = _coverage_for([relationship_text, ledger_text, world_pressure_text])
    wo_rel_hit, wo_rel_rate = _coverage_for([ledger_text, world_pressure_text])
    wo_ledger_hit, wo_ledger_rate = _coverage_for([relationship_text, world_pressure_text])
    wo_wp_hit, wo_wp_rate = _coverage_for([relationship_text, ledger_text])

    return {
        "policy_version": _LC1I_POLICY_VERSION,
        "status": "ok",
        "overlap_min": _LC1I_OVERLAP_MIN,
        "evidence_count": len(evidence_tokens),
        "full_hit_count": full_hit,
        "full_coverage_rate": round(full_rate, 4),
        "ablations": {
            "without_relationship_v2": {
                "hit_count": wo_rel_hit,
                "coverage_rate": round(wo_rel_rate, 4),
                "coverage_drop": round(max(full_rate - wo_rel_rate, 0.0), 4),
            },
            "without_ledger": {
                "hit_count": wo_ledger_hit,
                "coverage_rate": round(wo_ledger_rate, 4),
                "coverage_drop": round(max(full_rate - wo_ledger_rate, 0.0), 4),
            },
            "without_world_pressure": {
                "hit_count": wo_wp_hit,
                "coverage_rate": round(wo_wp_rate, 4),
                "coverage_drop": round(max(full_rate - wo_wp_rate, 0.0), 4),
            },
        },
    }


def _build_lc1j_imported_idea_gate(
    *,
    db,
    chat_session_id: str,
    candidate_full_coverage_rate: float | None = None,
    candidate_false_negative_rate: float | None = None,
    candidate_false_positive_rate: float | None = None,
) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1J_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "gate_result": "off",
            "degradation_detected": False,
        }

    ablation = _build_lc1i_ablation_compare(db=db, chat_session_id=sid)
    replay = _build_lc1h_forgetting_hallucination_replay(db=db, chat_session_id=sid)

    baseline_full_coverage_rate = float(ablation.get("full_coverage_rate") or 0.0)
    baseline_false_negative_rate = float(replay.get("false_negative_rate") or 0.0)
    baseline_false_positive_rate = float(replay.get("false_positive_rate") or 0.0)

    def _candidate_or_baseline(candidate: float | None, baseline: float) -> float:
        if candidate is None:
            return baseline
        return min(max(float(candidate), 0.0), 1.0)

    candidate_full = _candidate_or_baseline(candidate_full_coverage_rate, baseline_full_coverage_rate)
    candidate_fn = _candidate_or_baseline(candidate_false_negative_rate, baseline_false_negative_rate)
    candidate_fp = _candidate_or_baseline(candidate_false_positive_rate, baseline_false_positive_rate)

    coverage_drop = max(baseline_full_coverage_rate - candidate_full, 0.0)
    false_negative_increase = max(candidate_fn - baseline_false_negative_rate, 0.0)
    false_positive_increase = max(candidate_fp - baseline_false_positive_rate, 0.0)

    checks = [
        {
            "name": "full_coverage_not_degraded",
            "passed": coverage_drop <= _LC1J_MAX_COVERAGE_DROP,
            "detail": (
                f"coverage_drop={round(coverage_drop, 4)}"
                f" <= max_drop={_LC1J_MAX_COVERAGE_DROP}"
            ),
        },
        {
            "name": "false_negative_not_degraded",
            "passed": false_negative_increase <= _LC1J_MAX_FALSE_NEGATIVE_INCREASE,
            "detail": (
                f"false_negative_increase={round(false_negative_increase, 4)}"
                f" <= max_increase={_LC1J_MAX_FALSE_NEGATIVE_INCREASE}"
            ),
        },
        {
            "name": "false_positive_not_degraded",
            "passed": false_positive_increase <= _LC1J_MAX_FALSE_POSITIVE_INCREASE,
            "detail": (
                f"false_positive_increase={round(false_positive_increase, 4)}"
                f" <= max_increase={_LC1J_MAX_FALSE_POSITIVE_INCREASE}"
            ),
        },
    ]

    failed = [item["name"] for item in checks if not item.get("passed")]
    gate_pass = len(failed) == 0

    return {
        "policy_version": _LC1J_POLICY_VERSION,
        "status": "ok",
        "gate_result": "pass" if gate_pass else "warn",
        "degradation_detected": not gate_pass,
        "before_after_mode": "baseline_surface_vs_candidate_surface",
        "thresholds": {
            "max_coverage_drop": _LC1J_MAX_COVERAGE_DROP,
            "max_false_negative_increase": _LC1J_MAX_FALSE_NEGATIVE_INCREASE,
            "max_false_positive_increase": _LC1J_MAX_FALSE_POSITIVE_INCREASE,
        },
        "baseline": {
            "full_coverage_rate": round(baseline_full_coverage_rate, 4),
            "false_negative_rate": round(baseline_false_negative_rate, 4),
            "false_positive_rate": round(baseline_false_positive_rate, 4),
        },
        "candidate": {
            "full_coverage_rate": round(candidate_full, 4),
            "false_negative_rate": round(candidate_fn, 4),
            "false_positive_rate": round(candidate_fp, 4),
            "candidate_overrides_applied": {
                "full_coverage_rate": candidate_full_coverage_rate is not None,
                "false_negative_rate": candidate_false_negative_rate is not None,
                "false_positive_rate": candidate_false_positive_rate is not None,
            },
        },
        "deltas": {
            "coverage_drop": round(coverage_drop, 4),
            "false_negative_increase": round(false_negative_increase, 4),
            "false_positive_increase": round(false_positive_increase, 4),
        },
        "checks": checks,
        "failed_checks": failed,
        "sources": {
            "ablation_compare_policy_version": ablation.get("policy_version"),
            "forgetting_hallucination_policy_version": replay.get("policy_version"),
            "evidence_count": int(ablation.get("evidence_count") or 0),
        },
    }


def _build_lc1o_deterministic_preview_ledger_lightweight_validation(
    *,
    db,
    chat_session_id: str,
    candidate_preview_coverage_ratio: float | None = None,
    candidate_ledger_mode: str | None = None,
    candidate_observed_critic_calls: int | None = None,
    candidate_without_deterministic_critic_calls: int | None = None,
    candidate_observed_latency_proxy_ms: float | None = None,
    candidate_without_deterministic_latency_proxy_ms: float | None = None,
    max_sample_rows: int = 160,
    narrative_bundle: dict | None = None,
) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1O_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "gate_result": "off",
        }

    limit_rows = max(20, min(int(max_sample_rows or 160), 600))
    pipeline_rows = (
        db.query(AuditLog)
        .filter(
            AuditLog.chat_session_id == sid,
            AuditLog.event_type == "critic_pipeline_trace",
        )
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        .limit(limit_rows)
        .all()
    )
    preview_rows = (
        db.query(AuditLog)
        .filter(
            AuditLog.chat_session_id == sid,
            AuditLog.event_type == "critic_preview_trace",
        )
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        .limit(limit_rows)
        .all()
    )

    if not pipeline_rows:
        return {
            "policy_version": _LC1O_POLICY_VERSION,
            "status": "off",
            "reason": "pipeline_trace_missing",
            "gate_result": "off",
            "sample_count": 0,
        }

    def _safe_details(raw: str | None) -> dict:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    pipeline_details = [_safe_details(row.details_json) for row in reversed(pipeline_rows)]
    preview_details = [_safe_details(row.details_json) for row in reversed(preview_rows)]

    pipeline_turns: list[int] = []
    degraded_count = 0
    warning_count = 0
    for item in pipeline_details:
        turn = item.get("turn_index")
        if isinstance(turn, int):
            pipeline_turns.append(turn)
        status = str(item.get("status") or "").strip().lower()
        if status in {"degraded", "failed"}:
            degraded_count += 1
        if isinstance(item.get("warnings"), list):
            warning_count += len(item.get("warnings") or [])

    preview_turns_ok: set[int] = set()
    preview_ok_count = 0
    for item in preview_details:
        if str(item.get("status") or "").strip().lower() == "ok":
            preview_ok_count += 1
        turn = item.get("turn_index")
        if isinstance(turn, int) and str(item.get("status") or "").strip().lower() == "ok":
            preview_turns_ok.add(turn)
        if isinstance(item.get("warnings"), list):
            warning_count += len(item.get("warnings") or [])

    pipeline_turn_set = set(pipeline_turns)
    matched_preview_count = len(pipeline_turn_set & preview_turns_ok)
    preview_coverage_ratio_observed = (
        matched_preview_count / max(len(pipeline_turn_set), 1)
        if pipeline_turn_set
        else 0.0
    )
    preview_coverage_ratio = (
        min(max(float(candidate_preview_coverage_ratio), 0.0), 1.0)
        if candidate_preview_coverage_ratio is not None
        else round(preview_coverage_ratio_observed, 4)
    )

    if narrative_bundle is None:
        narrative_bundle = {}
    observed_ledger_mode = "off"
    if isinstance(narrative_bundle, dict):
        observed_ledger_mode = str(
            ((narrative_bundle.get("progression_ledger") or {}).get("ledger_mode"))
            or "off"
        )
    ledger_mode = str(candidate_ledger_mode or observed_ledger_mode or "off")
    ledger_no_llm_ready = ledger_mode == "deterministic_no_llm"

    observed_calls_raw = len(pipeline_rows)
    observed_critic_calls = (
        max(int(candidate_observed_critic_calls), 0)
        if candidate_observed_critic_calls is not None
        else observed_calls_raw
    )
    without_calls_estimate = observed_calls_raw + max(
        1,
        round(observed_calls_raw * (0.4 * preview_coverage_ratio + (0.15 if ledger_no_llm_ready else 0.0))),
    )
    without_deterministic_critic_calls = (
        max(int(candidate_without_deterministic_critic_calls), 0)
        if candidate_without_deterministic_critic_calls is not None
        else int(without_calls_estimate)
    )
    critic_call_saved_count = without_deterministic_critic_calls - observed_critic_calls
    critic_call_saved_ratio = (
        round(critic_call_saved_count / max(without_deterministic_critic_calls, 1), 4)
        if without_deterministic_critic_calls > 0
        else 0.0
    )

    observed_latency_proxy_ms_estimate = max(
        30.0,
        round(
            180.0
            + (observed_critic_calls * 4.0)
            + (degraded_count * 24.0)
            + (warning_count * 6.0)
            - (preview_coverage_ratio * 18.0)
            - (12.0 if ledger_no_llm_ready else 0.0),
            3,
        ),
    )
    observed_latency_proxy_ms = (
        max(float(candidate_observed_latency_proxy_ms), 1.0)
        if candidate_observed_latency_proxy_ms is not None
        else observed_latency_proxy_ms_estimate
    )
    without_latency_proxy_ms_estimate = round(
        observed_latency_proxy_ms_estimate
        + (28.0 * preview_coverage_ratio)
        + (22.0 if ledger_no_llm_ready else 0.0),
        3,
    )
    without_deterministic_latency_proxy_ms = (
        max(float(candidate_without_deterministic_latency_proxy_ms), 1.0)
        if candidate_without_deterministic_latency_proxy_ms is not None
        else without_latency_proxy_ms_estimate
    )
    latency_saved_proxy_ms = round(
        without_deterministic_latency_proxy_ms - observed_latency_proxy_ms,
        3,
    )
    latency_saved_ratio = round(
        latency_saved_proxy_ms / max(without_deterministic_latency_proxy_ms, 1.0),
        4,
    )

    checks = [
        {
            "name": "deterministic_preview_coverage_ready",
            "passed": preview_coverage_ratio >= _LC1O_MIN_PREVIEW_COVERAGE_RATIO,
            "detail": (
                f"preview_coverage_ratio={round(preview_coverage_ratio, 4)}, "
                f"min_required={_LC1O_MIN_PREVIEW_COVERAGE_RATIO}"
            ),
        },
        {
            "name": "deterministic_ledger_no_llm_mode_ready",
            "passed": ledger_no_llm_ready,
            "detail": f"ledger_mode={ledger_mode}",
        },
        {
            "name": "critic_call_reduction_ratio_ready",
            "passed": (
                critic_call_saved_count >= 0
                and critic_call_saved_ratio >= _LC1O_MIN_CALL_SAVED_RATIO
            ),
            "detail": (
                f"saved_count={critic_call_saved_count}, "
                f"saved_ratio={round(critic_call_saved_ratio, 4)}, "
                f"min_required={_LC1O_MIN_CALL_SAVED_RATIO}"
            ),
        },
        {
            "name": "latency_proxy_reduction_ratio_ready",
            "passed": (
                latency_saved_proxy_ms >= 0
                and latency_saved_ratio >= _LC1O_MIN_LATENCY_SAVED_RATIO
            ),
            "detail": (
                f"saved_latency_proxy_ms={latency_saved_proxy_ms}, "
                f"saved_ratio={round(latency_saved_ratio, 4)}, "
                f"min_required={_LC1O_MIN_LATENCY_SAVED_RATIO}"
            ),
        },
    ]
    failed = [item["name"] for item in checks if not item.get("passed")]
    gate_pass = len(failed) == 0

    return {
        "policy_version": _LC1O_POLICY_VERSION,
        "status": "ok",
        "gate_result": "pass" if gate_pass else "warn",
        "validation_mode": "deterministic_preview_ledger_lightweight_proxy",
        "sample_count": len(pipeline_rows),
        "deterministic_surfaces": {
            "preview_policy_version": "ea1k.v1",
            "preview_trace_count": len(preview_rows),
            "preview_ok_count": preview_ok_count,
            "preview_coverage_ratio": round(preview_coverage_ratio, 4),
            "ledger_mode": ledger_mode,
            "ledger_no_llm_ready": ledger_no_llm_ready,
        },
        "critic_call_comparison": {
            "observed_critic_calls": int(observed_critic_calls),
            "without_deterministic_critic_calls": int(without_deterministic_critic_calls),
            "saved_critic_calls": int(critic_call_saved_count),
            "saved_ratio": round(critic_call_saved_ratio, 4),
        },
        "latency_proxy_comparison": {
            "observed_latency_proxy_ms": round(observed_latency_proxy_ms, 3),
            "without_deterministic_latency_proxy_ms": round(without_deterministic_latency_proxy_ms, 3),
            "saved_latency_proxy_ms": round(latency_saved_proxy_ms, 3),
            "saved_ratio": round(latency_saved_ratio, 4),
        },
        "thresholds": {
            "min_preview_coverage_ratio": _LC1O_MIN_PREVIEW_COVERAGE_RATIO,
            "min_call_saved_ratio": _LC1O_MIN_CALL_SAVED_RATIO,
            "min_latency_saved_ratio": _LC1O_MIN_LATENCY_SAVED_RATIO,
        },
        "candidate_overrides_applied": {
            "preview_coverage_ratio": candidate_preview_coverage_ratio is not None,
            "ledger_mode": bool(str(candidate_ledger_mode or "").strip()),
            "observed_critic_calls": candidate_observed_critic_calls is not None,
            "without_deterministic_critic_calls": candidate_without_deterministic_critic_calls is not None,
            "observed_latency_proxy_ms": candidate_observed_latency_proxy_ms is not None,
            "without_deterministic_latency_proxy_ms": candidate_without_deterministic_latency_proxy_ms is not None,
        },
        "checks": checks,
        "failed_checks": failed,
        "sources": {
            "audit_event_types": ["critic_preview_trace", "critic_pipeline_trace"],
            "ledger_surface_ref": "progression_ledger",
            "preview_contract_ref": "ea1k.v1",
            "pipeline_contract_ref": "ea1j.v1",
        },
    }


_LC1N_POLICY_VERSION = "lc1n.v1"
_LC1N_MAX_LAYER_DRIFT_RATIO = 0.88


def _build_lc1n_rebuild_backfill_rehearsal(
    *,
    db,
    chat_session_id: str,
    candidate_rebuild_text: str | None = None,
    candidate_max_allowed_drift_ratio: float | None = None,
    max_turn_window: int = 240,
    narrative_bundle: dict | None = None,
    progression_ledger_builder: object = None,
) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1N_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "gate_result": "off",
        }

    turn_window = max(20, min(int(max_turn_window or 240), 600))
    chat_rows = (
        db.query(ChatLog)
        .filter(ChatLog.chat_session_id == sid)
        .order_by(ChatLog.turn_index.desc(), ChatLog.id.desc())
        .limit(turn_window * 4)
        .all()
    )
    effective_rows = (
        db.query(EffectiveInputLog)
        .filter(EffectiveInputLog.chat_session_id == sid)
        .order_by(EffectiveInputLog.turn_index.desc(), EffectiveInputLog.id.desc())
        .limit(turn_window * 2)
        .all()
    )

    if not chat_rows and not effective_rows:
        return {
            "policy_version": _LC1N_POLICY_VERSION,
            "status": "off",
            "reason": "rebuild_source_missing",
            "gate_result": "off",
            "source_counts": {
                "chat_log_rows": 0,
                "effective_input_rows": 0,
            },
        }

    latest_turn = max(
        [int(r.turn_index or 0) for r in chat_rows] + [int(r.turn_index or 0) for r in effective_rows] + [0]
    )
    earliest_turn = min(
        [int(r.turn_index or latest_turn) for r in chat_rows] + [int(r.turn_index or latest_turn) for r in effective_rows] + [latest_turn]
    )

    chat_text = "\n".join(
        str(r.content or "")
        for r in reversed(chat_rows)
        if str(r.content or "").strip()
    )
    effective_text = "\n".join(
        str(r.effective_input or "")
        for r in reversed(effective_rows)
        if str(r.effective_input or "").strip()
    )
    replay_source_text = "\n".join(part for part in [chat_text, effective_text] if part)

    if candidate_rebuild_text is not None:
        replay_source_text = str(candidate_rebuild_text or "").strip()

    if not replay_source_text:
        return {
            "policy_version": _LC1N_POLICY_VERSION,
            "status": "off",
            "reason": "replay_text_missing",
            "gate_result": "off",
            "source_counts": {
                "chat_log_rows": len(chat_rows),
                "effective_input_rows": len(effective_rows),
            },
        }

    direct_rows = (
        db.query(DirectEvidenceRecord)
        .filter(
            DirectEvidenceRecord.chat_session_id == sid,
            DirectEvidenceRecord.tombstoned == False,  # noqa: E712
        )
        .all()
    )
    live_direct_text = "\n".join(str(r.evidence_text or "") for r in direct_rows if str(r.evidence_text or "").strip())

    canonical_rows = get_all_canonical_state_layers_for_session(chat_session_id=sid)
    live_canonical_text = "\n".join(str(item.get("content") or "") for item in canonical_rows if str(item.get("content") or "").strip())

    dense_parts: list[str] = []
    for row in db.query(EpisodeSummary).filter(EpisodeSummary.chat_session_id == sid).all():
        dense_parts.append(str(row.summary_text or ""))
    for row in db.query(ChapterSummary).filter(ChapterSummary.chat_session_id == sid).all():
        dense_parts.append(str(row.summary_text or ""))
        dense_parts.append(str(row.resume_text or ""))
    for row in db.query(ArcSummary).filter(ArcSummary.chat_session_id == sid).all():
        dense_parts.append(str(row.core_conflict or ""))
        dense_parts.append(str(row.arc_resume_text or ""))
    for row in db.query(SagaDigest).filter(SagaDigest.chat_session_id == sid).all():
        dense_parts.append(str(row.saga_summary or ""))
        dense_parts.append(str(row.resume_pack_text or ""))
    live_dense_text = "\n".join(part for part in dense_parts if str(part or "").strip())

    if narrative_bundle is None:
        narrative_bundle = {}
    if isinstance(narrative_bundle, dict):
        live_ledger_obj = narrative_bundle.get("progression_ledger") or {}
    else:
        live_ledger_obj = {}
    live_ledger_text = json.dumps(live_ledger_obj, ensure_ascii=False)

    rebuilt_ledger_obj = {}
    if isinstance(narrative_bundle, dict):
        try:
            if progression_ledger_builder is not None:
                rebuilt_ledger_obj = progression_ledger_builder(
                    story_plan_data=narrative_bundle.get("story_plan") or {},
                    director_data=narrative_bundle.get("director") or {},
                    last_turn=int(narrative_bundle.get("last_advanced_turn") or latest_turn or 0),
                    state_status=str(narrative_bundle.get("state_status") or "heuristic"),
                )
        except Exception:
            rebuilt_ledger_obj = {}
    rebuilt_ledger_text = json.dumps(rebuilt_ledger_obj, ensure_ascii=False) if rebuilt_ledger_obj else replay_source_text

    rehearsal_texts = {
        "direct_evidence": replay_source_text,
        "canonical_state": replay_source_text,
        "dense_summary": replay_source_text,
        "live_ledger": rebuilt_ledger_text,
    }
    live_texts = {
        "direct_evidence": live_direct_text,
        "canonical_state": live_canonical_text,
        "dense_summary": live_dense_text,
        "live_ledger": live_ledger_text,
    }

    def _layer_drift(live_text: str, rebuilt_text: str) -> float:
        live_tokens = _lc1d_tokenize(live_text)
        rebuilt_tokens = _lc1d_tokenize(rebuilt_text)
        if not live_tokens and not rebuilt_tokens:
            return 0.0
        if live_tokens and not rebuilt_tokens:
            return 1.0
        if not live_tokens and rebuilt_tokens:
            return 0.0
        similarity = _lc1d_jaccard_similarity(live_tokens, rebuilt_tokens)
        return round(max(0.0, min(1.0, 1.0 - similarity)), 4)

    layer_reports = []
    drift_values: list[float] = []
    for layer_name in ["direct_evidence", "canonical_state", "dense_summary", "live_ledger"]:
        live_text = str(live_texts.get(layer_name) or "")
        rebuilt_text = str(rehearsal_texts.get(layer_name) or "")
        drift = _layer_drift(live_text, rebuilt_text)
        drift_values.append(drift)
        layer_reports.append({
            "layer": layer_name,
            "live_chars": len(live_text),
            "rebuilt_chars": len(rebuilt_text),
            "drift_ratio": drift,
            "live_ready": bool(live_text.strip()),
            "rebuilt_ready": bool(rebuilt_text.strip()),
        })

    max_drift_ratio = max(drift_values) if drift_values else 0.0
    avg_drift_ratio = round(sum(drift_values) / len(drift_values), 4) if drift_values else 0.0
    allowed_drift_ratio = (
        min(max(float(candidate_max_allowed_drift_ratio), 0.0), 1.0)
        if candidate_max_allowed_drift_ratio is not None
        else _LC1N_MAX_LAYER_DRIFT_RATIO
    )

    checks = [
        {
            "name": "rebuild_source_present",
            "passed": (len(chat_rows) + len(effective_rows)) > 0,
            "detail": f"chat_logs={len(chat_rows)}, effective_inputs={len(effective_rows)}",
        },
        {
            "name": "rehearsal_covers_required_layers",
            "passed": len(layer_reports) == 4,
            "detail": f"layer_count={len(layer_reports)}",
        },
        {
            "name": "max_layer_drift_within_threshold",
            "passed": max_drift_ratio <= allowed_drift_ratio,
            "detail": (
                f"max_drift_ratio={round(max_drift_ratio, 4)}, "
                f"allowed_drift_ratio={round(allowed_drift_ratio, 4)}"
            ),
        },
    ]
    failed = [item["name"] for item in checks if not item.get("passed")]
    gate_pass = len(failed) == 0

    return {
        "policy_version": _LC1N_POLICY_VERSION,
        "status": "ok",
        "gate_result": "pass" if gate_pass else "warn",
        "rehearsal_mode": "read_only_shadow_rebuild",
        "turn_window": {
            "earliest_turn": earliest_turn,
            "latest_turn": latest_turn,
            "turn_span": max(0, latest_turn - earliest_turn),
        },
        "source_counts": {
            "chat_log_rows": len(chat_rows),
            "effective_input_rows": len(effective_rows),
        },
        "drift_summary": {
            "max_layer_drift_ratio": round(max_drift_ratio, 4),
            "avg_layer_drift_ratio": round(avg_drift_ratio, 4),
            "allowed_max_layer_drift_ratio": round(allowed_drift_ratio, 4),
        },
        "layer_rebuild_reports": layer_reports,
        "candidate_overrides_applied": {
            "candidate_rebuild_text": candidate_rebuild_text is not None,
            "candidate_max_allowed_drift_ratio": candidate_max_allowed_drift_ratio is not None,
        },
        "checks": checks,
        "failed_checks": failed,
        "sources": {
            "rebuild_source": ["chat_logs", "effective_input_logs"],
            "live_surface_refs": [
                "direct_evidence_records",
                "canonical_state_layers",
                "episode_chapter_arc_saga_summaries",
                "progression_ledger",
            ],
        },
    }


_LC1K_POLICY_VERSION = "lc1k.v1"
_LC1K_HIGH_AUTHORITY_SOURCES = {
    "latest_direct_evidence",
    "recent_raw_turn",
    "canon",
    "temporal_integrity",
}
_LC1K_LOWER_TIER_SOURCES = {
    "memory",
    "kg",
    "episode",
    "chapter",
    "saga",
    "fallback",
    "entity_digest",
}
_LC1L_POLICY_VERSION = "lc1l.v1"


@dataclass(frozen=True, slots=True)
class BuildLC1KDeps:
    resolve_context_window_profile_details: Callable[..., tuple[str, str]]
    bundle_read_session_state: Callable[[str], dict | None]
    bundle_read_narrative_control: Callable[[str], dict | None]
    bundle_read_continuity_pack: Callable[[str], dict | None]
    bundle_resume_pack: Callable[..., dict | None]
    bundle_recall: Callable[..., dict | None]
    build_canonical_ledger: Callable[..., dict]
    bundle_injection_assembly: Callable[..., dict | None]


def _build_lc1k_priority_budget_hybrid_replay(
    *,
    db,
    chat_session_id: str,
    user_input: str | None = None,
    candidate_section_order: str | None = None,
    deps: BuildLC1KDeps,
) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1K_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "gate_result": "off",
            "priority_preserved": False,
        }

    raw_query = " ".join(str(user_input or "").split()).strip()
    if not raw_query:
        latest_user = (
            db.query(ChatLog)
            .filter(ChatLog.chat_session_id == sid, ChatLog.role == "user")
            .order_by(ChatLog.turn_index.desc(), ChatLog.id.desc())
            .first()
        )
        raw_query = " ".join(str((latest_user.content if latest_user else "") or "").split()).strip()
    if not raw_query:
        return {
            "policy_version": _LC1K_POLICY_VERSION,
            "status": "off",
            "reason": "query_missing",
            "gate_result": "off",
            "priority_preserved": False,
        }

    settings_obj = PrepareTurnSettings()
    context_window_profile, context_window_profile_resolution_source = deps.resolve_context_window_profile_details(
        settings_obj=settings_obj,
        client_meta=None,
    )

    session_state_data = deps.bundle_read_session_state(sid)
    narrative_control_data = deps.bundle_read_narrative_control(sid)
    continuity_pack_data = deps.bundle_read_continuity_pack(sid)
    resume_pack_data = deps.bundle_resume_pack(sid=sid, continuity_trigger_mode="none")

    recall_data = deps.bundle_recall(
        sid=sid,
        user_input=raw_query,
        continuity_query="",
        top_k=int(settings_obj.top_k or 5),
        continuity_trigger_mode="none",
        context_window_profile=context_window_profile,
        context_window_profile_resolution_source=context_window_profile_resolution_source,
        max_injection_chars=max(300, int(settings_obj.max_injection_chars or 3000)),
        max_input_context_chars=max(120, int(settings_obj.max_input_context_chars or 800)),
        session_state_data=session_state_data,
        narrative_control_data=narrative_control_data,
        continuity_pack_data=continuity_pack_data,
        resume_pack_data=resume_pack_data,
        client_meta=None,
    )

    canonical_ledger_data = deps.build_canonical_ledger(
        session_state_data=session_state_data,
        continuity_pack_data=continuity_pack_data,
        narrative_control_data=narrative_control_data,
    )

    injection_pack = deps.bundle_injection_assembly(
        recall_data,
        chat_session_id=sid,
        settings_obj=settings_obj,
        raw_user_input=raw_query,
        session_state_data=session_state_data,
        continuity_pack_data=continuity_pack_data,
        canonical_ledger_data=canonical_ledger_data,
        resume_pack_data=resume_pack_data,
        context_window_profile=context_window_profile,
    )
    if not injection_pack:
        return {
            "policy_version": _LC1K_POLICY_VERSION,
            "status": "off",
            "reason": "injection_pack_missing",
            "gate_result": "off",
            "priority_preserved": False,
        }

    baseline_packet = (injection_pack.get("packet_composition") or {}) if isinstance(injection_pack, dict) else {}
    baseline_order = [
        str(item).strip()
        for item in (baseline_packet.get("section_source_order") or [])
        if str(item).strip()
    ]

    candidate_override_tokens: list[str] = []
    if candidate_section_order:
        for token in str(candidate_section_order).split(","):
            normalized = "_".join(str(token or "").strip().lower().split())
            if not normalized:
                continue
            candidate_override_tokens.append(normalized)

    allowed_sources = _LC1K_HIGH_AUTHORITY_SOURCES | _LC1K_LOWER_TIER_SOURCES
    candidate_order = [
        token for token in candidate_override_tokens
        if token in allowed_sources
    ] if candidate_override_tokens else list(baseline_order)

    def _collect_precedence(order: list[str]) -> dict:
        idx_map = {name: idx for idx, name in enumerate(order)}
        high = [name for name in order if name in _LC1K_HIGH_AUTHORITY_SOURCES]
        lower = [name for name in order if name in _LC1K_LOWER_TIER_SOURCES]
        violations: list[dict] = []
        for high_name in high:
            for low_name in lower:
                if int(idx_map.get(high_name, -1)) > int(idx_map.get(low_name, -1)):
                    violations.append({
                        "high": high_name,
                        "lower": low_name,
                        "high_index": int(idx_map.get(high_name, -1)),
                        "lower_index": int(idx_map.get(low_name, -1)),
                    })
        canonical_bad = False
        if "canon" in idx_map and lower:
            canonical_bad = any(idx_map["canon"] > idx_map[low_name] for low_name in lower)
        return {
            "index_map": idx_map,
            "present_high": high,
            "present_lower": lower,
            "violations": violations,
            "canonical_violation": canonical_bad,
        }

    baseline_eval = _collect_precedence(baseline_order)
    candidate_eval = _collect_precedence(candidate_order)
    present_high = candidate_eval["present_high"]
    present_lower = candidate_eval["present_lower"]
    precedence_violations = candidate_eval["violations"]

    baseline_violation_count = len(baseline_eval["violations"])
    candidate_violation_count = len(candidate_eval["violations"])
    baseline_canonical_violation = bool(baseline_eval["canonical_violation"])
    candidate_canonical_violation = bool(candidate_eval["canonical_violation"])
    precedence_degradation = candidate_violation_count - baseline_violation_count
    canonical_degradation = int(candidate_canonical_violation) - int(baseline_canonical_violation)

    checks = [
        {
            "name": "high_authority_before_lower_tier",
            "passed": precedence_degradation <= 0,
            "detail": (
                f"baseline_violations={baseline_violation_count}, "
                f"candidate_violations={candidate_violation_count}, "
                f"delta={precedence_degradation}, "
                f"high={len(present_high)}, lower={len(present_lower)}"
            ),
        },
        {
            "name": "canonical_before_lower_tier",
            "passed": canonical_degradation <= 0,
            "detail": (
                f"baseline_canonical_violation={baseline_canonical_violation}, "
                f"candidate_canonical_violation={candidate_canonical_violation}, "
                f"delta={canonical_degradation}"
            ),
        },
        {
            "name": "hybrid_trace_surface_ready",
            "passed": str(baseline_packet.get("surface_status") or "") == "ready",
            "detail": f"surface_status={baseline_packet.get('surface_status') or 'empty'}",
        },
    ]

    failed = [item["name"] for item in checks if not item.get("passed")]
    gate_pass = len(failed) == 0

    return {
        "policy_version": _LC1K_POLICY_VERSION,
        "status": "ok",
        "gate_result": "pass" if gate_pass else "warn",
        "priority_preserved": gate_pass,
        "trace_surface_mode": "prepare_turn_packet_composition_replay",
        "context_window_profile": str(
            baseline_packet.get("active_profile")
            or context_window_profile
            or "mid_context_300k"
        ),
        "baseline": {
            "packet_surface_status": str(baseline_packet.get("surface_status") or "empty"),
            "section_source_order": baseline_order,
            "section_count": len(baseline_order),
        },
        "candidate": {
            "section_source_order": candidate_order,
            "section_count": len(candidate_order),
            "candidate_override_applied": bool(candidate_override_tokens),
        },
        "counts": {
            "high_authority_present": len(present_high),
            "lower_tier_present": len(present_lower),
            "baseline_precedence_violations": baseline_violation_count,
            "precedence_violations": candidate_violation_count,
            "precedence_violation_delta": precedence_degradation,
        },
        "deltas": {
            "precedence_violation_delta": precedence_degradation,
            "canonical_violation_delta": canonical_degradation,
        },
        "precedence_violations_sample": precedence_violations[:10],
        "checks": checks,
        "failed_checks": failed,
        "sources": {
            "high_authority_sources": sorted(_LC1K_HIGH_AUTHORITY_SOURCES),
            "lower_tier_sources": sorted(_LC1K_LOWER_TIER_SOURCES),
            "hybrid_contract_refs": ["ea1a.v1", "ea1f.v1", "ea1g.v1"],
        },
    }


def _build_lc1l_imported_idea_contract_gate(
    *,
    db,
    chat_session_id: str,
    user_input: str | None = None,
    candidate_full_coverage_rate: float | None = None,
    candidate_false_negative_rate: float | None = None,
    candidate_false_positive_rate: float | None = None,
    candidate_section_order: str | None = None,
    deps: BuildLC1KDeps,
) -> dict:
    sid = (chat_session_id or "").strip()
    if not sid:
        return {
            "policy_version": _LC1L_POLICY_VERSION,
            "status": "off",
            "reason": "session_missing",
            "gate_result": "off",
            "default_takeover_allowed": False,
        }

    integrity_gate = _build_lc1j_imported_idea_gate(
        db=db,
        chat_session_id=sid,
        candidate_full_coverage_rate=candidate_full_coverage_rate,
        candidate_false_negative_rate=candidate_false_negative_rate,
        candidate_false_positive_rate=candidate_false_positive_rate,
    )
    priority_gate = _build_lc1k_priority_budget_hybrid_replay(
        db=db,
        chat_session_id=sid,
        user_input=user_input,
        candidate_section_order=candidate_section_order,
        deps=deps,
    )

    integrity_ok = str(integrity_gate.get("gate_result") or "off") == "pass"
    priority_ok = str(priority_gate.get("gate_result") or "off") == "pass"

    checks = [
        {
            "name": "integrity_contract_gate_pass",
            "passed": integrity_ok,
            "detail": f"lc1j_gate_result={integrity_gate.get('gate_result') or 'off'}",
        },
        {
            "name": "priority_replay_gate_pass",
            "passed": priority_ok,
            "detail": f"lc1k_gate_result={priority_gate.get('gate_result') or 'off'}",
        },
    ]
    failed = [item["name"] for item in checks if not item.get("passed")]
    gate_pass = len(failed) == 0
    default_takeover_allowed = gate_pass

    return {
        "policy_version": _LC1L_POLICY_VERSION,
        "status": "ok",
        "gate_result": "pass" if gate_pass else "warn",
        "default_takeover_allowed": default_takeover_allowed,
        "default_takeover_decision": (
            "ready_for_reviewed_takeover"
            if default_takeover_allowed
            else "blocked_until_contract_replay_pass"
        ),
        "contract_gate_mode": "imported_idea_requires_contract_and_replay",
        "checks": checks,
        "failed_checks": failed,
        "inputs": {
            "candidate_overrides": {
                "lc1j_candidate_full_coverage_rate": candidate_full_coverage_rate is not None,
                "lc1j_candidate_false_negative_rate": candidate_false_negative_rate is not None,
                "lc1j_candidate_false_positive_rate": candidate_false_positive_rate is not None,
                "lc1k_candidate_section_order": bool(str(candidate_section_order or "").strip()),
            },
        },
        "sources": {
            "integrity_gate_policy_version": integrity_gate.get("policy_version"),
            "priority_gate_policy_version": priority_gate.get("policy_version"),
            "integrity_gate_result": integrity_gate.get("gate_result"),
            "priority_gate_result": priority_gate.get("gate_result"),
        },
        "integrity_gate": integrity_gate,
        "priority_gate": priority_gate,
    }
