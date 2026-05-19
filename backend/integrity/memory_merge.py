from __future__ import annotations

import json
from collections.abc import Callable

from sqlalchemy.orm import Session

from backend.models import AuditLog, Memory


SEMANTIC_MERGE_POLICY_VERSION = "gap2a.v1"
SEMANTIC_MERGE_THRESHOLD = 0.92


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _load_summary(summary_json: str | None) -> dict:
    if not summary_json:
        return {}
    try:
        payload = json.loads(summary_json)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_embedding(embedding_json: str | None) -> list[float] | None:
    if not embedding_json:
        return None
    try:
        payload = json.loads(embedding_json)
    except Exception:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    return payload


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _merge_unique_strings(old_items: object, new_items: object) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for raw_items in (old_items, new_items):
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            text = str(item or "").strip()
            if not text:
                continue
            dedupe_key = text.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            merged.append(text)
    return merged


class MemoryMerger:
    def __init__(self, semantic_merge_threshold: float = SEMANTIC_MERGE_THRESHOLD):
        self.semantic_merge_threshold = float(semantic_merge_threshold)

    def should_merge(
        self,
        db: Session,
        *,
        chat_session_id: str,
        turn_summary: str,
        embedding: list[float] | None,
        archive_wing: str | None,
        archive_room: str | None,
        cosine_fn: Callable[[object, object], float] | None,
    ) -> dict:
        normalized_summary = _normalize_text(turn_summary)
        best_row: Memory | None = None
        best_similarity = 0.0
        best_reason = ""

        rows = (
            db.query(Memory)
            .filter(Memory.chat_session_id == chat_session_id)
            .order_by(Memory.turn_index.desc().nullslast(), Memory.id.desc())
            .limit(150)
            .all()
        )
        for row in rows:
            if archive_wing and getattr(row, "archive_wing", None) not in (None, archive_wing):
                continue
            if archive_room and getattr(row, "archive_room", None) not in (None, archive_room):
                continue

            summary_payload = _load_summary(row.summary_json)
            existing_summary = _normalize_text(summary_payload.get("turn_summary"))
            if normalized_summary and existing_summary and normalized_summary == existing_summary:
                return {
                    "decision": "merge",
                    "match": row,
                    "similarity": 1.0,
                    "reason": "exact_turn_summary",
                }

            if embedding is None or cosine_fn is None:
                continue

            existing_embedding = _load_embedding(row.embedding)
            if existing_embedding is None:
                continue

            similarity = float(cosine_fn(embedding, existing_embedding) or 0.0)
            if similarity >= self.semantic_merge_threshold and similarity > best_similarity:
                best_row = row
                best_similarity = similarity
                best_reason = "semantic_similarity"

        if best_row is None:
            return {"decision": "insert_new", "match": None, "similarity": 0.0, "reason": ""}
        return {
            "decision": "merge",
            "match": best_row,
            "similarity": best_similarity,
            "reason": best_reason,
        }

    def merge_memories(
        self,
        db: Session,
        *,
        existing_row: Memory,
        turn_index: int,
        summary_json: dict,
        importance: float,
        embedding_json: str | None,
        embedding_model_name: str | None,
        archive_wing: str | None,
        archive_room: str | None,
        evidence_json: str | None,
        emotional_intensity: float | None,
        narrative_significance: float | None,
        emotional_boost: float | None,
        similarity: float,
        merge_reason: str,
        source: str,
    ) -> dict:
        existing_summary = _load_summary(existing_row.summary_json)
        merged_summary = dict(existing_summary)

        old_importance_score = _coerce_float(existing_summary.get("importance_score"), 0.0)
        new_importance_score = _coerce_float(summary_json.get("importance_score"), 0.0)
        merged_summary.update({k: v for k, v in summary_json.items() if v is not None})
        merged_summary["importance_score"] = max(old_importance_score, new_importance_score)

        old_turn_summary = str(existing_summary.get("turn_summary") or "").strip()
        new_turn_summary = str(summary_json.get("turn_summary") or "").strip()
        if len(new_turn_summary) > len(old_turn_summary):
            merged_summary["turn_summary"] = new_turn_summary
        elif old_turn_summary:
            merged_summary["turn_summary"] = old_turn_summary

        old_evidence = existing_summary.get("evidence_excerpts")
        new_evidence = summary_json.get("evidence_excerpts")
        merged_evidence = _merge_unique_strings(old_evidence, new_evidence)
        if merged_evidence:
            merged_summary["evidence_excerpts"] = merged_evidence

        existing_row.summary_json = json.dumps(merged_summary, ensure_ascii=False)
        existing_row.turn_index = max(int(existing_row.turn_index or 0), int(turn_index or 0))
        existing_row.importance = max(_coerce_float(existing_row.importance, 0.0), float(importance or 0.0))

        if embedding_json:
            existing_row.embedding = embedding_json
        if embedding_model_name:
            existing_row.embedding_model = embedding_model_name
        if archive_wing:
            existing_row.archive_wing = archive_wing
        if archive_room:
            existing_row.archive_room = archive_room

        if emotional_boost is not None:
            existing_row.emotional_boost = max(_coerce_float(existing_row.emotional_boost, 0.0), float(emotional_boost or 0.0)) or None
        if emotional_intensity is not None:
            existing_row.emotional_intensity = max(_coerce_float(existing_row.emotional_intensity, 0.0), float(emotional_intensity))
        if narrative_significance is not None:
            existing_row.narrative_significance = max(_coerce_float(existing_row.narrative_significance, 0.0), float(narrative_significance))

        if evidence_json:
            existing_row.evidence = json.dumps(merged_evidence, ensure_ascii=False) if merged_evidence else evidence_json
        elif merged_evidence:
            existing_row.evidence = json.dumps(merged_evidence, ensure_ascii=False)

        db.add(
            AuditLog(
                event_type="memory_semantic_merge",
                chat_session_id=existing_row.chat_session_id,
                target_type="memory",
                target_id=existing_row.id,
                summary=f"Merged duplicate memory for turn {turn_index}",
                details_json=json.dumps(
                    {
                        "policy_version": SEMANTIC_MERGE_POLICY_VERSION,
                        "merge_reason": merge_reason,
                        "similarity": similarity,
                        "existing_turn_index": existing_row.turn_index,
                        "merged_turn_index": turn_index,
                    },
                    ensure_ascii=False,
                ),
                source=source,
            )
        )

        return {
            "policy_version": SEMANTIC_MERGE_POLICY_VERSION,
            "merge_reason": merge_reason,
            "similarity": similarity,
            "memory_id": existing_row.id,
        }