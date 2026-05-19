"""Palace Bridge — DB 탐색 + 제한적 편집 로직 (Sprint 2-E / 2-F).

DB Explorer UI를 위한 데이터 접근 계층.
chat_logs, memories, kg_triples를 session 범위로 조회한다 (Sprint 2-E).
memories, kg_triples의 허용된 필드만 제한적으로 수정한다 (Sprint 2-F).
Sprint 3-D: reindex_session — embedding 재생성 배치 처리.
Phase 3-2: generate_episode_summary — 구간 요약 자동 생성.
"""

import json
import logging
import re
from datetime import datetime, timezone
from time import perf_counter

import httpx
from sqlalchemy import func, desc, UniqueConstraint
from sqlalchemy.orm import Session

from backend import to_kst

from backend.archive.bridge import ArchiveBridge, _normalize_kg_entity_label
from backend.database import SessionLocal, Base
from backend.direct_evidence import (
    DIRECT_EVIDENCE_ARCHIVE_STATES,
    DIRECT_EVIDENCE_CAPTURE_VERIFICATIONS,
    DIRECT_EVIDENCE_COMMITTED_GATES,
    DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS,
    DIRECT_EVIDENCE_RETENTION_IMPORTANCE_TIERS,
    DIRECT_EVIDENCE_RETENTION_POLICY_VERSION,
    DIRECT_EVIDENCE_RETENTION_WINDOWS_TURNS,
    build_state_counts,
    is_current_truth_excluded,
    is_deleted_turn_tombstone_retained,
    is_gate_approved_current_truth,
    is_gate_blocked_signal,
    is_retention_expired,
    is_within_retention_window,
    normalize_archive_state,
    normalize_capture_verification,
    normalize_committed_gate,
    resolve_archive_bucket,
    resolve_committed_gate,
    resolve_retention_importance_tier,
    resolve_retention_ttl_turns,
)
from backend.embedding_provenance import (
    STEP13_EMBEDDING_PROVENANCE_POLICY_VERSION as EMBEDDING_PROVENANCE_POLICY_VERSION,
    STEP13_NEEDS_REEMBED_POLICY_VERSION as NEEDS_REEMBED_POLICY_VERSION,
    assess_memory_embedding_state,
)
from backend.models import ChatLog, EffectiveInputLog, Memory, DirectEvidenceRecord, KGTriple, AuditLog, CriticFeedback, ActiveState, CanonicalStateLayer, EpisodeSummary, ChapterSummary, ArcSummary, SagaDigest, DEFAULT_SESSION_ID, GuidancePlanState, GuidanceCompactRecord, MaintenancePassState, WorldRule, PendingThread
from backend.settings import settings

logger = logging.getLogger(__name__)

_STEP13_PORTABILITY_PACKAGE_POLICY_VERSION = "sp1a.v1"
_STEP13_PORTABILITY_INGEST_GATE_POLICY_VERSION = "sp1b.v1"
_STEP13_PORTABILITY_REBUILD_HANDOFF_POLICY_VERSION = "sp1d.v1"
_STEP13_PORTABILITY_OPERATION_POLICY_VERSION = "sp1e.v1"
_STEP13_EMBEDDING_PROVENANCE_POLICY_VERSION = EMBEDDING_PROVENANCE_POLICY_VERSION
_STEP13_NEEDS_REEMBED_POLICY_VERSION = NEEDS_REEMBED_POLICY_VERSION
_STEP13_REEMBED_SCHEDULING_POLICY_VERSION = "em1d.v1"

# 최대 한 번에 반환할 행 수 (안전 제한)
MAX_LIMIT = 200
DEFAULT_LIMIT = 30


def _clamp_limit(limit: int | None) -> int:
    if limit is None or limit < 1:
        return DEFAULT_LIMIT
    return min(limit, MAX_LIMIT)


def _clamp_offset(offset: int | None) -> int:
    if offset is None or offset < 0:
        return 0
    return offset


def _safe_session_id(chat_session_id: str | None) -> str | None:
    """None이면 None(전역 조회), 빈 문자열이면 None."""
    if chat_session_id and isinstance(chat_session_id, str) and chat_session_id.strip():
        return chat_session_id.strip()
    return None


def _to_epoch_seconds(value) -> float | None:
    """datetime/iso 문자열을 epoch seconds로 변환한다. 실패 시 None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        try:
            return float(value.timestamp())
        except Exception:
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            normalized = raw.replace("Z", "+00:00")
            return float(datetime.fromisoformat(normalized).timestamp())
        except Exception:
            return None
    return None


def _safe_round(value: float, ndigits: int = 3) -> float:
    try:
        return round(float(value), ndigits)
    except Exception:
        return 0.0


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return _safe_round(sum(values) / len(values), 3)


def _safe_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * 0.95)
    return _safe_round(sorted_values[index], 3)


def _safe_json_dict(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _assess_memory_reembed_state(*, embedding_value, embedding_model, current_model) -> dict:
    """Step 13 EM-1a/EM-1b: 현재 프로젝트 모델과 row provenance를 비교해 재색인 필요 여부를 고정한다."""
    return assess_memory_embedding_state(
        embedding_value=embedding_value,
        embedding_model=embedding_model,
        current_model=current_model,
    )


def _build_session_ingest_gate_report(db: Session, *, source_session_id: str, target_session_id: str) -> dict:
    """Step 13 SP-1b: session migrate/import apply 전 dedupe와 lineage 위험을 요약한다."""
    source = str(source_session_id or "").strip()
    target = str(target_session_id or "").strip()

    source_direct_rows = (
        db.query(DirectEvidenceRecord)
        .filter(DirectEvidenceRecord.chat_session_id == source)
        .order_by(DirectEvidenceRecord.id.asc())
        .all()
    )
    target_direct_hashes = {
        str(row[0]).strip()
        for row in (
            db.query(DirectEvidenceRecord.source_hash)
            .filter(DirectEvidenceRecord.chat_session_id == target)
            .filter(DirectEvidenceRecord.source_hash.is_not(None))
            .all()
        )
        if str(row[0] or "").strip()
    }

    duplicate_source_hashes: list[str] = []
    missing_source_hash_count = 0
    lineage_missing_session_origin_count = 0
    lineage_foreign_session_origin_count = 0
    lineage_matching_session_origin_count = 0
    tombstoned_source_rows = 0
    superseded_source_rows = 0
    stale_duplicate_source_hashes: list[str] = []
    tombstone_merge_candidate_hashes: list[str] = []
    superseded_merge_candidate_hashes: list[str] = []

    for row in source_direct_rows:
        source_hash = str(getattr(row, "source_hash", "") or "").strip()
        if not source_hash:
            missing_source_hash_count += 1
        elif source_hash in target_direct_hashes:
            duplicate_source_hashes.append(source_hash)
            if is_current_truth_excluded(
                tombstoned=getattr(row, "tombstoned", False),
                superseded_by_id=getattr(row, "superseded_by_id", None),
            ):
                stale_duplicate_source_hashes.append(source_hash)
                if bool(getattr(row, "tombstoned", False)):
                    tombstone_merge_candidate_hashes.append(source_hash)
                if getattr(row, "superseded_by_id", None) is not None:
                    superseded_merge_candidate_hashes.append(source_hash)

        parsed_lineage = _safe_json_dict(getattr(row, "lineage_json", None))
        session_origin = str(parsed_lineage.get("session_origin") or "").strip()
        if not session_origin:
            lineage_missing_session_origin_count += 1
        elif session_origin == source:
            lineage_matching_session_origin_count += 1
        else:
            lineage_foreign_session_origin_count += 1

        if bool(getattr(row, "tombstoned", False)):
            tombstoned_source_rows += 1
        if getattr(row, "superseded_by_id", None) is not None:
            superseded_source_rows += 1

    source_memory_turns = {
        int(turn)
        for (turn,) in (
            db.query(Memory.turn_index)
            .filter(Memory.chat_session_id == source)
            .filter(Memory.turn_index.is_not(None))
            .all()
        )
    }
    target_memory_turns = {
        int(turn)
        for (turn,) in (
            db.query(Memory.turn_index)
            .filter(Memory.chat_session_id == target)
            .filter(Memory.turn_index.is_not(None))
            .all()
        )
    }

    source_effective_turns = {
        int(turn)
        for (turn,) in (
            db.query(EffectiveInputLog.turn_index)
            .filter(EffectiveInputLog.chat_session_id == source)
            .filter(EffectiveInputLog.turn_index.is_not(None))
            .all()
        )
    }
    target_effective_turns = {
        int(turn)
        for (turn,) in (
            db.query(EffectiveInputLog.turn_index)
            .filter(EffectiveInputLog.chat_session_id == target)
            .filter(EffectiveInputLog.turn_index.is_not(None))
            .all()
        )
    }

    source_canonical_pairs = {
        (str(layer_type or "").strip(), int(source_turn))
        for layer_type, source_turn in (
            db.query(CanonicalStateLayer.layer_type, CanonicalStateLayer.source_turn)
            .filter(CanonicalStateLayer.chat_session_id == source)
            .filter(CanonicalStateLayer.source_turn.is_not(None))
            .all()
        )
    }
    target_canonical_pairs = {
        (str(layer_type or "").strip(), int(source_turn))
        for layer_type, source_turn in (
            db.query(CanonicalStateLayer.layer_type, CanonicalStateLayer.source_turn)
            .filter(CanonicalStateLayer.chat_session_id == target)
            .filter(CanonicalStateLayer.source_turn.is_not(None))
            .all()
        )
    }

    duplicate_source_hashes = sorted(set(duplicate_source_hashes))
    stale_duplicate_source_hashes = sorted(set(stale_duplicate_source_hashes))
    tombstone_merge_candidate_hashes = sorted(set(tombstone_merge_candidate_hashes))
    superseded_merge_candidate_hashes = sorted(set(superseded_merge_candidate_hashes))
    overlapping_memory_turns = sorted(source_memory_turns.intersection(target_memory_turns))
    overlapping_effective_input_turns = sorted(source_effective_turns.intersection(target_effective_turns))
    overlapping_canonical_source_pairs = sorted(source_canonical_pairs.intersection(target_canonical_pairs))

    review_required = bool(
        duplicate_source_hashes
        or lineage_foreign_session_origin_count > 0
        or missing_source_hash_count > 0
    )

    return {
        "policy_version": _STEP13_PORTABILITY_INGEST_GATE_POLICY_VERSION,
        "mode": "manual_session_import_gate",
        "session_origin": source,
        "target_session": target,
        "apply_requires_gate_status": "ready",
        "dedupe_keys_by_surface": {
            "direct_evidence_records": ["source_hash"],
            "memories": ["source_turn"],
            "effective_input_logs": ["source_turn"],
            "canonical_state_layers": ["layer_type", "source_turn"],
        },
        "review_required": review_required,
        "recommended_gate_status": "review_required" if review_required else "ready",
        "direct_evidence": {
            "source_rows": len(source_direct_rows),
            "target_hash_count": len(target_direct_hashes),
            "duplicate_source_hash_count": len(duplicate_source_hashes),
            "duplicate_source_hash_preview": duplicate_source_hashes[:10],
            "stale_duplicate_source_hash_count": len(stale_duplicate_source_hashes),
            "stale_duplicate_source_hash_preview": stale_duplicate_source_hashes[:10],
            "missing_source_hash_count": missing_source_hash_count,
            "lineage_matching_session_origin_count": lineage_matching_session_origin_count,
            "lineage_missing_session_origin_count": lineage_missing_session_origin_count,
            "lineage_foreign_session_origin_count": lineage_foreign_session_origin_count,
            "tombstoned_source_rows": tombstoned_source_rows,
            "superseded_source_rows": superseded_source_rows,
            "tombstone_merge_candidate_count": len(tombstone_merge_candidate_hashes),
            "tombstone_merge_candidate_preview": tombstone_merge_candidate_hashes[:10],
            "superseded_merge_candidate_count": len(superseded_merge_candidate_hashes),
            "superseded_merge_candidate_preview": superseded_merge_candidate_hashes[:10],
        },
        "turn_collision_review": {
            "memory_source_turn_collision_count": len(overlapping_memory_turns),
            "memory_source_turn_collision_preview": overlapping_memory_turns[:10],
            "effective_input_turn_collision_count": len(overlapping_effective_input_turns),
            "effective_input_turn_collision_preview": overlapping_effective_input_turns[:10],
            "canonical_source_turn_collision_count": len(overlapping_canonical_source_pairs),
            "canonical_source_turn_collision_preview": [
                {"layer_type": layer_type, "source_turn": source_turn}
                for layer_type, source_turn in overlapping_canonical_source_pairs[:10]
            ],
        },
    }


def _collect_direct_evidence_duplicate_merges(
    db: Session,
    *,
    source_session_id: str,
    target_session_id: str,
) -> dict:
    source = str(source_session_id or "").strip()
    target = str(target_session_id or "").strip()

    source_rows = (
        db.query(DirectEvidenceRecord)
        .filter(DirectEvidenceRecord.chat_session_id == source)
        .filter(DirectEvidenceRecord.source_hash.is_not(None))
        .order_by(DirectEvidenceRecord.id.asc())
        .all()
    )
    target_rows = (
        db.query(DirectEvidenceRecord)
        .filter(DirectEvidenceRecord.chat_session_id == target)
        .filter(DirectEvidenceRecord.source_hash.is_not(None))
        .order_by(DirectEvidenceRecord.id.asc())
        .all()
    )

    source_by_id = {int(row.id): row for row in source_rows}
    source_hash_rows: dict[str, list[DirectEvidenceRecord]] = {}
    target_hash_rows: dict[str, list[DirectEvidenceRecord]] = {}
    for row in source_rows:
        source_hash = str(getattr(row, "source_hash", "") or "").strip()
        if source_hash:
            source_hash_rows.setdefault(source_hash, []).append(row)
    for row in target_rows:
        source_hash = str(getattr(row, "source_hash", "") or "").strip()
        if source_hash:
            target_hash_rows.setdefault(source_hash, []).append(row)

    merges: list[dict] = []
    tombstone_hashes: set[str] = set()
    superseded_hashes: set[str] = set()
    unresolved_superseded_hashes: set[str] = set()

    for source_hash, matching_source_rows in source_hash_rows.items():
        matching_target_rows = list(target_hash_rows.get(source_hash) or [])
        if not matching_target_rows:
            continue

        target_primary = matching_target_rows[0]
        for source_row in matching_source_rows:
            if not is_current_truth_excluded(
                tombstoned=source_row.tombstoned,
                superseded_by_id=source_row.superseded_by_id,
            ):
                continue

            merge_entry = {
                "source_hash": source_hash,
                "source_record_id": int(source_row.id),
                "target_record_id": int(target_primary.id),
                "source_tombstoned": bool(source_row.tombstoned),
                "source_superseded_by_id": source_row.superseded_by_id,
                "resolved_target_superseded_by_id": None,
                "replacement_resolution": None,
            }

            if bool(source_row.tombstoned):
                tombstone_hashes.add(source_hash)

            replacement_id = getattr(source_row, "superseded_by_id", None)
            if replacement_id is not None:
                superseded_hashes.add(source_hash)
                replacement_resolution = None
                resolved_target_superseded_by_id = None

                replacement_source_row = source_by_id.get(int(replacement_id))
                if replacement_source_row is not None:
                    replacement_hash = str(getattr(replacement_source_row, "source_hash", "") or "").strip()
                    target_replacement_row = None
                    if replacement_hash:
                        target_matches = list(target_hash_rows.get(replacement_hash) or [])
                        if target_matches:
                            target_replacement_row = target_matches[0]
                            replacement_resolution = "target_hash_match"

                    if target_replacement_row is None:
                        target_replacement_row = replacement_source_row
                        replacement_resolution = "source_row_move"

                    resolved_target_superseded_by_id = int(target_replacement_row.id)
                else:
                    existing_target_replacement = (
                        db.query(DirectEvidenceRecord)
                        .filter(DirectEvidenceRecord.chat_session_id == target)
                        .filter(DirectEvidenceRecord.id == int(replacement_id))
                        .first()
                    )
                    if existing_target_replacement is not None:
                        resolved_target_superseded_by_id = int(existing_target_replacement.id)
                        replacement_resolution = "target_id_match"
                    else:
                        unresolved_superseded_hashes.add(source_hash)
                        replacement_resolution = "unresolved"

                merge_entry["resolved_target_superseded_by_id"] = resolved_target_superseded_by_id
                merge_entry["replacement_resolution"] = replacement_resolution

            merges.append(merge_entry)

    return {
        "merges": merges,
        "tombstone_hashes": sorted(tombstone_hashes),
        "superseded_hashes": sorted(superseded_hashes),
        "unresolved_superseded_hashes": sorted(unresolved_superseded_hashes),
    }


def _table_has_single_session_unique_key(table) -> bool:
    """chat_session_id 단일 고유키/인덱스 여부를 반환한다."""
    try:
        col = table.c.get("chat_session_id")
    except Exception:
        col = None

    if col is None:
        return False
    if getattr(col, "unique", False):
        return True

    for cons in list(getattr(table, "constraints", []) or []):
        if not isinstance(cons, UniqueConstraint):
            continue
        cols = [c.name for c in list(getattr(cons, "columns", []) or [])]
        if cols == ["chat_session_id"]:
            return True

    for idx in list(getattr(table, "indexes", []) or []):
        if not getattr(idx, "unique", False):
            continue
        cols = [c.name for c in list(getattr(idx, "columns", []) or [])]
        if cols == ["chat_session_id"]:
            return True

    return False


def _pick_session_unique_row_winner(table_name: str, source_row: dict | None, target_row: dict | None) -> str:
    """단일 고유 세션 행 충돌 시 source/target 중 유지할 쪽을 결정한다."""
    if source_row and not target_row:
        return "source"
    if target_row and not source_row:
        return "target"
    if not source_row and not target_row:
        return "target"

    # guidance_plan_states는 last_turn 우선으로 최신성을 판단한다.
    if table_name == "guidance_plan_states":
        s_last = int(source_row.get("last_turn") if source_row.get("last_turn") is not None else -1)
        t_last = int(target_row.get("last_turn") if target_row.get("last_turn") is not None else -1)
        if s_last != t_last:
            return "source" if s_last > t_last else "target"

    # 일반 규칙: updated/executed/created 시각이 더 최신인 행을 유지.
    for key in ("updated_at", "executed_at", "created_at"):
        s_ts = _to_epoch_seconds(source_row.get(key))
        t_ts = _to_epoch_seconds(target_row.get(key))
        if s_ts is None and t_ts is None:
            continue
        s_val = s_ts if s_ts is not None else float("-inf")
        t_val = t_ts if t_ts is not None else float("-inf")
        if s_val != t_val:
            return "source" if s_val > t_val else "target"

    # 보조 규칙: 정수 계열 메타데이터 비교.
    for key in ("turn_index", "source_turn", "compacted_turn", "id"):
        s_raw = source_row.get(key)
        t_raw = target_row.get(key)
        if s_raw is None and t_raw is None:
            continue
        try:
            s_num = float(s_raw) if s_raw is not None else float("-inf")
            t_num = float(t_raw) if t_raw is not None else float("-inf")
        except Exception:
            continue
        if s_num != t_num:
            return "source" if s_num > t_num else "target"

    # 동률이면 target(현재 canonical) 유지.
    return "target"


def _is_direct_evidence_retained_for_consumption(
    evidence_row: DirectEvidenceRecord,
    *,
    latest_turn_index: int | None,
) -> bool:
    """EA-1l: direct evidence 소비 시 importance/lineage retention window를 적용한다."""
    return is_within_retention_window(
        source_turn_end=evidence_row.source_turn_end,
        latest_turn_index=latest_turn_index,
        archive_state=evidence_row.archive_state,
        capture_verification=evidence_row.capture_verification,
        repair_needed=evidence_row.repair_needed,
        committed_gate=evidence_row.committed_gate,
        tombstoned=evidence_row.tombstoned,
        lineage=evidence_row.lineage_json,
    )


_OOC_PREFIX_RE = re.compile(
    r"^(?:/)?(?:ooc|out\s*of\s*character)\b(?:\s*[:：]|\s|$)|^#{1,6}\s*(?:ooc|out\s*of\s*character)\b",
    re.IGNORECASE,
)


def _is_ooc_text(text: str | None) -> bool:
    """plugin.js의 OOC 기준과 동일 계열로 배치 경로를 보호한다."""
    if not text or not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if _OOC_PREFIX_RE.match(s):
        return True
    if re.match(r"^\(\s*ooc\s*(?:[:：\)]|$)", s, re.IGNORECASE):
        return True
    if re.match(r"^\[\s*ooc\s*(?:[:：\]]|$)", s, re.IGNORECASE):
        return True
    if re.match(r"^\(\(\s*(?:ooc|out\s*of\s*character)\s*\)\)$", s, re.IGNORECASE):
        return True
    if re.match(r"^\[\[\s*(?:ooc|out\s*of\s*character)\s*\]\]$", s, re.IGNORECASE):
        return True
    if re.match(r"^<\s*ooc\s*>", s, re.IGNORECASE):
        return True
    return False


class PalaceBridge:
    """읽기 전용 DB Explorer 조회 계층."""

    # ────────────────────────────────────────────
    # Sessions
    # ────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        """저장된 session 목록을 반환한다.
        각 session에 대해 chat_logs, memories, kg_triples 건수와 최근 활동 시각을 포함한다.
        """
        db: Session = SessionLocal()
        try:
            # chat_logs 기준 session 목록 + 건수 + 최근 시각
            log_stats = (
                db.query(
                    ChatLog.chat_session_id,
                    func.count(ChatLog.id).label("log_count"),
                    func.max(ChatLog.created_at).label("last_log_at"),
                )
                .group_by(ChatLog.chat_session_id)
                .all()
            )

            mem_stats = dict(
                db.query(Memory.chat_session_id, func.count(Memory.id))
                .group_by(Memory.chat_session_id)
                .all()
            )

            kg_stats = dict(
                db.query(KGTriple.chat_session_id, func.count(KGTriple.id))
                .group_by(KGTriple.chat_session_id)
                .all()
            )

            # session 집합 (중복 제거)
            all_session_ids = set()
            log_map = {}
            for sid, cnt, last_at in log_stats:
                all_session_ids.add(sid)
                log_map[sid] = {"count": cnt, "last_at": str(last_at) if last_at else None}

            for sid in mem_stats:
                all_session_ids.add(sid)
            for sid in kg_stats:
                all_session_ids.add(sid)

            results = []
            for sid in sorted(all_session_ids):
                log_info = log_map.get(sid, {"count": 0, "last_at": None})
                results.append({
                    "chat_session_id": sid,
                    "chat_logs_count": log_info["count"],
                    "memories_count": mem_stats.get(sid, 0),
                    "kg_triples_count": kg_stats.get(sid, 0),
                    "last_activity": log_info["last_at"],
                })

            # 최근 활동순으로 정렬 (내림차순)
            results.sort(key=lambda x: x["last_activity"] or "", reverse=True)
            return results

        except Exception as exc:
            logger.error("list_sessions 실패: %s", exc)
            return []
        finally:
            db.close()

    # ────────────────────────────────────────────
    # Chat Logs
    # ────────────────────────────────────────────

    def get_chat_logs(
        self,
        chat_session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict:
        """chat_logs를 조회한다. session 필터가 있으면 해당 session만."""
        db: Session = SessionLocal()
        lim = _clamp_limit(limit)
        off = _clamp_offset(offset)
        sid = _safe_session_id(chat_session_id)

        try:
            q = db.query(ChatLog)
            if sid:
                q = q.filter(ChatLog.chat_session_id == sid)

            total = q.count()
            rows = (
                q.order_by(desc(ChatLog.turn_index), desc(ChatLog.id))
                .offset(off)
                .limit(lim)
                .all()
            )

            items = []
            for r in rows:
                content_str = r.content or ""
                items.append({
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "turn_index": r.turn_index,
                    "role": r.role,
                    "content": content_str,
                    "preview": content_str[:120] + ("…" if len(content_str) > 120 else ""),
                    "created_at": to_kst(r.created_at),
                })

            return {
                "items": items,
                "total": total,
                "limit": lim,
                "offset": off,
                "has_more": (off + lim) < total,
            }

        except Exception as exc:
            logger.error("get_chat_logs 실패: %s", exc)
            return {"items": [], "total": 0, "limit": lim, "offset": off, "has_more": False}
        finally:
            db.close()

    # ────────────────────────────────────────────
    # Memories
    # ────────────────────────────────────────────

    def get_memories(
        self,
        chat_session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict:
        """memories를 조회한다."""
        db: Session = SessionLocal()
        lim = _clamp_limit(limit)
        off = _clamp_offset(offset)
        sid = _safe_session_id(chat_session_id)

        try:
            q = db.query(Memory)
            if sid:
                q = q.filter(Memory.chat_session_id == sid)

            total = q.count()
            rows = (
                q.order_by(desc(Memory.created_at), desc(Memory.id))
                .offset(off)
                .limit(lim)
                .all()
            )

            items = []
            for r in rows:
                # summary_json 파싱 시도 — 실패 시 원문 그대로
                summary_raw = r.summary_json or ""
                summary_preview = ""
                try:
                    if summary_raw:
                        parsed = json.loads(summary_raw) if isinstance(summary_raw, str) else summary_raw
                        if isinstance(parsed, dict):
                            # 핵심 필드 추출
                            parts = []
                            for key in ("scene_summary", "core_meaning", "emotional_shift", "summary"):
                                if key in parsed and parsed[key]:
                                    parts.append(str(parsed[key])[:80])
                            summary_preview = " | ".join(parts) if parts else str(parsed)[:120]
                        else:
                            summary_preview = str(parsed)[:120]
                    else:
                        summary_preview = "(비어 있음)"
                except Exception:
                    summary_preview = summary_raw[:120] if summary_raw else "(파싱 실패)"

                items.append({
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "source_turn": r.turn_index,
                    "importance": r.importance,
                    "archive_wing": r.library_wing,
                    "archive_room": r.library_room,
                    "summary_json": summary_raw,
                    "summary_preview": summary_preview,
                    "embedding_model": getattr(r, "embedding_model", None),
                    # Phase 5-1: critic 확장 필드
                    "evidence": getattr(r, "evidence", None),
                    "emotional_intensity": getattr(r, "emotional_intensity", None),
                    "narrative_significance": getattr(r, "narrative_significance", None),
                    # Phase 5-2: importance boost trace
                    "emotional_boost": getattr(r, "emotional_boost", None),
                    "created_at": to_kst(r.created_at),
                })

            return {
                "items": items,
                "total": total,
                "limit": lim,
                "offset": off,
                "has_more": (off + lim) < total,
            }

        except Exception as exc:
            logger.error("get_memories 실패: %s", exc)
            return {"items": [], "total": 0, "limit": lim, "offset": off, "has_more": False}
        finally:
            db.close()

    def get_direct_evidence_records(
        self,
        chat_session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include_tombstoned: bool = True,
    ) -> dict:
        """direct_evidence_records를 조회한다."""
        db: Session = SessionLocal()
        lim = _clamp_limit(limit)
        off = _clamp_offset(offset)
        sid = _safe_session_id(chat_session_id)

        try:
            q = db.query(DirectEvidenceRecord)
            if sid:
                q = q.filter(DirectEvidenceRecord.chat_session_id == sid)
            if not include_tombstoned:
                q = q.filter(DirectEvidenceRecord.tombstoned.is_(False))

            latest_turn_index = None
            if sid:
                latest_turn_index = (
                    db.query(func.max(ChatLog.turn_index))
                    .filter(ChatLog.chat_session_id == sid)
                    .scalar()
                )

            total = q.count()
            rows = (
                q.order_by(desc(DirectEvidenceRecord.created_at), desc(DirectEvidenceRecord.id))
                .offset(off)
                .limit(lim)
                .all()
            )

            items = []
            for r in rows:
                text = r.evidence_text or ""
                try:
                    parsed_source_message_ids = json.loads(r.source_message_ids_json) if r.source_message_ids_json else []
                    if not isinstance(parsed_source_message_ids, list):
                        parsed_source_message_ids = []
                except Exception:
                    parsed_source_message_ids = []
                try:
                    parsed_lineage = json.loads(r.lineage_json) if r.lineage_json else {}
                    if not isinstance(parsed_lineage, dict):
                        parsed_lineage = {}
                except Exception:
                    parsed_lineage = {}
                normalized_archive_state = normalize_archive_state(r.archive_state)
                normalized_capture_verification = normalize_capture_verification(r.capture_verification)
                normalized_committed_gate = resolve_committed_gate(
                    normalized_capture_verification,
                    r.repair_needed,
                    r.committed_gate,
                )
                retention_importance_tier = resolve_retention_importance_tier(
                    archive_state=normalized_archive_state,
                    capture_verification=normalized_capture_verification,
                    repair_needed=r.repair_needed,
                    committed_gate=normalized_committed_gate,
                    tombstoned=r.tombstoned,
                    lineage=parsed_lineage,
                )
                retention_ttl_turns = resolve_retention_ttl_turns(
                    archive_state=normalized_archive_state,
                    capture_verification=normalized_capture_verification,
                    repair_needed=r.repair_needed,
                    committed_gate=normalized_committed_gate,
                    tombstoned=r.tombstoned,
                    lineage=parsed_lineage,
                )
                retention_expired = is_retention_expired(
                    source_turn_end=r.source_turn_end,
                    latest_turn_index=latest_turn_index,
                    archive_state=normalized_archive_state,
                    capture_verification=normalized_capture_verification,
                    repair_needed=r.repair_needed,
                    committed_gate=normalized_committed_gate,
                    tombstoned=r.tombstoned,
                    lineage=parsed_lineage,
                )
                archive_bucket = resolve_archive_bucket(
                    normalized_archive_state,
                    normalized_capture_verification,
                    r.repair_needed,
                )
                excluded_from_current_truth = is_current_truth_excluded(
                    tombstoned=r.tombstoned,
                    superseded_by_id=r.superseded_by_id,
                )
                tombstone_retained_in_window = is_deleted_turn_tombstone_retained(
                    tombstoned=r.tombstoned,
                    source_turn_end=r.source_turn_end,
                    latest_turn_index=latest_turn_index,
                    retention_window_turns=DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS,
                )
                items.append({
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "evidence_kind": r.evidence_kind,
                    "evidence_text": text,
                    "evidence_preview": text[:120] + ("…" if len(text) > 120 else ""),
                    "source_turn_start": r.source_turn_start,
                    "source_turn_end": r.source_turn_end,
                    "turn_anchor": r.turn_anchor,
                    "source_message_ids_json": r.source_message_ids_json,
                    "source_message_ids": parsed_source_message_ids,
                    "source_hash": r.source_hash,
                    "archive_state": r.archive_state,
                    "normalized_archive_state": normalized_archive_state,
                    "archive_bucket": archive_bucket,
                    "capture_stage": r.capture_stage,
                    "capture_verification": r.capture_verification,
                    "normalized_capture_verification": normalized_capture_verification,
                    "committed_gate": r.committed_gate,
                    "normalized_committed_gate": normalized_committed_gate,
                    "lineage_json": r.lineage_json,
                    "lineage": parsed_lineage,
                    "repair_needed": r.repair_needed,
                    "tombstoned": r.tombstoned,
                    "superseded_by_id": r.superseded_by_id,
                    "excluded_from_current_truth": excluded_from_current_truth,
                    "tombstone_retained_in_window": tombstone_retained_in_window,
                    "tombstone_retention_expired": bool(r.tombstoned) and not tombstone_retained_in_window,
                    "retention_policy_version": DIRECT_EVIDENCE_RETENTION_POLICY_VERSION,
                    "retention_importance_tier": retention_importance_tier,
                    "retention_ttl_turns": retention_ttl_turns,
                    "retention_expired": retention_expired,
                    "retention_blocked_from_consumption": bool(retention_expired),
                    "created_at": to_kst(r.created_at),
                })

            state_counts = build_state_counts(rows)
            cost_measurement = self._build_direct_evidence_cost_measurement(
                db=db,
                chat_session_id=sid,
                state_counts=state_counts,
            )

            return {
                "items": items,
                "total": total,
                "limit": lim,
                "offset": off,
                "has_more": (off + lim) < total,
                "latest_turn_index": latest_turn_index,
                "state_counts": state_counts,
                "cost_measurement": cost_measurement,
                "state_contract": {
                    "archive_states": list(DIRECT_EVIDENCE_ARCHIVE_STATES),
                    "capture_verifications": list(DIRECT_EVIDENCE_CAPTURE_VERIFICATIONS),
                    "committed_gates": list(DIRECT_EVIDENCE_COMMITTED_GATES),
                    "deleted_turn_tombstone_retention_window_turns": DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS,
                    "retention_policy_version": DIRECT_EVIDENCE_RETENTION_POLICY_VERSION,
                    "retention_importance_tiers": list(DIRECT_EVIDENCE_RETENTION_IMPORTANCE_TIERS),
                    "retention_windows_turns": DIRECT_EVIDENCE_RETENTION_WINDOWS_TURNS,
                    "cost_measurement_policy_version": "lc1a.v1",
                },
            }

        except Exception as exc:
            logger.error("get_direct_evidence_records 실패: %s", exc)
            return {"items": [], "total": 0, "limit": lim, "offset": off, "has_more": False}
        finally:
            db.close()

    def _build_direct_evidence_cost_measurement(
        self,
        *,
        db: Session,
        chat_session_id: str | None,
        state_counts: dict,
    ) -> dict:
        measurement = {
            "policy_version": "lc1a.v1",
            "audit_window_size": 200,
            "direct_evidence_write": {
                "sample_count": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "last_latency_ms": 0.0,
                "avg_inserted": 0.0,
                "avg_skipped": 0.0,
                "avg_write_chars": 0.0,
            },
            "repair_queue": {
                "queue_count": int(state_counts.get("repair_queue", 0) or 0),
                "review_sample_count": 0,
                "revalidate_sample_count": 0,
                "avg_review_latency_ms": 0.0,
                "avg_revalidate_latency_ms": 0.0,
                "last_revalidate_latency_ms": 0.0,
            },
        }

        if not chat_session_id:
            return measurement

        audit_rows = (
            db.query(AuditLog)
            .filter(AuditLog.chat_session_id == chat_session_id)
            .filter(
                AuditLog.event_type.in_(
                    ["critic_ingest_trace", "direct_evidence_review", "direct_evidence_revalidate"]
                )
            )
            .order_by(desc(AuditLog.id))
            .limit(measurement["audit_window_size"])
            .all()
        )

        write_latencies: list[float] = []
        write_inserted: list[float] = []
        write_skipped: list[float] = []
        write_chars: list[float] = []
        review_latencies: list[float] = []
        revalidate_latencies: list[float] = []

        for row in audit_rows:
            details = _safe_json_dict(row.details_json)
            if row.event_type == "critic_ingest_trace":
                if str(details.get("surface") or "") != "direct_evidence":
                    continue
                trace = details.get("trace") if isinstance(details.get("trace"), dict) else {}
                latency_ms = float(trace.get("elapsed_ms") or 0.0)
                write_latencies.append(latency_ms)
                write_inserted.append(float(trace.get("inserted") or 0.0))
                write_skipped.append(float(trace.get("skipped") or 0.0))
                write_chars.append(float(trace.get("write_chars") or 0.0))
                continue

            if row.event_type == "direct_evidence_review":
                cm = details.get("cost_measurement") if isinstance(details.get("cost_measurement"), dict) else {}
                review_latencies.append(float(cm.get("latency_ms") or 0.0))
                continue

            if row.event_type == "direct_evidence_revalidate":
                cm = details.get("cost_measurement") if isinstance(details.get("cost_measurement"), dict) else {}
                revalidate_latencies.append(float(cm.get("latency_ms") or 0.0))

        if write_latencies:
            measurement["direct_evidence_write"].update({
                "sample_count": len(write_latencies),
                "avg_latency_ms": _safe_mean(write_latencies),
                "p95_latency_ms": _safe_p95(write_latencies),
                "last_latency_ms": _safe_round(write_latencies[0], 3),
                "avg_inserted": _safe_mean(write_inserted),
                "avg_skipped": _safe_mean(write_skipped),
                "avg_write_chars": _safe_mean(write_chars),
            })

        if review_latencies:
            measurement["repair_queue"].update({
                "review_sample_count": len(review_latencies),
                "avg_review_latency_ms": _safe_mean(review_latencies),
            })

        if revalidate_latencies:
            measurement["repair_queue"].update({
                "revalidate_sample_count": len(revalidate_latencies),
                "avg_revalidate_latency_ms": _safe_mean(revalidate_latencies),
                "last_revalidate_latency_ms": _safe_round(revalidate_latencies[0], 3),
            })

        return measurement

    # ────────────────────────────────────────────
    # KG Triples
    # ────────────────────────────────────────────

    def get_kg_triples(
        self,
        chat_session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict:
        """kg_triples를 조회한다."""
        db: Session = SessionLocal()
        lim = _clamp_limit(limit)
        off = _clamp_offset(offset)
        sid = _safe_session_id(chat_session_id)

        try:
            q = db.query(KGTriple)
            if sid:
                q = q.filter(KGTriple.chat_session_id == sid)

            total = q.count()
            rows = (
                q.order_by(desc(KGTriple.created_at), desc(KGTriple.id))
                .offset(off)
                .limit(lim)
                .all()
            )

            items = []
            for r in rows:
                items.append({
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "valid_from": r.valid_from,
                    "valid_to": r.valid_to,
                    "source_turn": r.source_turn,
                    "created_at": to_kst(r.created_at),
                })

            return {
                "items": items,
                "total": total,
                "limit": lim,
                "offset": off,
                "has_more": (off + lim) < total,
            }

        except Exception as exc:
            logger.error("get_kg_triples 실패: %s", exc)
            return {"items": [], "total": 0, "limit": lim, "offset": off, "has_more": False}
        finally:
            db.close()

    # ────────────────────────────────────────────
    # Chapter Summaries
    # ────────────────────────────────────────────

    def get_chapter_summaries(
        self,
        chat_session_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict:
        """chapter_summaries를 읽기 전용 explorer surface로 조회한다."""
        db: Session = SessionLocal()
        lim = _clamp_limit(limit)
        off = _clamp_offset(offset)
        sid = _safe_session_id(chat_session_id)

        try:
            q = db.query(ChapterSummary)
            if sid:
                q = q.filter(ChapterSummary.chat_session_id == sid)

            total = q.count()
            rows = (
                q.order_by(desc(ChapterSummary.to_turn), desc(ChapterSummary.id))
                .offset(off)
                .limit(lim)
                .all()
            )

            items = []
            for r in rows:
                item = _chapter_summary_to_dict(r)
                summary_text = item.get("summary_text") or ""
                item["summary_preview"] = summary_text[:160] + ("..." if len(summary_text) > 160 else "")
                items.append(item)

            return {
                "items": items,
                "total": total,
                "limit": lim,
                "offset": off,
                "has_more": (off + lim) < total,
            }

        except Exception as exc:
            logger.error("get_chapter_summaries 실패: %s", exc)
            return {"items": [], "total": 0, "limit": lim, "offset": off, "has_more": False}
        finally:
            db.close()

    # ────────────────────────────────────────────
    # Sprint 2-F: Memory 제한적 수정
    # ────────────────────────────────────────────

    # 수정 허용 필드 (화이트리스트)
    MEMORY_EDITABLE_FIELDS = {"summary_json", "importance", "archive_wing", "archive_room"}
    KG_EDITABLE_FIELDS = {"subject", "predicate", "object", "valid_from", "valid_to"}

    def update_memory(
        self,
        memory_id: int,
        chat_session_id: str,
        fields: dict,
    ) -> dict:
        """memory row를 제한적으로 수정한다.

        - memory_id에 해당하는 row가 존재해야 한다.
        - row의 chat_session_id가 요청과 일치해야 한다.
        - 허용된 필드만 수정된다.
        - summary_json이 포함되면 JSON 유효성 검증을 수행한다.

        Returns:
            {"ok": True/False, "detail": str, "updated_fields": [...]}
        """
        sid = _safe_session_id(chat_session_id)
        if not sid:
            return {"ok": False, "detail": "chat_session_id가 필요합니다.", "updated_fields": []}

        # 허용된 필드만 필터링
        allowed = {k: v for k, v in fields.items() if k in self.MEMORY_EDITABLE_FIELDS}
        if not allowed:
            return {"ok": False, "detail": "수정 가능한 필드가 없습니다.", "updated_fields": []}

        # summary_json JSON 유효성 검증
        if "summary_json" in allowed:
            raw = allowed["summary_json"]
            if raw is not None and isinstance(raw, str) and raw.strip():
                try:
                    json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    return {"ok": False, "detail": "summary_json이 유효한 JSON이 아닙니다.", "updated_fields": []}

        # importance 숫자 sanitize
        if "importance" in allowed:
            try:
                val = float(allowed["importance"]) if allowed["importance"] is not None else None
                if val is not None:
                    val = max(0.0, min(10.0, val))
                allowed["importance"] = val
            except (ValueError, TypeError):
                return {"ok": False, "detail": "importance는 숫자여야 합니다.", "updated_fields": []}

        # archive_wing / archive_room 문자열 sanitize
        for fld in ("archive_wing", "archive_room"):
            if fld in allowed:
                val = allowed[fld]
                allowed[fld] = str(val).strip()[:255] if val is not None else None

        db: Session = SessionLocal()
        try:
            row = db.query(Memory).filter(Memory.id == memory_id).first()
            if not row:
                return {"ok": False, "detail": f"Memory #{memory_id}를 찾을 수 없습니다.", "updated_fields": []}

            if row.chat_session_id != sid:
                return {"ok": False, "detail": "세션 불일치: 다른 세션의 데이터는 수정할 수 없습니다.", "updated_fields": []}

            updated = []
            if "summary_json" in allowed:
                row.summary_json = allowed["summary_json"]
                updated.append("summary_json")
            if "importance" in allowed:
                row.importance = allowed["importance"]
                updated.append("importance")
            if "archive_wing" in allowed:
                row.library_wing = allowed["archive_wing"]
                updated.append("archive_wing")
            if "archive_room" in allowed:
                row.library_room = allowed["archive_room"]
                updated.append("archive_room")

            db.commit()
            logger.info("Memory #%d 수정 완료 (session=%s, fields=%s)", memory_id, sid, updated)
            return {"ok": True, "detail": "수정 완료", "updated_fields": updated}

        except Exception as exc:
            db.rollback()
            logger.error("update_memory 실패 (id=%d): %s", memory_id, exc)
            return {"ok": False, "detail": f"DB 오류: {exc}", "updated_fields": []}
        finally:
            db.close()

    # ────────────────────────────────────────────
    # Sprint 2-F: KG Triple 제한적 수정
    # ────────────────────────────────────────────

    def update_kg_triple(
        self,
        triple_id: int,
        chat_session_id: str,
        fields: dict,
    ) -> dict:
        """kg_triple row를 제한적으로 수정한다.

        - triple_id에 해당하는 row가 존재해야 한다.
        - row의 chat_session_id가 요청과 일치해야 한다.
        - 허용된 필드만 수정된다.

        Returns:
            {"ok": True/False, "detail": str, "updated_fields": [...]}
        """
        sid = _safe_session_id(chat_session_id)
        if not sid:
            return {"ok": False, "detail": "chat_session_id가 필요합니다.", "updated_fields": []}

        allowed = {k: v for k, v in fields.items() if k in self.KG_EDITABLE_FIELDS}
        if not allowed:
            return {"ok": False, "detail": "수정 가능한 필드가 없습니다.", "updated_fields": []}

        # subject/object는 canonical token으로 정규화, predicate는 trim만 수행
        for fld in ("subject", "object"):
            if fld in allowed:
                val = _normalize_kg_entity_label(allowed[fld], allow_player_token=True)
                if not val:
                    return {"ok": False, "detail": f"{fld}은(는) 유효한 엔티티여야 합니다.", "updated_fields": []}
                allowed[fld] = val
        if "predicate" in allowed:
            val = str(allowed["predicate"]).strip()[:255] if allowed["predicate"] is not None else ""
            if not val:
                return {"ok": False, "detail": "predicate은(는) 비어 있을 수 없습니다.", "updated_fields": []}
            allowed["predicate"] = val

        # valid_from, valid_to — 정수 또는 None
        for fld in ("valid_from", "valid_to"):
            if fld in allowed:
                val = allowed[fld]
                if val is None or val == "" or val == "null":
                    allowed[fld] = None
                else:
                    try:
                        allowed[fld] = int(val)
                    except (ValueError, TypeError):
                        return {"ok": False, "detail": f"{fld}은(는) 정수 또는 빈 값이어야 합니다.", "updated_fields": []}

        db: Session = SessionLocal()
        try:
            row = db.query(KGTriple).filter(KGTriple.id == triple_id).first()
            if not row:
                return {"ok": False, "detail": f"KGTriple #{triple_id}을(를) 찾을 수 없습니다.", "updated_fields": []}

            if row.chat_session_id != sid:
                return {"ok": False, "detail": "세션 불일치: 다른 세션의 데이터는 수정할 수 없습니다.", "updated_fields": []}

            updated = []
            for fld in ("subject", "predicate", "object", "valid_from", "valid_to"):
                if fld in allowed:
                    setattr(row, fld, allowed[fld])
                    updated.append(fld)

            db.commit()
            logger.info("KGTriple #%d 수정 완료 (session=%s, fields=%s)", triple_id, sid, updated)
            return {"ok": True, "detail": "수정 완료", "updated_fields": updated}

        except Exception as exc:
            db.rollback()
            logger.error("update_kg_triple 실패 (id=%d): %s", triple_id, exc)
            return {"ok": False, "detail": f"DB 오류: {exc}", "updated_fields": []}
        finally:
            db.close()

    # ────────────────────────────────────────────
    # Sprint 3-A-1: Memory 삭제 (hard delete)
    # ────────────────────────────────────────────

    def delete_memory(self, memory_id: int, chat_session_id: str) -> dict:
        """memory row를 hard delete한다.

        - memory_id에 해당하는 row가 존재해야 한다.
        - row의 chat_session_id가 요청과 일치해야 한다.

        Returns:
            {"ok": True/False, "detail": str, "deleted_id": int|None, "deleted_type": str}
        """
        sid = _safe_session_id(chat_session_id)
        if not sid:
            return {"ok": False, "detail": "chat_session_id가 필요합니다.", "deleted_id": None, "deleted_type": "memory"}

        db: Session = SessionLocal()
        try:
            row = db.query(Memory).filter(Memory.id == memory_id).first()
            if not row:
                return {"ok": False, "detail": f"Memory #{memory_id}를 찾을 수 없습니다.", "deleted_id": None, "deleted_type": "memory"}

            if row.chat_session_id != sid:
                return {"ok": False, "detail": "세션 불일치: 다른 세션의 데이터는 삭제할 수 없습니다.", "deleted_id": None, "deleted_type": "memory"}

            db.delete(row)
            db.commit()
            logger.info("Memory #%d 삭제 완료 (session=%s)", memory_id, sid)
            return {"ok": True, "detail": "삭제 완료", "deleted_id": memory_id, "deleted_type": "memory"}

        except Exception as exc:
            db.rollback()
            logger.error("delete_memory 실패 (id=%d): %s", memory_id, exc)
            return {"ok": False, "detail": f"DB 오류: {exc}", "deleted_id": None, "deleted_type": "memory"}
        finally:
            db.close()

    # ────────────────────────────────────────────
    # Sprint 3-A-1: KG Triple 삭제 (hard delete)
    # ────────────────────────────────────────────

    def delete_kg_triple(self, triple_id: int, chat_session_id: str) -> dict:
        """kg_triple row를 hard delete한다.

        - triple_id에 해당하는 row가 존재해야 한다.
        - row의 chat_session_id가 요청과 일치해야 한다.

        Returns:
            {"ok": True/False, "detail": str, "deleted_id": int|None, "deleted_type": str}
        """
        sid = _safe_session_id(chat_session_id)
        if not sid:
            return {"ok": False, "detail": "chat_session_id가 필요합니다.", "deleted_id": None, "deleted_type": "kg_triple"}

        db: Session = SessionLocal()
        try:
            row = db.query(KGTriple).filter(KGTriple.id == triple_id).first()
            if not row:
                return {"ok": False, "detail": f"KGTriple #{triple_id}을(를) 찾을 수 없습니다.", "deleted_id": None, "deleted_type": "kg_triple"}

            if row.chat_session_id != sid:
                return {"ok": False, "detail": "세션 불일치: 다른 세션의 데이터는 삭제할 수 없습니다.", "deleted_id": None, "deleted_type": "kg_triple"}

            db.delete(row)
            db.commit()
            logger.info("KGTriple #%d 삭제 완료 (session=%s)", triple_id, sid)
            return {"ok": True, "detail": "삭제 완료", "deleted_id": triple_id, "deleted_type": "kg_triple"}

        except Exception as exc:
            db.rollback()
            logger.error("delete_kg_triple 실패 (id=%d): %s", triple_id, exc)
            return {"ok": False, "detail": f"DB 오류: {exc}", "deleted_id": None, "deleted_type": "kg_triple"}
        finally:
            db.close()

    # ────────────────────────────────────────────
    # Phase 1-3: KG Recall (entity-name 기반)
    # ────────────────────────────────────────────

    def recall_kg_triples(
        self,
        chat_session_id: str,
        entities: list[str],
        limit: int = 20,
        current_turn: int | None = None,
    ) -> list[dict]:
        """현재 session에서 entities에 매칭되는 KG triple을 recall한다.

        매칭 규칙 (MVP):
        - subject 또는 object가 entity와 case-insensitive / trim 매치
        - valid_to가 None이거나 현재 유효한 triple 우선
        - 최근 생성순 정렬
        - 중복 triple 제거 (subject+predicate+object 키 기준)

        Parameters
        ----------
        chat_session_id : str
            세션 ID (필수). 다른 세션의 triple은 절대 반환하지 않는다.
        entities : list[str]
            매칭할 entity 이름 목록.
        limit : int
            최대 반환 건수 (1~50).

        Returns
        -------
        list[dict]  각 triple의 필드를 포함한 dict 리스트.
        """
        sid = _safe_session_id(chat_session_id)
        if not sid:
            return []
        if not entities or not isinstance(entities, list):
            return []

        # normalize entity names
        normalized = []
        for e in entities:
            if isinstance(e, str):
                n = e.strip()
                if n:
                    normalized.append(n.lower())
        if not normalized:
            return []

        limit = max(1, min(50, limit)) if isinstance(limit, int) and limit > 0 else 20

        db: Session = SessionLocal()
        try:
            rows = (
                db.query(KGTriple)
                .filter(KGTriple.chat_session_id == sid)
                .order_by(desc(KGTriple.created_at), desc(KGTriple.id))
                .all()
            )

            matched = []
            seen_keys = set()
            for r in rows:
                subj_lower = (r.subject or "").strip().lower()
                obj_lower = (r.object or "").strip().lower()

                hit = False
                for n in normalized:
                    # exact match (기존)
                    if n == subj_lower or n == obj_lower:
                        hit = True
                        break
                    # substring match: entity가 subject/object에 포함되거나 그 반대
                    if len(n) >= 2 and (n in subj_lower or n in obj_lower or subj_lower in n or obj_lower in n):
                        hit = True
                        break
                if not hit:
                    continue

                # Sprint 4-C-1: valid_to 필터링 (current_turn 전달 시)
                if current_turn is not None and r.valid_to is not None:
                    if r.valid_to < current_turn:
                        continue

                # 중복 제거: subject+predicate+object
                dedup_key = f"{subj_lower}|{(r.predicate or '').strip().lower()}|{obj_lower}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                matched.append({
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "valid_from": r.valid_from,
                    "valid_to": r.valid_to,
                    "source_turn": r.source_turn,
                    "created_at": to_kst(r.created_at),
                })

                if len(matched) >= limit:
                    break

            # valid_to가 None(현재 유효)인 triple을 앞에 배치
            matched.sort(key=lambda t: (0 if t["valid_to"] is None else 1, -(t["source_turn"] or 0)))
            return matched[:limit]

        except Exception as exc:
            logger.error("recall_kg_triples 실패 (session=%s): %s", sid, exc)
            return []
        finally:
            db.close()

    # ────────────────────────────────────────────
    # Sprint 3-C-2: Session Export (읽기 전용)
    # ────────────────────────────────────────────

    def export_session(self, chat_session_id: str) -> dict:
        """단일 session의 chat_logs, memories, kg_triples를 JSON export 구조로 반환한다.

        읽기 전용. session boundary를 절대 넘지 않는다.
        embedding raw 벡터는 제외한다 (민감정보/크기).

        Returns:
            {"ok": True, "data": {...}} 또는
            {"ok": False, "detail": str}
        """
        sid = _safe_session_id(chat_session_id)
        if not sid:
            return {"ok": False, "detail": "chat_session_id가 필요합니다."}

        db: Session = SessionLocal()
        try:
            from datetime import datetime, timezone
            current_embedding_model = (settings.PROJECT_EMBEDDING_MODEL or "").strip()

            # ── chat_logs ──
            log_rows = (
                db.query(ChatLog)
                .filter(ChatLog.chat_session_id == sid)
                .order_by(ChatLog.turn_index, ChatLog.id)
                .all()
            )
            chat_logs = []
            for r in log_rows:
                chat_logs.append({
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "turn_index": r.turn_index,
                    "role": r.role,
                    "content": r.content or "",
                    "created_at": to_kst(r.created_at),
                })

            # ── memories ──
            mem_rows = (
                db.query(Memory)
                .filter(Memory.chat_session_id == sid)
                .order_by(Memory.turn_index, Memory.id)
                .all()
            )
            memories = []
            memory_provenance_status_counts: dict[str, int] = {}
            memory_needs_reembed_count = 0
            for r in mem_rows:
                reembed_state = _assess_memory_reembed_state(
                    embedding_value=r.embedding,
                    embedding_model=getattr(r, "embedding_model", None),
                    current_model=current_embedding_model,
                )
                state_key = str(reembed_state["status"])
                memory_provenance_status_counts[state_key] = memory_provenance_status_counts.get(state_key, 0) + 1
                if reembed_state.get("needs_reembed") is True:
                    memory_needs_reembed_count += 1
                memories.append({
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "source_turn": r.turn_index,
                    "summary_json": r.summary_json or "",
                    "importance": r.importance,
                    "archive_wing": r.library_wing,
                    "archive_room": r.library_room,
                    "embedding_model": getattr(r, "embedding_model", None),
                    "embedding_provenance_status": reembed_state["status"],
                    "needs_reembed": reembed_state["needs_reembed"],
                    "created_at": to_kst(r.created_at),
                })

            effective_input_rows = (
                db.query(EffectiveInputLog)
                .filter(EffectiveInputLog.chat_session_id == sid)
                .order_by(EffectiveInputLog.turn_index, EffectiveInputLog.id)
                .all()
            )
            effective_inputs = [
                {
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "turn_index": r.turn_index,
                    "effective_input": r.effective_input or "",
                    "created_at": to_kst(r.created_at),
                }
                for r in effective_input_rows
            ]

            evidence_rows = (
                db.query(DirectEvidenceRecord)
                .filter(DirectEvidenceRecord.chat_session_id == sid)
                .order_by(DirectEvidenceRecord.source_turn_end, DirectEvidenceRecord.id)
                .all()
            )
            direct_evidence_records = []
            for r in evidence_rows:
                try:
                    parsed_source_message_ids = json.loads(r.source_message_ids_json) if r.source_message_ids_json else []
                    if not isinstance(parsed_source_message_ids, list):
                        parsed_source_message_ids = []
                except Exception:
                    parsed_source_message_ids = []
                parsed_lineage = _safe_json_dict(r.lineage_json)
                direct_evidence_records.append({
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "evidence_kind": r.evidence_kind,
                    "evidence_text": r.evidence_text or "",
                    "source_turn_start": r.source_turn_start,
                    "source_turn_end": r.source_turn_end,
                    "turn_anchor": r.turn_anchor,
                    "source_message_ids": parsed_source_message_ids,
                    "source_hash": r.source_hash,
                    "archive_state": r.archive_state,
                    "capture_stage": r.capture_stage,
                    "capture_verification": r.capture_verification,
                    "committed_gate": r.committed_gate,
                    "lineage": parsed_lineage,
                    "repair_needed": bool(r.repair_needed),
                    "tombstoned": bool(r.tombstoned),
                    "superseded_by_id": r.superseded_by_id,
                    "created_at": to_kst(r.created_at),
                })

            canonical_rows = (
                db.query(CanonicalStateLayer)
                .filter(CanonicalStateLayer.chat_session_id == sid)
                .order_by(CanonicalStateLayer.turn_index, CanonicalStateLayer.id)
                .all()
            )
            canonical_state_layers = [
                {
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "layer_type": r.layer_type,
                    "content_json": r.content or "",
                    "source_state_type": r.source_state_type,
                    "turn_index": r.turn_index,
                    "source_turn": r.source_turn,
                    "source_record": r.source_record,
                    "last_verified_turn": r.last_verified_turn,
                    "confidence": r.confidence,
                    "created_at": to_kst(r.created_at),
                }
                for r in canonical_rows
            ]

            direct_evidence_source_hash_count = sum(1 for row in direct_evidence_records if row.get("source_hash"))
            direct_evidence_tombstoned_count = sum(1 for row in direct_evidence_records if row.get("tombstoned"))
            direct_evidence_superseded_count = sum(1 for row in direct_evidence_records if row.get("superseded_by_id") is not None)
            canonical_with_source_turn_count = sum(1 for row in canonical_state_layers if row.get("source_turn") is not None)
            canonical_with_source_record_count = sum(1 for row in canonical_state_layers if row.get("source_record") is not None)

            # ── kg_triples ──
            kg_rows = (
                db.query(KGTriple)
                .filter(KGTriple.chat_session_id == sid)
                .order_by(KGTriple.source_turn, KGTriple.id)
                .all()
            )
            kg_triples = []
            for r in kg_rows:
                kg_triples.append({
                    "id": r.id,
                    "chat_session_id": r.chat_session_id,
                    "source_turn": r.source_turn,
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "valid_from": r.valid_from,
                    "valid_to": r.valid_to,
                    "created_at": to_kst(r.created_at),
                })

            # ── chapter_summaries ──
            chapter_rows = (
                db.query(ChapterSummary)
                .filter(ChapterSummary.chat_session_id == sid)
                .order_by(ChapterSummary.from_turn, ChapterSummary.id)
                .all()
            )
            chapter_summaries = [_chapter_summary_to_dict(r) for r in chapter_rows]

            arc_rows = (
                db.query(ArcSummary)
                .filter(ArcSummary.chat_session_id == sid)
                .order_by(ArcSummary.from_turn, ArcSummary.id)
                .all()
            )
            arc_summaries = [_arc_summary_to_dict(r) for r in arc_rows]

            saga_rows = (
                db.query(SagaDigest)
                .filter(SagaDigest.chat_session_id == sid)
                .order_by(SagaDigest.from_turn, SagaDigest.id)
                .all()
            )
            saga_digests = [_saga_digest_to_dict(r) for r in saga_rows]

            export_data = {
                "export_version": "1.1",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "chat_session_id": sid,
                "summary": {
                    "chat_logs_count": len(chat_logs),
                    "effective_inputs_count": len(effective_inputs),
                    "memories_count": len(memories),
                    "direct_evidence_records_count": len(direct_evidence_records),
                    "canonical_state_layers_count": len(canonical_state_layers),
                    "kg_triples_count": len(kg_triples),
                    "chapter_summaries_count": len(chapter_summaries),
                    "arc_summaries_count": len(arc_summaries),
                    "saga_digests_count": len(saga_digests),
                },
                "portability_contract": {
                    "package_policy_version": _STEP13_PORTABILITY_PACKAGE_POLICY_VERSION,
                    "rebuild_handoff_policy_version": _STEP13_PORTABILITY_REBUILD_HANDOFF_POLICY_VERSION,
                    "operation_policy_version": _STEP13_PORTABILITY_OPERATION_POLICY_VERSION,
                    "package_mode": "manual_db_snapshot_export",
                    "manual_first": True,
                    "auto_copy_detection": "deferred",
                    "session_origin": sid,
                    "portable_units": [
                        "chat_logs",
                        "effective_inputs",
                        "memories",
                        "direct_evidence_records",
                        "canonical_state_layers",
                        "kg_triples",
                        "chapter_summaries",
                        "arc_summaries",
                        "saga_digests",
                        "guidance_snapshot",
                    ],
                    "lineage_surfaces": [
                        "direct_evidence_records.source_hash",
                        "direct_evidence_records.source_turn_start",
                        "direct_evidence_records.source_turn_end",
                        "direct_evidence_records.turn_anchor",
                        "direct_evidence_records.lineage",
                        "direct_evidence_records.tombstoned",
                        "direct_evidence_records.superseded_by_id",
                        "canonical_state_layers.source_turn",
                        "canonical_state_layers.source_record",
                        "canonical_state_layers.last_verified_turn",
                    ],
                    "rebuild_handoff": {
                        "dirty_event_type": "backfill_import",
                        "rebuild_mode": "selective",
                        "start_point": "next_prepare_turn_fetch",
                        "source_rows": [
                            "chat_logs",
                            "effective_input_logs",
                            "direct_evidence_records",
                            "canonical_state_layers",
                        ],
                        "rebuild_targets": [
                            "direct_evidence",
                            "canonical_state",
                            "dense_summary",
                            "sidecar",
                        ],
                        "inherited_runtime_policy_versions": {
                            "dirty_matrix": "or1h.v1",
                            "rebuild": "or1i.v1",
                            "stale_serving_guard": "or1j.v1",
                        },
                    },
                },
                "embedding_provenance": {
                    "policy_version": _STEP13_EMBEDDING_PROVENANCE_POLICY_VERSION,
                    "needs_reembed_policy_version": _STEP13_NEEDS_REEMBED_POLICY_VERSION,
                    "current_project_embedding_model": current_embedding_model or None,
                    "memory_status_counts": memory_provenance_status_counts,
                    "needs_reembed_count": memory_needs_reembed_count,
                },
                "lineage_summary": {
                    "direct_evidence_source_hash_count": direct_evidence_source_hash_count,
                    "direct_evidence_tombstoned_count": direct_evidence_tombstoned_count,
                    "direct_evidence_superseded_count": direct_evidence_superseded_count,
                    "canonical_layers_with_source_turn_count": canonical_with_source_turn_count,
                    "canonical_layers_with_source_record_count": canonical_with_source_record_count,
                },
                "chat_logs": chat_logs,
                "effective_inputs": effective_inputs,
                "memories": memories,
                "direct_evidence_records": direct_evidence_records,
                "canonical_state_layers": canonical_state_layers,
                "kg_triples": kg_triples,
                "chapter_summaries": chapter_summaries,
                "arc_summaries": arc_summaries,
                "saga_digests": saga_digests,
            }

            # ── L-3b: guidance snapshot 포함 ─────────────────────────────
            try:
                gps: GuidancePlanState | None = (
                    db.query(GuidancePlanState)
                    .filter(GuidancePlanState.chat_session_id == sid)
                    .first()
                )
                if gps:
                    _sp = {}
                    _dir = {}
                    try:
                        _sp = json.loads(gps.story_plan_json or "{}")
                    except Exception:
                        pass
                    try:
                        _dir = json.loads(gps.director_json or "{}")
                    except Exception:
                        pass
                    compact_rows = (
                        db.query(GuidanceCompactRecord)
                        .filter(GuidanceCompactRecord.chat_session_id == sid)
                        .order_by(
                            GuidanceCompactRecord.compacted_turn.desc().nullslast(),
                            GuidanceCompactRecord.id.desc(),
                        )
                        .limit(20)
                        .all()
                    )
                    mnt_row: MaintenancePassState | None = (
                        db.query(MaintenancePassState)
                        .filter(MaintenancePassState.chat_session_id == sid)
                        .order_by(MaintenancePassState.turn_index.desc(), MaintenancePassState.id.desc())
                        .first()
                    )
                    export_data["guidance_snapshot"] = {
                        "state_status": gps.state_status or "empty",
                        "last_turn": gps.last_turn if gps.last_turn is not None else -1,
                        "story_plan": _sp,
                        "director": _dir,
                        "compact_records": [
                            {
                                "record_type": r.record_type or "",
                                "title": r.title or "",
                                "compact_summary": r.compact_summary,
                                "origin_type": r.origin_type,
                                "source_turn": r.source_turn,
                                "compacted_turn": r.compacted_turn,
                                "emotional_weight": r.emotional_weight,
                            }
                            for r in compact_rows
                        ],
                        "maintenance_last": {
                            "turn_index": mnt_row.turn_index,
                            "status": mnt_row.status or "ok",
                            "shadow_only": bool(mnt_row.shadow_only),
                        } if mnt_row else None,
                    }
                    export_data["summary"]["guidance_compact_records_count"] = len(compact_rows)
                else:
                    export_data["guidance_snapshot"] = None
            except Exception as g_exc:
                logger.warning("export_session guidance snapshot 포함 실패 (non-fatal): %s", g_exc)
                export_data["guidance_snapshot"] = None
            # ─────────────────────────────────────────────────────────────

            logger.info(
                "Session export 완료 (session=%s, logs=%d, mem=%d, kg=%d, chapters=%d, arcs=%d, sagas=%d)",
                sid, len(chat_logs), len(memories), len(kg_triples), len(chapter_summaries), len(arc_summaries), len(saga_digests),
            )
            return {"ok": True, "data": export_data}

        except Exception as exc:
            logger.error("export_session 실패 (session=%s): %s", sid, exc)
            return {"ok": False, "detail": f"DB 오류: {exc}"}
        finally:
            db.close()

    # ================================================================== #
    #  Sprint 3-D — Reindex Batch
    # ================================================================== #

    def reindex_session(
        self,
        chat_session_id: str,
        force: bool = False,
        batch_size: int = 20,
        max_items: int = 200,
    ) -> dict:
        """세션의 memory embedding을 현재 모델로 재생성한다.

        Parameters
        ----------
        chat_session_id : str
            대상 세션 ID (필수).
        force : bool
            True이면 이미 현재 모델과 같은 row도 재생성.
        batch_size : int
            DB commit 단위 (1~100).
        max_items : int
            한 번에 처리할 최대 row 수 (1~500).

        Returns
        -------
        dict  with keys: ok, total, processed, succeeded, failed, skipped, failed_ids, model
        """
        sid = (chat_session_id or "").strip()
        if not sid:
            return {"ok": False, "detail": "chat_session_id가 필요합니다."}

        current_model = (settings.PROJECT_EMBEDDING_MODEL or "").strip()
        if not current_model:
            return {"ok": False, "detail": "PROJECT_EMBEDDING_MODEL이 설정되지 않았습니다."}

        batch_size = max(1, min(100, batch_size))
        max_items = max(1, min(500, max_items))

        embedder = ArchiveBridge()

        db: Session = SessionLocal()
        try:
            session_rows = (
                db.query(Memory)
                .filter(Memory.chat_session_id == sid)
                .order_by(Memory.id)
                .all()
            )
            total_in_session = len(session_rows)

            provenance_status_counts: dict[str, int] = {}
            candidate_reason_counts: dict[str, int] = {}
            natural_candidate_rows = []
            natural_candidate_reason_counts: dict[str, int] = {}

            for row in session_rows:
                reembed_state = _assess_memory_reembed_state(
                    embedding_value=row.embedding,
                    embedding_model=getattr(row, "embedding_model", None),
                    current_model=current_model,
                )
                status_key = str(reembed_state["status"])
                provenance_status_counts[status_key] = provenance_status_counts.get(status_key, 0) + 1
                if reembed_state.get("needs_reembed") is True:
                    natural_candidate_rows.append((row, reembed_state))
                    reason_key = str(reembed_state["reason"])
                    natural_candidate_reason_counts[reason_key] = natural_candidate_reason_counts.get(reason_key, 0) + 1

            if force:
                candidates = [(row, _assess_memory_reembed_state(
                    embedding_value=row.embedding,
                    embedding_model=getattr(row, "embedding_model", None),
                    current_model=current_model,
                )) for row in session_rows[:max_items]]
                candidate_reason_counts = dict(provenance_status_counts)
                selection_mode = "force_all"
            else:
                candidates = natural_candidate_rows[:max_items]
                candidate_reason_counts = dict(natural_candidate_reason_counts)
                selection_mode = "needs_reembed_only"

            # OOC 사용자 턴은 재색인 대상에서 제외한다.
            ooc_turn_rows = (
                db.query(ChatLog.turn_index, ChatLog.content)
                .filter(
                    ChatLog.chat_session_id == sid,
                    func.lower(ChatLog.role) == "user",
                )
                .all()
            )
            ooc_turn_indices = {
                int(ti)
                for ti, content in ooc_turn_rows
                if ti is not None and _is_ooc_text(content)
            }

            succeeded = 0
            failed = 0
            skipped = 0
            ooc_skipped = 0
            failed_ids: list[int] = []
            updated_turn_indices: list[int] = []

            for i in range(0, len(candidates), batch_size):
                batch = candidates[i : i + batch_size]
                for row, _reembed_state in batch:
                    if row.turn_index in ooc_turn_indices:
                        skipped += 1
                        ooc_skipped += 1
                        continue

                    # embed 텍스트 추출 (archive/store.py와 동일 로직)
                    embed_text = None
                    if row.summary_json:
                        try:
                            sj = json.loads(row.summary_json) if isinstance(row.summary_json, str) else row.summary_json
                            embed_text = sj.get("turn_summary") if isinstance(sj, dict) else None
                        except Exception:
                            pass
                        if not embed_text:
                            embed_text = row.summary_json if isinstance(row.summary_json, str) else json.dumps(row.summary_json, ensure_ascii=False)

                    if not embed_text or not embed_text.strip():
                        skipped += 1
                        continue

                    vec = embedder._get_embedding(embed_text)
                    if vec is not None:
                        row.embedding = json.dumps(vec)
                        row.embedding_model = current_model
                        succeeded += 1
                        if row.turn_index is not None:
                            updated_turn_indices.append(int(row.turn_index))
                    else:
                        failed += 1
                        failed_ids.append(row.id)

                # batch commit
                try:
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error("reindex batch commit 실패: %s", exc)
                    failed += len(batch) - skipped
                    failed_ids.extend(r.id for r in batch)

            processed = succeeded + failed + skipped
            updated_turn_preview = updated_turn_indices[:20]
            updated_turn_range = None
            if updated_turn_indices:
                updated_turn_range = [min(updated_turn_indices), max(updated_turn_indices)]

            logger.info(
                "reindex_session 완료 (session=%s, model=%s, total=%d, processed=%d, ok=%d, fail=%d, skip=%d, ooc_skip=%d)",
                sid, current_model, total_in_session, processed, succeeded, failed, skipped, ooc_skipped,
            )
            return {
                "ok": True,
                "provenance_policy_version": _STEP13_EMBEDDING_PROVENANCE_POLICY_VERSION,
                "needs_reembed_policy_version": _STEP13_NEEDS_REEMBED_POLICY_VERSION,
                "selection_mode": selection_mode,
                "total_in_session": total_in_session,
                "candidates": len(candidates),
                "natural_candidates": len(natural_candidate_rows),
                "processed": processed,
                "succeeded": succeeded,
                "failed": failed,
                "skipped": skipped,
                "ooc_skipped": ooc_skipped,
                "failed_ids": failed_ids[:20],  # preview
                "model": current_model,
                "provenance_status_counts": provenance_status_counts,
                "candidate_reason_counts": candidate_reason_counts,
                "reembed_schedule": {
                    "policy_version": _STEP13_REEMBED_SCHEDULING_POLICY_VERSION,
                    "scope": "session",
                    "execution_mode": "manual_admin_batch_foreground",
                    "background_reembed": {
                        "current_mode": "deferred",
                        "status": "manual_trigger_required",
                        "next_owner": "future_background_queue",
                    },
                    "embedding_model_update": {
                        "current_project_embedding_model": current_model,
                        "write_rule": "set_embedding_and_embedding_model_on_success",
                        "updated_rows": succeeded,
                        "updated_turn_preview": updated_turn_preview,
                        "updated_turn_range": updated_turn_range,
                    },
                    "retrieval_index_refresh_handoff": {
                        "refresh_owner": "retrieval_index_registry",
                        "refresh_mode": "mark_dirty_then_next_prepare_turn_refresh",
                        "dirty_reason": "reembed_provenance_refresh",
                        "refresh_trigger": "next_prepare_turn",
                        "request_when_updated_rows_gt": 0,
                    },
                },
            }

        except Exception as exc:
            logger.error("reindex_session 실패 (session=%s): %s", sid, exc)
            return {"ok": False, "detail": f"DB 오류: {exc}"}
        finally:
            db.close()

    # ================================================================== #
    #  Sprint 3-D-2 — Rescan (누락 턴 재처리)
    # ================================================================== #

    def rescan_session(
        self,
        chat_session_id: str,
        max_items: int = 50,
        turn_indices: list[int] | None = None,
    ) -> dict:
        """세션의 chat_logs를 훑어 memory가 없는 턴을 Critic으로 재처리한다.

        Parameters
        ----------
        chat_session_id : str
            대상 세션 ID (필수).
        max_items : int
            한 번에 처리할 최대 누락 턴 수 (1~100).
        turn_indices : list[int] | None
            지정 시, 해당 turn_index 중 memory가 없는 턴만 재처리.

        Returns
        -------
        dict  with keys: ok, total_turns, existing_memory_turns, missing_turns,
              processed, succeeded, failed, failed_turn_indices
        """
        sid = (chat_session_id or "").strip()
        if not sid:
            return {"ok": False, "detail": "chat_session_id가 필요합니다."}

        max_items = max(1, min(100, max_items))
        normalized_turn_indices: list[int] = []
        seen_turn_indices: set[int] = set()
        for raw_turn in list(turn_indices or []):
            try:
                turn_value = int(raw_turn)
            except Exception:
                continue
            if turn_value < 0 or turn_value in seen_turn_indices:
                continue
            seen_turn_indices.add(turn_value)
            normalized_turn_indices.append(turn_value)
        normalized_turn_indices.sort()

        from backend.roles.critic import Critic
        critic = Critic()

        db: Session = SessionLocal()
        try:
            # 1) 세션의 모든 chat_log turn_index 집합
            log_turns = (
                db.query(ChatLog.turn_index)
                .filter(ChatLog.chat_session_id == sid)
                .distinct()
                .all()
            )
            all_turn_indices = sorted({r[0] for r in log_turns})

            # 2) 세션에 이미 memory가 있는 turn_index 집합
            mem_turns = (
                db.query(Memory.turn_index)
                .filter(Memory.chat_session_id == sid)
                .distinct()
                .all()
            )
            existing_turn_indices = {r[0] for r in mem_turns}

            # 3) 누락 턴 찾기
            missing_raw = [t for t in all_turn_indices if t not in existing_turn_indices]
            if normalized_turn_indices:
                requested_turn_set = set(normalized_turn_indices)
                missing_raw = [t for t in missing_raw if t in requested_turn_set]

            if not missing_raw:
                return {
                    "ok": True,
                    "total_turns": len(all_turn_indices),
                    "existing_memory_turns": len(existing_turn_indices),
                    "missing_turns": 0,
                    "ooc_skipped_turns": 0,
                    "requested_turn_count": len(normalized_turn_indices),
                    "requested_turn_preview": normalized_turn_indices[:20],
                    "processed": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "failed_turn_indices": [],
                }

            # 4) 전체 chat_logs를 한 번에 로드 (context 구성용)
            all_logs = (
                db.query(ChatLog)
                .filter(ChatLog.chat_session_id == sid)
                .order_by(ChatLog.turn_index, ChatLog.id)
                .all()
            )
            # turn_index → list of {role, content}
            logs_by_turn: dict[int, list[dict]] = {}
            for log in all_logs:
                logs_by_turn.setdefault(log.turn_index, []).append({
                    "role": log.role,
                    "content": log.content,
                })

            # OOC 사용자 턴은 rescan 대상에서 제외한다.
            ooc_turn_indices = {
                int(log.turn_index)
                for log in all_logs
                if (log.role or "").lower() == "user" and _is_ooc_text(log.content)
            }
            ooc_skipped_turns = sum(1 for t in missing_raw if t in ooc_turn_indices)
            missing = [t for t in missing_raw if t not in ooc_turn_indices]
            missing = missing[:max_items]

            if not missing:
                return {
                    "ok": True,
                    "total_turns": len(all_turn_indices),
                    "existing_memory_turns": len(existing_turn_indices),
                    "missing_turns": 0,
                    "ooc_skipped_turns": ooc_skipped_turns,
                    "requested_turn_count": len(normalized_turn_indices),
                    "requested_turn_preview": normalized_turn_indices[:20],
                    "processed": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "failed_turn_indices": [],
                }

        except Exception as exc:
            logger.error("rescan_session DB 조회 실패 (session=%s): %s", sid, exc)
            return {"ok": False, "detail": f"DB 오류: {exc}"}
        finally:
            db.close()

        # 5) 누락 턴에 대해 Critic 호출 (DB 세션 밖에서)
        succeeded = 0
        failed = 0
        failed_turn_indices: list[int] = []

        for turn_idx in missing:
            try:
                # turn_content: 해당 턴의 user+assistant 합침
                entries = logs_by_turn.get(turn_idx, [])
                if not entries:
                    failed += 1
                    failed_turn_indices.append(turn_idx)
                    continue

                turn_content = "\n".join(
                    f"[{e['role']}] {e['content']}" for e in entries if e.get("content")
                )
                if not turn_content.strip():
                    failed += 1
                    failed_turn_indices.append(turn_idx)
                    continue

                # context: 이전 턴들의 최근 메시지 (최대 6개)
                context_msgs: list[dict] = []
                for prev_t in all_turn_indices:
                    if prev_t >= turn_idx:
                        break
                    for e in logs_by_turn.get(prev_t, []):
                        if not e.get("content"):
                            continue
                        if (e.get("role") or "").lower() == "user" and _is_ooc_text(e.get("content")):
                            continue
                        context_msgs.append({"role": e["role"], "content": e["content"]})
                context_msgs = context_msgs[-6:]

                critic.run(turn_idx, turn_content, context_msgs, chat_session_id=sid)
                succeeded += 1

            except Exception as exc:
                logger.warning("rescan turn %d 실패: %s", turn_idx, exc)
                failed += 1
                failed_turn_indices.append(turn_idx)

        processed = succeeded + failed
        logger.info(
            "rescan_session 완료 (session=%s, total_turns=%d, missing=%d, ooc_skip=%d, requested=%d, ok=%d, fail=%d)",
            sid, len(all_turn_indices), len(missing), ooc_skipped_turns, len(normalized_turn_indices), succeeded, failed,
        )
        return {
            "ok": True,
            "total_turns": len(all_turn_indices),
            "existing_memory_turns": len(existing_turn_indices),
            "missing_turns": len(missing),
            "ooc_skipped_turns": ooc_skipped_turns,
            "requested_turn_count": len(normalized_turn_indices),
            "requested_turn_preview": normalized_turn_indices[:20],
            "processed": processed,
            "succeeded": succeeded,
            "failed": failed,
            "failed_turn_indices": failed_turn_indices[:20],
        }

    # ================================================================== #
    #  HS-1j — Session ID Migration (compat alias -> primary)
    # ================================================================== #

    def migrate_session_data(
        self,
        source_session_id: str,
        target_session_id: str,
        dry_run: bool = False,
        gate_status: str | None = None,
    ) -> dict:
        """source session의 row를 target session으로 안전 이관한다.

        - 대상: chat_session_id 컬럼을 가진 모든 테이블
        - 기본: UPDATE ... WHERE chat_session_id=source
        - 단일 unique(chat_session_id) 테이블: source/target 충돌 시 최신 행 우선 병합
        """
        source = (source_session_id or "").strip()
        target = (target_session_id or "").strip()

        if not source:
            return {"ok": False, "detail": "source_session_id가 필요합니다."}
        if not target:
            return {"ok": False, "detail": "target_session_id가 필요합니다."}
        if source == target:
            return {"ok": False, "detail": "source_session_id와 target_session_id가 동일합니다."}

        db: Session = SessionLocal()
        try:
            ingest_gate = _build_session_ingest_gate_report(
                db,
                source_session_id=source,
                target_session_id=target,
            )
            duplicate_stale_merge = _collect_direct_evidence_duplicate_merges(
                db,
                source_session_id=source,
                target_session_id=target,
            )
            unresolved_superseded_hashes = list(duplicate_stale_merge.get("unresolved_superseded_hashes") or [])
            if unresolved_superseded_hashes:
                direct_evidence_gate = ingest_gate.setdefault("direct_evidence", {})
                direct_evidence_gate["unresolved_superseded_merge_count"] = len(unresolved_superseded_hashes)
                direct_evidence_gate["unresolved_superseded_merge_preview"] = unresolved_superseded_hashes[:10]
                ingest_gate["review_required"] = True
                ingest_gate["recommended_gate_status"] = "review_required"
            normalized_gate_status = str(gate_status or "").strip().lower()

            if not dry_run and normalized_gate_status != "ready":
                return {
                    "ok": False,
                    "detail": "session migrate apply는 gate_status=ready 확인 후에만 진행할 수 있습니다.",
                    "ingest_gate": ingest_gate,
                }

            session_tables = [
                table
                for table in Base.metadata.sorted_tables
                if "chat_session_id" in table.c
            ]

            table_reports: list[dict] = []
            moved_rows = 0
            dropped_rows = 0
            tables_with_source_rows = 0

            for table in session_tables:
                source_rows = int(
                    db.query(func.count())
                    .select_from(table)
                    .filter(table.c.chat_session_id == source)
                    .scalar()
                    or 0
                )
                target_rows = int(
                    db.query(func.count())
                    .select_from(table)
                    .filter(table.c.chat_session_id == target)
                    .scalar()
                    or 0
                )

                if source_rows <= 0:
                    continue

                tables_with_source_rows += 1
                report = {
                    "table": table.name,
                    "source_rows": source_rows,
                    "target_rows": target_rows,
                    "moved_rows": 0,
                    "dropped_rows": 0,
                    "mode": "update",
                }

                if table.name == DirectEvidenceRecord.__tablename__:
                    report["dedupe_key"] = "source_hash"
                    report["dedupe_candidate_rows"] = int(
                        (ingest_gate.get("direct_evidence") or {}).get("duplicate_source_hash_count") or 0
                    )
                    report["stale_merge_policy_version"] = "sp1c.v1"
                    report["stale_duplicate_candidate_rows"] = int(
                        (ingest_gate.get("direct_evidence") or {}).get("stale_duplicate_source_hash_count") or 0
                    )

                if table.name == DirectEvidenceRecord.__tablename__ and unresolved_superseded_hashes:
                    report["unresolved_superseded_merge_rows"] = len(unresolved_superseded_hashes)

                has_single_unique_key = _table_has_single_session_unique_key(table)
                if has_single_unique_key and target_rows > 0:
                    report["mode"] = "unique_merge"

                if dry_run:
                    table_reports.append(report)
                    continue

                if table.name == DirectEvidenceRecord.__tablename__ and unresolved_superseded_hashes:
                    return {
                        "ok": False,
                        "detail": "unresolved superseded direct evidence merge가 남아 있어 session migrate apply를 진행할 수 없습니다.",
                        "ingest_gate": ingest_gate,
                        "table_reports": table_reports + [report],
                    }

                if has_single_unique_key and target_rows > 0:
                    source_row = (
                        db.execute(
                            table.select()
                            .where(table.c.chat_session_id == source)
                            .limit(1)
                        )
                        .mappings()
                        .first()
                    )
                    target_row = (
                        db.execute(
                            table.select()
                            .where(table.c.chat_session_id == target)
                            .limit(1)
                        )
                        .mappings()
                        .first()
                    )
                    winner = _pick_session_unique_row_winner(table.name, source_row, target_row)
                    report["winner"] = winner

                    if winner == "source":
                        dropped = db.execute(
                            table.delete().where(table.c.chat_session_id == target)
                        ).rowcount or 0
                        moved = db.execute(
                            table.update()
                            .where(table.c.chat_session_id == source)
                            .values(chat_session_id=target)
                        ).rowcount or 0
                        report["dropped_rows"] += int(dropped)
                        report["moved_rows"] += int(moved)
                    else:
                        dropped = db.execute(
                            table.delete().where(table.c.chat_session_id == source)
                        ).rowcount or 0
                        report["dropped_rows"] += int(dropped)
                else:
                    if table.name == DirectEvidenceRecord.__tablename__:
                        stale_merges_applied = 0
                        tombstone_propagated_rows = 0
                        superseded_propagated_rows = 0
                        for merge_entry in list(duplicate_stale_merge.get("merges") or []):
                            target_record_id = int(merge_entry.get("target_record_id") or 0)
                            if target_record_id <= 0:
                                continue

                            update_values = {}
                            if bool(merge_entry.get("source_tombstoned")):
                                update_values["tombstoned"] = True

                            resolved_target_superseded_by_id = merge_entry.get("resolved_target_superseded_by_id")
                            if resolved_target_superseded_by_id is not None:
                                update_values["superseded_by_id"] = int(resolved_target_superseded_by_id)

                            if not update_values:
                                continue

                            updated = db.execute(
                                table.update()
                                .where(table.c.id == target_record_id)
                                .where(table.c.chat_session_id == target)
                                .values(**update_values)
                            ).rowcount or 0
                            if updated:
                                stale_merges_applied += 1
                                if "tombstoned" in update_values:
                                    tombstone_propagated_rows += int(updated)
                                if "superseded_by_id" in update_values:
                                    superseded_propagated_rows += int(updated)

                        report["stale_merge_applied_rows"] = int(stale_merges_applied)
                        report["tombstone_propagated_rows"] = int(tombstone_propagated_rows)
                        report["superseded_propagated_rows"] = int(superseded_propagated_rows)

                        duplicate_hashes = list(
                            ((ingest_gate.get("direct_evidence") or {}).get("duplicate_source_hash_preview") or [])
                        )
                        if int((ingest_gate.get("direct_evidence") or {}).get("duplicate_source_hash_count") or 0) > len(duplicate_hashes):
                            duplicate_hashes = [
                                str(row[0]).strip()
                                for row in (
                                    db.query(DirectEvidenceRecord.source_hash)
                                    .filter(DirectEvidenceRecord.chat_session_id == target)
                                    .filter(DirectEvidenceRecord.source_hash.is_not(None))
                                    .all()
                                )
                                if str(row[0] or "").strip()
                            ]
                        if duplicate_hashes:
                            dedupe_dropped = db.execute(
                                table.delete()
                                .where(table.c.chat_session_id == source)
                                .where(table.c.source_hash.in_(duplicate_hashes))
                            ).rowcount or 0
                            report["dedupe_dropped_rows"] = int(dedupe_dropped)
                            report["dropped_rows"] += int(dedupe_dropped)

                    moved = db.execute(
                        table.update()
                        .where(table.c.chat_session_id == source)
                        .values(chat_session_id=target)
                    ).rowcount or 0
                    report["moved_rows"] = int(moved)

                moved_rows += int(report["moved_rows"])
                dropped_rows += int(report["dropped_rows"])
                table_reports.append(report)

            if dry_run:
                db.rollback()
            else:
                db.commit()

            # 적용 후 source 잔여 row 계산 (dry_run이면 현재 상태 그대로 집계)
            source_rows_remaining = 0
            for table in session_tables:
                source_rows_remaining += int(
                    db.query(func.count())
                    .select_from(table)
                    .filter(table.c.chat_session_id == source)
                    .scalar()
                    or 0
                )

            return {
                "ok": True,
                "dry_run": bool(dry_run),
                "source_session_id": source,
                "target_session_id": target,
                "tables_scanned": len(session_tables),
                "tables_with_source_rows": tables_with_source_rows,
                "moved_rows": moved_rows,
                "dropped_rows": dropped_rows,
                "source_rows_remaining": source_rows_remaining,
                "table_reports": table_reports,
                "ingest_gate": ingest_gate,
                "gate_status_used": normalized_gate_status or None,
            }

        except Exception as exc:
            db.rollback()
            logger.error("migrate_session_data 실패 (source=%s, target=%s): %s", source, target, exc)
            return {"ok": False, "detail": f"DB 오류: {exc}"}
        finally:
            db.close()


# ────────────────────────────────────────────
# Sprint 3-E-3: Audit Trail helper
# ────────────────────────────────────────────

def append_audit_log(
    *,
    event_type: str,
    chat_session_id: str | None = None,
    target_type: str | None = None,
    target_id: int | None = None,
    summary: str | None = None,
    details: dict | None = None,
    source: str = "api",
) -> None:
    """Best-effort audit log insertion. Never raises."""
    try:
        db: Session = SessionLocal()
        try:
            row = AuditLog(
                event_type=event_type,
                chat_session_id=chat_session_id,
                target_type=target_type,
                target_id=target_id,
                summary=summary,
                details_json=json.dumps(details, ensure_ascii=False) if details else None,
                source=source,
            )
            db.add(row)
            db.commit()
        finally:
            db.close()
    except Exception:
        logger.debug("audit log write failed", exc_info=True)


# ────────────────────────────────────────────
# Sprint 3-E-4: Critic Feedback helpers
# ────────────────────────────────────────────

VALID_FEEDBACK_VALUES = {"up", "down"}
VALID_FEEDBACK_TARGETS = {"memory": Memory, "kg_triple": KGTriple}


def save_critic_feedback(
    *,
    chat_session_id: str,
    target_type: str,
    target_id: int,
    feedback_value: str,
    feedback_note: str | None = None,
    source: str = "manual_ui",
) -> dict:
    """Critic 산출물에 대한 피드백을 저장한다.

    target row의 존재 여부 + session ownership을 검사한 뒤,
    append 형태로 critic_feedback 테이블에 기록한다.
    """
    if feedback_value not in VALID_FEEDBACK_VALUES:
        return {"ok": False, "detail": f"허용되지 않는 feedback_value: {feedback_value}"}

    model_cls = VALID_FEEDBACK_TARGETS.get(target_type)
    if model_cls is None:
        return {"ok": False, "detail": f"허용되지 않는 target_type: {target_type}"}

    db: Session = SessionLocal()
    try:
        target_row = db.query(model_cls).filter(model_cls.id == target_id).first()
        if target_row is None:
            return {"ok": False, "detail": f"{target_type} #{target_id} 을(를) 찾을 수 없습니다."}

        if target_row.chat_session_id != chat_session_id:
            return {"ok": False, "detail": "session 불일치: 다른 session의 row에는 피드백할 수 없습니다."}

        row = CriticFeedback(
            chat_session_id=chat_session_id,
            target_type=target_type,
            target_id=target_id,
            feedback_value=feedback_value,
            feedback_note=(feedback_note or "").strip() or None,
            source=source,
        )
        db.add(row)
        db.commit()
        db.refresh(row)

        return {
            "ok": True,
            "detail": f"{target_type} #{target_id} 에 {feedback_value} 피드백 저장됨",
            "feedback_id": row.id,
            "feedback_value": row.feedback_value,
        }
    finally:
        db.close()


def get_latest_feedback(
    *,
    chat_session_id: str,
    target_type: str,
    target_ids: list[int],
) -> dict[int, dict]:
    """주어진 target_ids에 대해 각각 최신 피드백 1건을 조회한다.

    Returns: {target_id: {"feedback_value": ..., "feedback_note": ..., "created_at": ...}}
    """
    if not target_ids:
        return {}
    db: Session = SessionLocal()
    try:
        from sqlalchemy import and_
        rows = (
            db.query(CriticFeedback)
            .filter(
                and_(
                    CriticFeedback.chat_session_id == chat_session_id,
                    CriticFeedback.target_type == target_type,
                    CriticFeedback.target_id.in_(target_ids),
                )
            )
            .order_by(CriticFeedback.id.desc())
            .all()
        )
        result: dict[int, dict] = {}
        for r in rows:
            if r.target_id not in result:  # 최신 1건만
                result[r.target_id] = {
                    "feedback_value": r.feedback_value,
                    "feedback_note": r.feedback_note,
                    "created_at": to_kst(r.created_at),
                }
        return result
    finally:
        db.close()


# ------------------------------------------------------------------ #
#  Phase 2-1: Active State CRUD
# ------------------------------------------------------------------ #

VALID_STATE_TYPES = {"scene_state", "relationship_state", "unresolved_threads"}
VALID_CANONICAL_LAYER_TYPES = set(VALID_STATE_TYPES)
HS1B_CANONICAL_UPSERT_POLICY_VERSION = "hs1b.v1"
HS1B_CANONICAL_LAYER_ORDER = ("scene_state", "relationship_state", "unresolved_threads")
HS1E_CANONICAL_PROVENANCE_POLICY_VERSION = "hs1e.v1"
HS1F_CANONICAL_VERIFIED_ONLY_PROMOTION_POLICY_VERSION = "hs1f.v1"
HS1G_RELATIONSHIP_STATE_POLICY_VERSION = "hs1g.v1"
HS1G_BRANCH_REGISTRY_KEYS = ("desire", "fear", "wound", "mask", "bond", "fixation")
HS1H_WORLD_CURRENT_STATE_POLICY_VERSION = "hs1h.v1"
HS1I_REL_WORLD_MINIMAL_SCHEMA_POLICY_VERSION = "hs1i.v1"
HS1J_RELATIONSHIP_V2_ROLLOUT_POLICY_VERSION = "hs1j.v1"
HS1J_REL_V2_STAGE1_MODE = "stage1_json_additive"
HS1J_REL_V2_STAGE2_GATE = "manual_contract_gate"
HS1J_ROLLOUT_HINT_KEYS = ("hot_query_required", "partial_update_required", "lineage_required")
HS1K_RELATIONSHIP_V2_BACKFILL_POLICY_VERSION = "hs1k.v1"
HS1K_RELATIONSHIP_V2_BACKFILL_MODE = "legacy_v1_to_minimal_v2"
HS1I_RELATIONSHIP_SEED_KEYS = (
    "_ea1h_conflict",
    "from",
    "to",
    "pair",
    "branch_registry",
    "change",
    "reason",
    "context_tag",
    "field_class",
    "confidence",
    "source_record",
    "source_turn",
    "last_verified_turn",
)
HS1I_WORLD_RULE_KEYS = ("scope", "scope_name", "category", "key", "value_preview", "source_turn")
HS1I_FACTION_STATUS_KEYS = ("scope_name", "key", "state", "value_preview", "source_turn")
HS1I_REGION_PRESSURE_KEYS = ("scope", "scope_name", "key", "pressure", "value_preview", "source_turn")
HS1I_OFFSCREEN_THREAD_KEYS = (
    "thread_type",
    "title",
    "owner",
    "target",
    "status",
    "last_seen_turn",
    "confidence",
    "details_preview",
)
HS1H_WORLD_RULES_LIMIT = 12
HS1H_FACTION_STATUS_LIMIT = 8
HS1H_REGION_PRESSURE_LIMIT = 8
HS1H_OFFSCREEN_THREAD_LIMIT = 8
HS1H_REGION_PRESSURE_HINT_TOKENS = (
    "pressure",
    "tension",
    "threat",
    "risk",
    "unrest",
    "instability",
    "압력",
    "긴장",
    "위협",
    "리스크",
    "불안",
)
HS1H_OFFSCREEN_HINT_TOKENS = (
    "offscreen",
    "off-screen",
    "background",
    "elsewhere",
    "원거리",
    "외부",
    "오프스크린",
    "부재",
    "다른 지역",
)
HS1H_OFFSCREEN_THREAD_TYPES = {
    "offscreen",
    "offscreen_thread",
}


def append_active_state(
    *,
    chat_session_id: str,
    state_type: str,
    content: str,
    turn_index: int,
) -> dict:
    """active_states 테이블에 새 row를 append한다.

    동일 session + state_type에 대해 항상 새 row를 추가하며,
    최신 상태는 latest query로 조회한다 (이력 보존).
    """
    sid = _safe_session_id(chat_session_id)
    if not sid:
        raise ValueError("chat_session_id is required")
    if state_type not in VALID_STATE_TYPES:
        raise ValueError(f"Invalid state_type: {state_type}. Must be one of {VALID_STATE_TYPES}")
    if not content or not content.strip():
        raise ValueError("content is required")

    db: Session = SessionLocal()
    try:
        row = ActiveState(
            chat_session_id=sid,
            state_type=state_type,
            content=content.strip(),
            turn_index=turn_index,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return {
            "id": row.id,
            "chat_session_id": row.chat_session_id,
            "state_type": row.state_type,
            "content": row.content,
            "turn_index": row.turn_index,
            "created_at": to_kst(row.created_at),
        }
    finally:
        db.close()


def get_latest_active_state(
    *,
    chat_session_id: str,
    state_type: str,
) -> dict | None:
    """특정 session + state_type의 최신 active state 1건을 반환한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return None

    db: Session = SessionLocal()
    try:
        row = (
            db.query(ActiveState)
            .filter(
                ActiveState.chat_session_id == sid,
                ActiveState.state_type == state_type,
            )
            .order_by(ActiveState.id.desc())
            .first()
        )
        if not row:
            return None
        return {
            "id": row.id,
            "chat_session_id": row.chat_session_id,
            "state_type": row.state_type,
            "content": row.content,
            "turn_index": row.turn_index,
            "created_at": to_kst(row.created_at),
        }
    finally:
        db.close()


def get_all_active_states_for_session(
    *,
    chat_session_id: str,
    state_type: str | None = None,
) -> list[dict]:
    """session의 최신 active state를 state_type별로 1건씩 반환한다.

    state_type이 지정되면 해당 타입만, 없으면 모든 타입의 최신 1건씩 반환.
    """
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    db: Session = SessionLocal()
    try:
        types_to_query = [state_type] if state_type else sorted(VALID_STATE_TYPES)
        results = []
        for st in types_to_query:
            row = (
                db.query(ActiveState)
                .filter(
                    ActiveState.chat_session_id == sid,
                    ActiveState.state_type == st,
                )
                .order_by(ActiveState.id.desc())
                .first()
            )
            if row:
                results.append({
                    "id": row.id,
                    "chat_session_id": row.chat_session_id,
                    "state_type": row.state_type,
                    "content": row.content,
                    "turn_index": row.turn_index,
                    "created_at": to_kst(row.created_at),
                })
        return results
    finally:
        db.close()


def append_canonical_state_layer(
    *,
    chat_session_id: str,
    layer_type: str,
    content: str,
    turn_index: int,
    source_state_type: str | None = None,
    source_turn: int | None = None,
    source_record: int | None = None,
    last_verified_turn: int | None = None,
    confidence: float | None = None,
) -> dict:
    """canonical_state_layers 테이블에 새 row를 append한다.

    HS-1a: active_states 계약을 유지한 채 병행 canonical layer를 별도 적재한다.
    """
    sid = _safe_session_id(chat_session_id)
    if not sid:
        raise ValueError("chat_session_id is required")
    if layer_type not in VALID_CANONICAL_LAYER_TYPES:
        raise ValueError(f"Invalid layer_type: {layer_type}. Must be one of {VALID_CANONICAL_LAYER_TYPES}")
    if not content or not content.strip():
        raise ValueError("content is required")

    source_type = source_state_type or layer_type
    if source_type not in VALID_STATE_TYPES:
        raise ValueError(f"Invalid source_state_type: {source_type}. Must be one of {VALID_STATE_TYPES}")

    db: Session = SessionLocal()
    try:
        normalized_confidence = None
        if confidence is not None:
            try:
                normalized_confidence = float(confidence)
            except Exception:
                normalized_confidence = None
            if normalized_confidence is not None:
                if normalized_confidence < 0.0:
                    normalized_confidence = 0.0
                if normalized_confidence > 1.0:
                    normalized_confidence = 1.0

        row = CanonicalStateLayer(
            chat_session_id=sid,
            layer_type=layer_type,
            content=content.strip(),
            source_state_type=source_type,
            turn_index=turn_index,
            source_turn=source_turn,
            source_record=source_record,
            last_verified_turn=last_verified_turn,
            confidence=normalized_confidence,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return {
            "id": row.id,
            "chat_session_id": row.chat_session_id,
            "layer_type": row.layer_type,
            "content": row.content,
            "source_state_type": row.source_state_type,
            "turn_index": row.turn_index,
            "source_turn": row.source_turn,
            "source_record": row.source_record,
            "last_verified_turn": row.last_verified_turn,
            "confidence": row.confidence,
            "created_at": to_kst(row.created_at),
        }
    finally:
        db.close()


def get_all_canonical_state_layers_for_session(
    *,
    chat_session_id: str,
    layer_type: str | None = None,
) -> list[dict]:
    """session의 최신 canonical state layer를 layer_type별로 1건씩 반환한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    db: Session = SessionLocal()
    try:
        if layer_type and layer_type not in VALID_CANONICAL_LAYER_TYPES:
            return []

        types_to_query = [layer_type] if layer_type else sorted(VALID_CANONICAL_LAYER_TYPES)
        results = []
        for lt in types_to_query:
            row = (
                db.query(CanonicalStateLayer)
                .filter(
                    CanonicalStateLayer.chat_session_id == sid,
                    CanonicalStateLayer.layer_type == lt,
                )
                .order_by(CanonicalStateLayer.id.desc())
                .first()
            )
            if row:
                results.append({
                    "id": row.id,
                    "chat_session_id": row.chat_session_id,
                    "layer_type": row.layer_type,
                    "content": row.content,
                    "source_state_type": row.source_state_type,
                    "turn_index": row.turn_index,
                    "source_turn": row.source_turn,
                    "source_record": row.source_record,
                    "last_verified_turn": row.last_verified_turn,
                    "confidence": row.confidence,
                    "created_at": to_kst(row.created_at),
                })
        return results
    finally:
        db.close()


def _get_direct_evidence_gate_snapshot_for_turn(
    *,
    chat_session_id: str,
    turn_index: int,
) -> dict:
    """현재 turn의 direct evidence gate 상태를 요약한다.

    canonical state 승격 경로에서 보수적 guard로 사용한다.
    - approved_count: gate 승인(`verified_direct` + `verified` + non-repair + gate in {finalize,manual})
    - blocked_count: repair/rejected/tombstoned/recovery 신호
    """
    sid = _safe_session_id(chat_session_id)
    snapshot = {
        "approved_count": 0,
        "blocked_count": 0,
    }
    if not sid:
        return snapshot

    db: Session = SessionLocal()
    try:
        evidence_rows = (
            db.query(DirectEvidenceRecord)
            .filter(
                DirectEvidenceRecord.chat_session_id == sid,
                DirectEvidenceRecord.source_turn_start <= turn_index,
                DirectEvidenceRecord.source_turn_end >= turn_index,
            )
            .order_by(DirectEvidenceRecord.id.asc())
            .all()
        )

        for evidence_row in evidence_rows:
            if is_gate_approved_current_truth(
                archive_state=evidence_row.archive_state,
                capture_verification=evidence_row.capture_verification,
                repair_needed=evidence_row.repair_needed,
                committed_gate=evidence_row.committed_gate,
                tombstoned=evidence_row.tombstoned,
                superseded_by_id=evidence_row.superseded_by_id,
            ):
                snapshot["approved_count"] += 1
                continue

            if is_gate_blocked_signal(
                archive_state=evidence_row.archive_state,
                capture_verification=evidence_row.capture_verification,
                repair_needed=evidence_row.repair_needed,
                committed_gate=evidence_row.committed_gate,
                tombstoned=evidence_row.tombstoned,
                superseded_by_id=evidence_row.superseded_by_id,
            ):
                snapshot["blocked_count"] += 1

        return snapshot
    finally:
        db.close()


def _collect_gate_approved_direct_evidence_seed_for_turn(
    *,
    chat_session_id: str,
    turn_index: int,
    max_items: int = 5,
    max_total_chars: int = 1200,
) -> list[dict]:
    """현재 turn 범위의 gate-approved direct evidence seed를 수집한다.

    EA-1e에서 canonical state 승격 입력원을 direct evidence 우선으로 재정렬할 때
    scene_state에 deterministic seed를 주입하는 용도로 사용한다.
    """
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    db: Session = SessionLocal()
    try:
        evidence_rows = (
            db.query(DirectEvidenceRecord)
            .filter(
                DirectEvidenceRecord.chat_session_id == sid,
                DirectEvidenceRecord.source_turn_start <= turn_index,
                DirectEvidenceRecord.source_turn_end >= turn_index,
            )
            .order_by(DirectEvidenceRecord.id.asc())
            .all()
        )

        seed_items: list[dict] = []
        used_chars = 0
        for evidence_row in evidence_rows:
            if not is_gate_approved_current_truth(
                archive_state=evidence_row.archive_state,
                capture_verification=evidence_row.capture_verification,
                repair_needed=evidence_row.repair_needed,
                committed_gate=evidence_row.committed_gate,
                tombstoned=evidence_row.tombstoned,
                superseded_by_id=evidence_row.superseded_by_id,
            ):
                continue

            if not _is_direct_evidence_retained_for_consumption(
                evidence_row,
                latest_turn_index=turn_index,
            ):
                continue

            evidence_text = " ".join(str(evidence_row.evidence_text or "").split()).strip()
            if not evidence_text:
                continue
            if len(seed_items) >= max_items:
                break
            if used_chars + len(evidence_text) > max_total_chars:
                break

            normalized_committed_gate = resolve_committed_gate(
                evidence_row.capture_verification,
                evidence_row.repair_needed,
                evidence_row.committed_gate,
            )
            seed_items.append({
                "record_id": int(evidence_row.id),
                "source_turn_start": evidence_row.source_turn_start,
                "source_turn_end": evidence_row.source_turn_end,
                "committed_gate": normalized_committed_gate,
                "evidence_text": evidence_text,
            })
            used_chars += len(evidence_text)

        return seed_items
    finally:
        db.close()


def _decorate_scene_state_with_direct_evidence_seed(
    *,
    scene_state: dict | None,
    evidence_seed: list[dict],
    conflict_resolution: dict | None = None,
) -> dict:
    """scene_state에 direct-evidence-first 승격 메타를 주입한다."""
    base_scene: dict = dict(scene_state) if isinstance(scene_state, dict) else {}
    if evidence_seed:
        record_ids = [int(item.get("record_id") or 0) for item in evidence_seed if int(item.get("record_id") or 0) > 0]
        base_scene["_promotion_input"] = {
            "source": "direct_evidence_first",
            "approved_evidence_count": len(evidence_seed),
            "approved_evidence_record_ids": record_ids,
        }
        base_scene["_direct_evidence_seed"] = evidence_seed

    if isinstance(conflict_resolution, dict) and conflict_resolution:
        base_scene["_conflict_resolution"] = conflict_resolution

    return base_scene


# EA-1h: deterministic canonical conflict resolution state machine
_EA1H_POLICY_VERSION = "ea1h.v1"
_EA1H_LOW_CONFIDENCE_THRESHOLD = 0.58
_EA1H_HIGH_IMPACT_FIELD_CLASSES = (
    "identity",
    "irreversible_world_rule",
    "death",
    "ownership",
)
_EA1H_LOW_CONFIDENCE_HINT_TOKENS = (
    "maybe",
    "uncertain",
    "rumor",
    "possible",
    "possibly",
    "seems",
    "seem",
    "probably",
    "추정",
    "불확실",
    "아마",
    "소문",
)
_EA1H_PARALLEL_CONTEXT_HINT_TOKENS = (
    "meanwhile",
    "parallel",
    "separate context",
    "different context",
    "alternate",
    "elsewhere",
    "동시에",
    "별개 맥락",
    "다른 맥락",
    "다른 장소",
)
_EA1H_HIGH_IMPACT_HINT_TOKENS = (
    "death",
    "dead",
    "killed",
    "identity",
    "real identity",
    "ownership",
    "owner",
    "world rule",
    "cannot",
    "irreversible",
    "사망",
    "죽음",
    "정체",
    "소유",
    "세계 규칙",
)

# EA-1i: conflict confidence/high-impact routing policy
_EA1I_POLICY_VERSION = "ea1i.v1"
_EA1I_AUTO_PROMOTE_MIN_CONFIDENCE = 0.72
_EA1I_HOLD_MIN_CONFIDENCE = 0.58
_EA1I_REPAIR_QUEUE_MAX_CONFIDENCE = 0.42
_EA1I_USER_CONFIRMATION_MIN_CONFIDENCE = 0.80
_EA1I_USER_CONFIRMATION_FIELD_CLASSES = _EA1H_HIGH_IMPACT_FIELD_CLASSES


def _ea1h_normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _ea1h_has_low_confidence_hint(*, incoming: dict) -> bool:
    merged_text = _ea1h_normalize_text(f"{incoming.get('change') or ''} {incoming.get('reason') or ''}")
    return any(token in merged_text for token in _EA1H_LOW_CONFIDENCE_HINT_TOKENS)


def _ea1h_pair_key(from_name: str | None, to_name: str | None) -> str | None:
    left = _ea1h_normalize_text(from_name)
    right = _ea1h_normalize_text(to_name)
    if not left and not right:
        return None
    if not left or not right:
        return left or right
    if left <= right:
        return f"{left}::{right}"
    return f"{right}::{left}"


def _ea1h_parse_pair_text(value: object) -> tuple[str | None, str | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    for separator in ("->", "→", "=>", "/", "|", ",", "&", "-"):
        if separator not in text:
            continue
        parts = [part.strip() for part in text.split(separator) if str(part or "").strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
    return None, None


def _ea1h_coerce_confidence(value: object, *, default: float = 0.72) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except Exception:
        return default
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _ea1h_normalize_relationship_change_item(raw_item: object) -> dict:
    if not isinstance(raw_item, dict):
        text = str(raw_item or "").strip()
        return {
            "raw_item": raw_item,
            "from": "",
            "to": "",
            "pair_key": None,
            "change": text,
            "reason": "",
            "context_tag": "",
            "field_class": "relationship",
            "confidence": _ea1h_coerce_confidence(None),
        }

    identity = raw_item.get("identity") if isinstance(raw_item.get("identity"), dict) else {}
    core_state = raw_item.get("core_state") if isinstance(raw_item.get("core_state"), dict) else {}
    context_layer = raw_item.get("context") if isinstance(raw_item.get("context"), dict) else {}
    verification = raw_item.get("verification") if isinstance(raw_item.get("verification"), dict) else {}

    from_name = str(
        raw_item.get("from")
        or raw_item.get("source")
        or identity.get("left_entity")
        or identity.get("source")
        or ""
    ).strip()
    to_name = str(
        raw_item.get("to")
        or raw_item.get("target")
        or identity.get("right_entity")
        or identity.get("target")
        or ""
    ).strip()
    if not from_name or not to_name:
        pair_from, pair_to = _ea1h_parse_pair_text(raw_item.get("pair") or identity.get("pair"))
        if not from_name and pair_from:
            from_name = pair_from
        if not to_name and pair_to:
            to_name = pair_to

    change = str(
        raw_item.get("change")
        or raw_item.get("delta")
        or raw_item.get("state")
        or raw_item.get("relation")
        or raw_item.get("status")
        or core_state.get("current_fact")
        or core_state.get("change")
        or core_state.get("status")
        or ""
    ).strip()
    reason = str(raw_item.get("reason") or raw_item.get("evidence") or core_state.get("reason") or "").strip()
    context_tag = str(
        raw_item.get("context_tag")
        or raw_item.get("scope")
        or raw_item.get("context")
        or context_layer.get("scene_context_tag")
        or context_layer.get("tag")
        or ""
    ).strip()
    field_class = _ea1h_normalize_text(
        raw_item.get("field_class")
        or raw_item.get("field")
        or core_state.get("field_class")
        or "relationship"
    )

    confidence_value = raw_item.get("confidence")
    if confidence_value is None:
        confidence_value = core_state.get("confidence")
    if confidence_value is None:
        confidence_value = verification.get("confidence")

    return {
        "raw_item": raw_item,
        "from": from_name,
        "to": to_name,
        "pair_key": _ea1h_pair_key(from_name, to_name),
        "change": change,
        "reason": reason,
        "context_tag": context_tag,
        "field_class": field_class,
        "confidence": _ea1h_coerce_confidence(confidence_value),
    }


def _ea1h_load_current_relationship_snapshot(*, chat_session_id: str) -> dict:
    latest = get_latest_active_state(chat_session_id=chat_session_id, state_type="relationship_state")
    if not latest:
        return {
            "state_id": None,
            "turn_index": None,
            "index": {},
        }

    content = latest.get("content")
    parsed = None
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
    except Exception:
        parsed = None
    if not isinstance(parsed, list):
        parsed = []

    index: dict[str, dict] = {}
    latest_turn_index = int(latest.get("turn_index") or 0)
    backfilled_items, backfill_summary = _hs1k_backfill_relationship_list_minimal_v2(
        raw_items=parsed,
        turn_index=latest_turn_index,
    )

    for item in backfilled_items:
        if not isinstance(item, dict):
            continue
        normalized = _ea1h_normalize_relationship_change_item(item)
        pair_key = normalized.get("pair_key")
        if not pair_key:
            continue
        index[pair_key] = {
            "state_id": latest.get("id"),
            "turn_index": latest_turn_index,
            "from": normalized.get("from") or "",
            "to": normalized.get("to") or "",
            "change": normalized.get("change") or "",
            "reason": normalized.get("reason") or "",
            "context_tag": normalized.get("context_tag") or "",
            "field_class": normalized.get("field_class") or "relationship",
        }

    return {
        "state_id": latest.get("id"),
        "turn_index": latest_turn_index,
        "index": index,
        "backfill": backfill_summary,
    }


def _ea1h_is_low_confidence_noise(*, incoming: dict) -> bool:
    confidence = float(incoming.get("confidence") or 0.0)
    if confidence < _EA1H_LOW_CONFIDENCE_THRESHOLD:
        return True

    return _ea1h_has_low_confidence_hint(incoming=incoming)


def _ea1h_is_parallel_context(*, incoming: dict, current: dict) -> bool:
    incoming_tag = _ea1h_normalize_text(incoming.get("context_tag"))
    current_tag = _ea1h_normalize_text(current.get("context_tag"))
    if incoming_tag and current_tag and incoming_tag != current_tag:
        return True

    reason_text = _ea1h_normalize_text(incoming.get("reason"))
    return any(token in reason_text for token in _EA1H_PARALLEL_CONTEXT_HINT_TOKENS)


def _ea1h_is_high_impact_conflict(*, incoming: dict, current: dict) -> bool:
    field_class = _ea1h_normalize_text(incoming.get("field_class") or current.get("field_class") or "")
    if field_class in _EA1H_HIGH_IMPACT_FIELD_CLASSES:
        return True

    merged_text = _ea1h_normalize_text(
        f"{incoming.get('change') or ''} {incoming.get('reason') or ''} {current.get('change') or ''}"
    )
    return any(token in merged_text for token in _EA1H_HIGH_IMPACT_HINT_TOKENS)


def _ea1h_build_manual_review_thread(*, incoming: dict, current: dict) -> str:
    left = str(incoming.get("from") or current.get("from") or "?").strip() or "?"
    right = str(incoming.get("to") or current.get("to") or "?").strip() or "?"
    previous_change = str(current.get("change") or "?").strip() or "?"
    incoming_change = str(incoming.get("change") or "?").strip() or "?"
    return f"[manual_review][ea1h] {left} ↔ {right}: {previous_change} -> {incoming_change}"


def _ea1i_build_user_confirmation_thread(*, incoming: dict, current: dict) -> str:
    left = str(incoming.get("from") or current.get("from") or "?").strip() or "?"
    right = str(incoming.get("to") or current.get("to") or "?").strip() or "?"
    field_class = _ea1h_normalize_text(incoming.get("field_class") or current.get("field_class") or "relationship")
    previous_change = str(current.get("change") or "?").strip() or "?"
    incoming_change = str(incoming.get("change") or "?").strip() or "?"
    return f"[user_confirmation][ea1i] {left} ↔ {right} ({field_class}): {previous_change} -> {incoming_change}"


def _ea1i_resolve_route(
    *,
    decision: str,
    conflict_class: str,
    confidence: float,
    field_class: str,
    low_confidence_hint: bool,
) -> str:
    normalized_field_class = _ea1h_normalize_text(field_class)

    if decision == "superseded":
        if confidence >= _EA1I_AUTO_PROMOTE_MIN_CONFIDENCE:
            return "auto_promote"
        return "hold_candidate"

    if decision == "manual_review":
        if (
            normalized_field_class in _EA1I_USER_CONFIRMATION_FIELD_CLASSES
            and confidence >= _EA1I_USER_CONFIRMATION_MIN_CONFIDENCE
        ):
            return "user_confirmation_candidate"
        return "hold_candidate"

    if decision == "hold":
        return "hold_candidate"

    if decision == "tombstone":
        if conflict_class == "low_confidence_noise":
            if confidence <= _EA1I_REPAIR_QUEUE_MAX_CONFIDENCE or low_confidence_hint:
                return "repair_queue_candidate"
            return "hold_candidate"
        return "repair_queue_candidate"

    return "hold_candidate"


def _ea1h_merge_manual_review_threads(
    *,
    threads: dict | None,
    manual_review_threads: list[str],
    user_confirmation_threads: list[str] | None = None,
) -> dict | None:
    additional_threads = []
    if isinstance(manual_review_threads, list):
        additional_threads.extend(manual_review_threads)
    if isinstance(user_confirmation_threads, list):
        additional_threads.extend(user_confirmation_threads)

    if not additional_threads:
        return threads

    base_threads = dict(threads) if isinstance(threads, dict) else {"opened": [], "resolved": []}
    opened = base_threads.get("opened")
    resolved = base_threads.get("resolved")
    if not isinstance(opened, list):
        opened = []
    if not isinstance(resolved, list):
        resolved = []

    seen = {str(item).strip() for item in opened if isinstance(item, str) and str(item).strip()}
    for line in additional_threads:
        normalized = str(line or "").strip()
        if not normalized or normalized in seen:
            continue
        opened.append(normalized)
        seen.add(normalized)

    base_threads["opened"] = opened
    base_threads["resolved"] = resolved
    return base_threads


def _ea1h_resolve_canonical_conflicts(
    *,
    chat_session_id: str,
    turn_index: int,
    relationship_changes: list | None,
    direct_evidence_seed: list[dict],
) -> dict:
    if not isinstance(relationship_changes, list):
        return {
            "promoted_relationship_changes": [],
            "manual_review_threads": [],
            "user_confirmation_threads": [],
            "summary": {
                "policy_version": _EA1H_POLICY_VERSION,
                "confidence_policy_version": _EA1I_POLICY_VERSION,
                "enabled": False,
                "low_confidence_threshold": _EA1H_LOW_CONFIDENCE_THRESHOLD,
                "confidence_thresholds": {
                    "auto_promote_min": _EA1I_AUTO_PROMOTE_MIN_CONFIDENCE,
                    "hold_min": _EA1I_HOLD_MIN_CONFIDENCE,
                    "repair_queue_max": _EA1I_REPAIR_QUEUE_MAX_CONFIDENCE,
                    "user_confirmation_min": _EA1I_USER_CONFIRMATION_MIN_CONFIDENCE,
                },
                "high_impact_field_classes": list(_EA1H_HIGH_IMPACT_FIELD_CLASSES),
                "high_impact_user_confirmation_fields": list(_EA1I_USER_CONFIRMATION_FIELD_CLASSES),
                "incoming_verified_evidence_count": len(direct_evidence_seed or []),
                "decision_counts": {
                    "promote": 0,
                    "superseded": 0,
                    "tombstone": 0,
                    "hold": 0,
                    "manual_review": 0,
                },
                "route_counts": {
                    "auto_promote": 0,
                    "repair_queue_candidate": 0,
                    "hold_candidate": 0,
                    "user_confirmation_candidate": 0,
                },
                "repair_queue_candidate_record_ids": [],
                "conflicts": [],
            },
        }

    current_snapshot = _ea1h_load_current_relationship_snapshot(chat_session_id=chat_session_id)
    current_index = current_snapshot.get("index") or {}

    latest_evidence_record_id = None
    for seed_item in direct_evidence_seed or []:
        rid = int(seed_item.get("record_id") or 0)
        if rid <= 0:
            continue
        if latest_evidence_record_id is None or rid > latest_evidence_record_id:
            latest_evidence_record_id = rid

    decision_counts = {
        "promote": 0,
        "superseded": 0,
        "tombstone": 0,
        "hold": 0,
        "manual_review": 0,
    }
    conflict_trace: list[dict] = []
    promoted_relationship_changes: list = []
    manual_review_threads: list[str] = []
    user_confirmation_threads: list[str] = []
    route_counts = {
        "auto_promote": 0,
        "repair_queue_candidate": 0,
        "hold_candidate": 0,
        "user_confirmation_candidate": 0,
    }
    repair_queue_candidate_record_ids: list[int] = []

    for index, raw_item in enumerate(relationship_changes):
        incoming = _ea1h_normalize_relationship_change_item(raw_item)
        pair_key = incoming.get("pair_key")
        current = current_index.get(pair_key) if pair_key else None

        incoming_change = _ea1h_normalize_text(incoming.get("change"))
        current_change = _ea1h_normalize_text((current or {}).get("change"))

        # current canonical fact가 없거나 실질 변화가 없으면 그대로 승격한다.
        if not current or not incoming_change or incoming_change == current_change:
            decision_counts["promote"] += 1
            if isinstance(raw_item, dict):
                promoted = dict(raw_item)
                if incoming.get("from") and not promoted.get("from"):
                    promoted["from"] = incoming.get("from")
                if incoming.get("to") and not promoted.get("to"):
                    promoted["to"] = incoming.get("to")
                if incoming.get("change") and not promoted.get("change"):
                    promoted["change"] = incoming.get("change")
                promoted_relationship_changes.append(promoted)
            else:
                promoted_relationship_changes.append(raw_item)
            continue

        low_confidence_noise = _ea1h_is_low_confidence_noise(incoming=incoming)
        high_impact_conflict = _ea1h_is_high_impact_conflict(incoming=incoming, current=current)
        parallel_context = _ea1h_is_parallel_context(incoming=incoming, current=current)
        low_confidence_hint = _ea1h_has_low_confidence_hint(incoming=incoming)
        confidence = float(incoming.get("confidence") or 0.0)
        lineage_clear = latest_evidence_record_id is not None
        recency_ok = turn_index >= int(current.get("turn_index") or 0)

        if low_confidence_noise:
            conflict_class = "low_confidence_noise"
            decision = "tombstone"
        elif high_impact_conflict:
            conflict_class = "hard_contradiction"
            decision = "manual_review"
        elif parallel_context:
            conflict_class = "parallel_context"
            decision = "hold"
        elif lineage_clear and recency_ok:
            conflict_class = "state_transition"
            if confidence >= _EA1I_AUTO_PROMOTE_MIN_CONFIDENCE:
                decision = "superseded"
            elif confidence >= _EA1I_HOLD_MIN_CONFIDENCE:
                decision = "hold"
            else:
                decision = "tombstone"
        else:
            conflict_class = "parallel_context"
            decision = "hold"

        route = _ea1i_resolve_route(
            decision=decision,
            conflict_class=conflict_class,
            confidence=confidence,
            field_class=incoming.get("field_class") or current.get("field_class") or "relationship",
            low_confidence_hint=low_confidence_hint,
        )
        route_counts[route] += 1

        decision_counts[decision] += 1
        conflict_item = {
            "index": index,
            "pair_key": pair_key,
            "from": incoming.get("from") or current.get("from") or "",
            "to": incoming.get("to") or current.get("to") or "",
            "current_change": current.get("change") or "",
            "incoming_change": incoming.get("change") or "",
            "confidence": round(confidence, 4),
            "field_class": incoming.get("field_class") or "relationship",
            "conflict_class": conflict_class,
            "decision": decision,
            "route": route,
            "lineage_clear": bool(lineage_clear),
            "recency_ok": bool(recency_ok),
            "incoming_evidence_record_id": latest_evidence_record_id,
            "current_state_id": current.get("state_id"),
        }
        conflict_trace.append(conflict_item)

        if decision == "superseded":
            promoted_item = dict(raw_item) if isinstance(raw_item, dict) else {"change": incoming.get("change") or ""}
            if incoming.get("from") and not promoted_item.get("from"):
                promoted_item["from"] = incoming.get("from")
            if incoming.get("to") and not promoted_item.get("to"):
                promoted_item["to"] = incoming.get("to")
            if incoming.get("change") and not promoted_item.get("change"):
                promoted_item["change"] = incoming.get("change")
            if incoming.get("reason") and not promoted_item.get("reason"):
                promoted_item["reason"] = incoming.get("reason")
            promoted_item["_ea1h_conflict"] = {
                "policy_version": _EA1H_POLICY_VERSION,
                "conflict_class": conflict_class,
                "decision": decision,
                "superseded_state_id": current.get("state_id"),
                "resolved_at_turn": turn_index,
                "incoming_evidence_record_id": latest_evidence_record_id,
            }
            promoted_relationship_changes.append(promoted_item)
        elif decision == "manual_review":
            manual_review_threads.append(
                _ea1h_build_manual_review_thread(incoming=incoming, current=current)
            )

        if route == "user_confirmation_candidate":
            user_confirmation_threads.append(
                _ea1i_build_user_confirmation_thread(incoming=incoming, current=current)
            )

        if route == "repair_queue_candidate" and latest_evidence_record_id is not None:
            repair_queue_candidate_record_ids.append(int(latest_evidence_record_id))

    summary = {
        "policy_version": _EA1H_POLICY_VERSION,
        "confidence_policy_version": _EA1I_POLICY_VERSION,
        "enabled": bool(relationship_changes),
        "low_confidence_threshold": _EA1H_LOW_CONFIDENCE_THRESHOLD,
        "confidence_thresholds": {
            "auto_promote_min": _EA1I_AUTO_PROMOTE_MIN_CONFIDENCE,
            "hold_min": _EA1I_HOLD_MIN_CONFIDENCE,
            "repair_queue_max": _EA1I_REPAIR_QUEUE_MAX_CONFIDENCE,
            "user_confirmation_min": _EA1I_USER_CONFIRMATION_MIN_CONFIDENCE,
        },
        "high_impact_field_classes": list(_EA1H_HIGH_IMPACT_FIELD_CLASSES),
        "high_impact_user_confirmation_fields": list(_EA1I_USER_CONFIRMATION_FIELD_CLASSES),
        "incoming_verified_evidence_count": len(direct_evidence_seed or []),
        "incoming_verified_latest_record_id": latest_evidence_record_id,
        "current_relationship_state_id": current_snapshot.get("state_id"),
        "current_snapshot_backfill": current_snapshot.get("backfill") or {
            "policy_version": HS1K_RELATIONSHIP_V2_BACKFILL_POLICY_VERSION,
            "mode": HS1K_RELATIONSHIP_V2_BACKFILL_MODE,
            "item_count": 0,
            "legacy_v1_item_count": 0,
            "converted_item_count": 0,
            "provenance_defaults_applied_count": 0,
        },
        "decision_counts": decision_counts,
        "route_counts": route_counts,
        "repair_queue_candidate_record_ids": sorted(set(repair_queue_candidate_record_ids)),
        "conflicts": conflict_trace[:20],
    }

    return {
        "promoted_relationship_changes": promoted_relationship_changes,
        "manual_review_threads": manual_review_threads,
        "user_confirmation_threads": user_confirmation_threads,
        "summary": summary,
    }


def _hs1e_parse_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _hs1e_parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed


def _hs1e_build_candidate_provenance(
    *,
    state_type: str,
    turn_index: int,
    content: str,
    direct_evidence_seed: list[dict],
) -> dict:
    source_record = None
    source_turn = None
    last_verified_turn = None

    for seed_item in direct_evidence_seed or []:
        record_id = _hs1e_parse_int(seed_item.get("record_id"))
        if record_id is None or record_id <= 0:
            continue
        if source_record is not None and record_id <= source_record:
            continue
        source_record = record_id
        source_turn = _hs1e_parse_int(seed_item.get("source_turn_start"))
        last_verified_turn = _hs1e_parse_int(seed_item.get("source_turn_end"))

    parsed_content = None
    try:
        parsed_content = json.loads(content) if isinstance(content, str) else content
    except Exception:
        parsed_content = None

    confidence_candidates: list[float] = []
    explicit_confidence_signal = False
    low_confidence_hint = False

    if state_type == "relationship_state" and isinstance(parsed_content, list):
        for item in parsed_content:
            if isinstance(item, dict) and "confidence" in item:
                explicit_confidence = _hs1e_parse_float(item.get("confidence"))
                if explicit_confidence is not None:
                    explicit_confidence_signal = True
            normalized = _ea1h_normalize_relationship_change_item(item)
            confidence = _hs1e_parse_float(normalized.get("confidence"))
            if confidence is not None:
                confidence_candidates.append(confidence)
            if _ea1h_has_low_confidence_hint(incoming=normalized):
                low_confidence_hint = True
    elif state_type == "scene_state" and isinstance(parsed_content, dict):
        conflict_summary = parsed_content.get("_conflict_resolution")
        if isinstance(conflict_summary, dict):
            conflicts = conflict_summary.get("conflicts")
            if isinstance(conflicts, list):
                for conflict_item in conflicts:
                    if not isinstance(conflict_item, dict):
                        continue
                    confidence = _hs1e_parse_float(conflict_item.get("confidence"))
                    if confidence is not None:
                        explicit_confidence_signal = True
                        confidence_candidates.append(confidence)
                    if str(conflict_item.get("conflict_class") or "").strip() == "low_confidence_noise":
                        low_confidence_hint = True

    if not low_confidence_hint:
        content_for_hint = content
        if state_type == "scene_state" and isinstance(parsed_content, dict):
            hint_payload = dict(parsed_content)
            hint_payload.pop("world_current_state", None)
            hint_payload.pop("world_current_state_policy_version", None)
            content_for_hint = json.dumps(hint_payload, ensure_ascii=False)

        content_text = _ea1h_normalize_text(content_for_hint)
        if content_text:
            low_confidence_hint = any(token in content_text for token in _EA1H_LOW_CONFIDENCE_HINT_TOKENS)

    confidence = None
    if confidence_candidates:
        confidence = round(max(confidence_candidates), 4)
    elif source_record is not None:
        confidence = round(_EA1I_AUTO_PROMOTE_MIN_CONFIDENCE, 4)

    has_confidence_signal = bool(explicit_confidence_signal)
    is_estimate_candidate = bool(has_confidence_signal or low_confidence_hint)

    return {
        "policy_version": HS1E_CANONICAL_PROVENANCE_POLICY_VERSION,
        "state_type": state_type,
        "source_turn": source_turn if source_turn is not None else turn_index,
        "source_record": source_record,
        "last_verified_turn": last_verified_turn,
        "confidence": confidence,
        "has_verified_source": bool(source_record is not None and last_verified_turn is not None),
        "has_confidence_signal": has_confidence_signal,
        "low_confidence_hint": bool(low_confidence_hint),
        "is_estimate_candidate": is_estimate_candidate,
    }


def _hs1f_should_block_unverified_estimate_promotion(*, provenance: dict) -> bool:
    if bool(provenance.get("has_verified_source")):
        return False

    is_estimate_candidate = provenance.get("is_estimate_candidate")
    if is_estimate_candidate is None:
        confidence = _hs1e_parse_float(provenance.get("confidence"))
        is_estimate_candidate = bool(provenance.get("low_confidence_hint")) or confidence is not None

    return bool(is_estimate_candidate)


def _hs1i_seed_relationship_payload(raw_payload: dict) -> dict:
    seeded: dict = {}
    for key in HS1I_RELATIONSHIP_SEED_KEYS:
        if key in raw_payload:
            seeded[key] = raw_payload.get(key)
    return seeded


def _hs1j_parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return False


def _hs1j_build_rollout_status(*, raw_hints: object) -> dict:
    hints = raw_hints if isinstance(raw_hints, dict) else {}
    stage2_requirements: dict[str, bool] = {}
    for key in HS1J_ROLLOUT_HINT_KEYS:
        stage2_requirements[key] = _hs1j_parse_bool(hints.get(key))

    requirements_confirmed = bool(stage2_requirements) and all(stage2_requirements.values())
    return {
        "policy_version": HS1J_RELATIONSHIP_V2_ROLLOUT_POLICY_VERSION,
        "strategy_stage": HS1J_REL_V2_STAGE1_MODE,
        "normalized_table_split": {
            "eligible": requirements_confirmed,
            "activated": False,
            "activation_gate": HS1J_REL_V2_STAGE2_GATE,
        },
        "stage2_requirements": stage2_requirements,
    }


def _hs1j_summarize_rollout_status(*, rel_changes_v2: list | None) -> dict:
    total_items = 0
    eligible_count = 0

    for item in rel_changes_v2 or []:
        if not isinstance(item, dict):
            continue
        total_items += 1
        rollout = item.get("relationship_v2_rollout") if isinstance(item.get("relationship_v2_rollout"), dict) else {}
        split = rollout.get("normalized_table_split") if isinstance(rollout.get("normalized_table_split"), dict) else {}
        if bool(split.get("eligible")):
            eligible_count += 1

    return {
        "relationship_v2_rollout_policy_version": HS1J_RELATIONSHIP_V2_ROLLOUT_POLICY_VERSION,
        "relationship_v2_rollout_stage": HS1J_REL_V2_STAGE1_MODE,
        "relationship_v2_normalized_split_gate": HS1J_REL_V2_STAGE2_GATE,
        "relationship_v2_normalized_split_activated": False,
        "relationship_v2_normalized_split_eligible": eligible_count > 0,
        "relationship_v2_normalized_split_eligible_count": eligible_count,
        "relationship_v2_rollout_item_count": total_items,
    }


def _hs1k_is_relationship_v2_item(raw_item: object) -> bool:
    if not isinstance(raw_item, dict):
        return False
    required_dict_layers = ("identity", "core_state", "dynamics", "context", "history", "verification")
    return all(isinstance(raw_item.get(layer), dict) for layer in required_dict_layers)


def _hs1k_summarize_backfill_status(*, rel_changes_v2: list | None) -> dict:
    item_count = 0
    legacy_count = 0
    converted_count = 0
    defaults_applied_count = 0

    for item in rel_changes_v2 or []:
        if not isinstance(item, dict):
            continue
        item_count += 1
        if str(item.get("relationship_state_backfill_source") or "") == "legacy_v1":
            legacy_count += 1
            converted_count += 1
        verification = item.get("verification") if isinstance(item.get("verification"), dict) else {}
        if bool(verification.get("provenance_defaults_applied")):
            defaults_applied_count += 1

    return {
        "relationship_v2_backfill_policy_version": HS1K_RELATIONSHIP_V2_BACKFILL_POLICY_VERSION,
        "relationship_v2_backfill_mode": HS1K_RELATIONSHIP_V2_BACKFILL_MODE,
        "relationship_v2_backfill_item_count": item_count,
        "relationship_v2_backfill_legacy_v1_item_count": legacy_count,
        "relationship_v2_backfill_converted_item_count": converted_count,
        "relationship_v2_backfill_provenance_defaults_applied_count": defaults_applied_count,
    }


def _hs1k_backfill_relationship_list_minimal_v2(*, raw_items: object, turn_index: int) -> tuple[list, dict]:
    if not isinstance(raw_items, list):
        empty_summary = {
            "policy_version": HS1K_RELATIONSHIP_V2_BACKFILL_POLICY_VERSION,
            "mode": HS1K_RELATIONSHIP_V2_BACKFILL_MODE,
            "item_count": 0,
            "legacy_v1_item_count": 0,
            "converted_item_count": 0,
            "provenance_defaults_applied_count": 0,
        }
        return [], empty_summary

    backfilled_items: list = []
    legacy_count = 0
    converted_count = 0
    defaults_applied_count = 0

    for raw_item in raw_items:
        was_v2 = _hs1k_is_relationship_v2_item(raw_item)
        if not was_v2:
            legacy_count += 1

        try:
            enriched = _hs1g_enrich_relationship_change_item(raw_item=raw_item, turn_index=turn_index)
        except Exception:
            if isinstance(raw_item, dict):
                enriched = dict(raw_item)
            else:
                enriched = {"change": str(raw_item or "").strip()}

        if not was_v2 and isinstance(enriched, dict):
            converted_count += 1

        verification = enriched.get("verification") if isinstance(enriched.get("verification"), dict) else {}
        if bool(verification.get("provenance_defaults_applied")):
            defaults_applied_count += 1

        backfilled_items.append(enriched)

    summary = {
        "policy_version": HS1K_RELATIONSHIP_V2_BACKFILL_POLICY_VERSION,
        "mode": HS1K_RELATIONSHIP_V2_BACKFILL_MODE,
        "item_count": len(backfilled_items),
        "legacy_v1_item_count": legacy_count,
        "converted_item_count": converted_count,
        "provenance_defaults_applied_count": defaults_applied_count,
    }
    return backfilled_items, summary


def _hs1g_build_branch_registry(*, raw_registry: object, turn_index: int) -> dict:
    registry_input = raw_registry if isinstance(raw_registry, dict) else {}
    normalized_registry: dict[str, dict] = {}

    for branch_key in HS1G_BRANCH_REGISTRY_KEYS:
        raw_branch = registry_input.get(branch_key)
        branch_state = None
        branch_activation = None
        branch_decay = None
        last_updated_turn = None

        if isinstance(raw_branch, dict):
            raw_state = raw_branch.get("state")
            if raw_state is not None:
                text = str(raw_state).strip()
                branch_state = text or None
            branch_activation = _hs1e_parse_float(raw_branch.get("activation"))
            branch_decay = _hs1e_parse_float(raw_branch.get("decay"))
            last_updated_turn = _hs1e_parse_int(raw_branch.get("last_updated_turn"))
        elif raw_branch is not None:
            text = str(raw_branch).strip()
            branch_state = text or None
            last_updated_turn = turn_index

        normalized_registry[branch_key] = {
            "state": branch_state,
            "activation": branch_activation if branch_activation is not None else 0.0,
            "decay": branch_decay if branch_decay is not None else 0.0,
            "last_updated_turn": last_updated_turn,
        }

    return normalized_registry


def _hs1g_enrich_relationship_change_item(*, raw_item: object, turn_index: int) -> dict:
    raw_payload = dict(raw_item) if isinstance(raw_item, dict) else {"change": str(raw_item or "").strip()}
    enriched = _hs1i_seed_relationship_payload(raw_payload)
    normalized = _ea1h_normalize_relationship_change_item(raw_payload)
    rollout_status = _hs1j_build_rollout_status(raw_hints=raw_payload.get("_hs1j_rollout_hints"))
    backfill_source = "v2_existing" if _hs1k_is_relationship_v2_item(raw_payload) else "legacy_v1"

    left_entity = str(normalized.get("from") or "").strip()
    right_entity = str(normalized.get("to") or "").strip()
    pair_key = normalized.get("pair_key")

    pair_label = str(enriched.get("pair") or "").strip()
    if not pair_label and left_entity and right_entity:
        pair_label = f"{left_entity}-{right_entity}"

    identity = {
        "pair_key": pair_key,
        "left_entity": left_entity,
        "right_entity": right_entity,
    }
    if pair_label:
        identity["pair"] = pair_label

    core_existing = raw_payload.get("core_state") if isinstance(raw_payload.get("core_state"), dict) else {}
    explicit_confidence = _hs1e_parse_float(enriched.get("confidence"))
    core_confidence = explicit_confidence
    if core_confidence is None:
        core_confidence = _hs1e_parse_float(core_existing.get("confidence"))
    core_state = {
        "current_fact": str(
            normalized.get("change")
            or core_existing.get("current_fact")
            or core_existing.get("change")
            or ""
        ).strip(),
        "reason": str(normalized.get("reason") or core_existing.get("reason") or "").strip(),
        "context_tag": str(normalized.get("context_tag") or core_existing.get("context_tag") or "").strip(),
        "field_class": str(normalized.get("field_class") or core_existing.get("field_class") or "relationship"),
        "confidence": core_confidence,
    }

    dynamics_existing = raw_payload.get("dynamics") if isinstance(raw_payload.get("dynamics"), dict) else {}
    activation_score = _hs1e_parse_float(dynamics_existing.get("activation_score"))
    if activation_score is None:
        activation_score = _hs1e_parse_float((dynamics_existing.get("activation") or {}).get("score"))
    decay_score = _hs1e_parse_float(dynamics_existing.get("decay_score"))
    if decay_score is None:
        decay_score = _hs1e_parse_float((dynamics_existing.get("decay") or {}).get("score"))

    dynamics = {
        "branch_registry": _hs1g_build_branch_registry(
            raw_registry=dynamics_existing.get("branch_registry") or enriched.get("branch_registry"),
            turn_index=turn_index,
        ),
        "activation_axis": {
            "score": activation_score if activation_score is not None else 0.0,
            "updated_turn": turn_index,
        },
        "decay_axis": {
            "score": decay_score if decay_score is not None else 0.0,
            "updated_turn": turn_index,
        },
    }

    context_existing = raw_payload.get("context") if isinstance(raw_payload.get("context"), dict) else {}
    context_layer = {
        "turn_index": turn_index,
        "source": str(context_existing.get("source") or "").strip() or "critic_delta",
    }
    scene_context_tag = str(
        context_existing.get("scene_context_tag")
        or normalized.get("context_tag")
        or ""
    ).strip()
    if scene_context_tag:
        context_layer["scene_context_tag"] = scene_context_tag

    history_layer = {
        "latest_transition": {
            "turn_index": turn_index,
            "change": str(normalized.get("change") or "").strip(),
            "reason": str(normalized.get("reason") or "").strip(),
            "conflict_decision": (
                (enriched.get("_ea1h_conflict") or {}).get("decision")
                if isinstance(enriched.get("_ea1h_conflict"), dict)
                else None
            ),
        }
    }

    verification_existing = raw_payload.get("verification") if isinstance(raw_payload.get("verification"), dict) else {}
    source_record = _hs1e_parse_int(enriched.get("source_record"))
    if source_record is None:
        source_record = _hs1e_parse_int(verification_existing.get("source_record"))
    source_turn = _hs1e_parse_int(enriched.get("source_turn"))
    if source_turn is None:
        source_turn = _hs1e_parse_int(verification_existing.get("source_turn"))
    source_turn_defaulted = False
    if source_turn is None:
        source_turn = turn_index
        source_turn_defaulted = True
    last_verified_turn = _hs1e_parse_int(enriched.get("last_verified_turn"))
    if last_verified_turn is None:
        last_verified_turn = _hs1e_parse_int(verification_existing.get("last_verified_turn"))
    has_verified_source = bool(source_record is not None and last_verified_turn is not None)

    provenance_defaults_applied = bool(
        source_record is None
        or last_verified_turn is None
        or source_turn_defaulted
    )

    verification_confidence = _hs1e_parse_float(verification_existing.get("confidence"))
    if verification_confidence is None:
        verification_confidence = explicit_confidence
    if verification_confidence is None:
        verification_confidence = core_confidence

    verification = {
        "policy_version": HS1G_RELATIONSHIP_STATE_POLICY_VERSION,
        "minimal_schema_policy_version": HS1I_REL_WORLD_MINIMAL_SCHEMA_POLICY_VERSION,
        "rollout_policy_version": HS1J_RELATIONSHIP_V2_ROLLOUT_POLICY_VERSION,
        "backfill_policy_version": HS1K_RELATIONSHIP_V2_BACKFILL_POLICY_VERSION,
        "backfill_mode": HS1K_RELATIONSHIP_V2_BACKFILL_MODE,
        "backfill_source": backfill_source,
        "provenance_defaults_applied": provenance_defaults_applied,
        "source_turn_defaulted": source_turn_defaulted,
        "has_verified_source": has_verified_source,
        "source_record": source_record,
        "source_turn": source_turn,
        "last_verified_turn": last_verified_turn,
        "verification_state": "verified" if has_verified_source else "unverified",
        "confidence": verification_confidence,
    }

    if left_entity and not enriched.get("from"):
        enriched["from"] = left_entity
    if right_entity and not enriched.get("to"):
        enriched["to"] = right_entity
    if pair_label and not enriched.get("pair"):
        enriched["pair"] = pair_label
    if normalized.get("change") and not enriched.get("change"):
        enriched["change"] = normalized.get("change")
    if normalized.get("reason") and not enriched.get("reason"):
        enriched["reason"] = normalized.get("reason")
    if normalized.get("context_tag") and not enriched.get("context_tag"):
        enriched["context_tag"] = normalized.get("context_tag")
    if normalized.get("field_class") and not enriched.get("field_class"):
        enriched["field_class"] = normalized.get("field_class")
    if "confidence" in enriched:
        enriched["confidence"] = _hs1e_parse_float(enriched.get("confidence"))

    enriched["relationship_state_policy_version"] = HS1G_RELATIONSHIP_STATE_POLICY_VERSION
    enriched["relationship_state_schema_policy_version"] = HS1I_REL_WORLD_MINIMAL_SCHEMA_POLICY_VERSION
    enriched["relationship_state_backfill_policy_version"] = HS1K_RELATIONSHIP_V2_BACKFILL_POLICY_VERSION
    enriched["relationship_state_backfill_source"] = backfill_source
    enriched["relationship_v2_rollout"] = rollout_status
    enriched["identity"] = identity
    enriched["core_state"] = core_state
    enriched["dynamics"] = dynamics
    enriched["context"] = context_layer
    enriched["history"] = history_layer
    enriched["verification"] = verification
    return enriched


def _hs1g_enrich_relationship_changes_v2(*, rel_changes: list, turn_index: int) -> list:
    enriched_list: list = []
    for item in rel_changes:
        try:
            enriched_list.append(_hs1g_enrich_relationship_change_item(raw_item=item, turn_index=turn_index))
        except Exception:
            if isinstance(item, dict):
                enriched_list.append(dict(item))
            else:
                enriched_list.append({"change": str(item or "").strip()})
    return enriched_list


def _hs1h_parse_json_object(raw_value: object) -> dict:
    if isinstance(raw_value, dict):
        return dict(raw_value)
    if not isinstance(raw_value, str):
        return {}
    text = raw_value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _hs1h_preview_text(raw_value: object, *, max_len: int = 180) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        text = raw_value.strip()
    else:
        try:
            text = json.dumps(raw_value, ensure_ascii=False)
        except Exception:
            text = str(raw_value).strip()
    if not text:
        return None
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return f"{normalized[:max_len]}..."


def _hs1h_is_faction_status_rule(*, scope: str, key: str, value_preview: str | None) -> bool:
    scope_norm = _ea1h_normalize_text(scope)
    if scope_norm == "faction":
        return True
    merged = _ea1h_normalize_text(f"{key} {value_preview or ''}")
    return ("faction" in merged) or ("세력" in merged)


def _hs1h_is_region_pressure_rule(*, scope: str, key: str, value_preview: str | None) -> bool:
    scope_norm = _ea1h_normalize_text(scope)
    if scope_norm == "region":
        return True

    merged = _ea1h_normalize_text(f"{key} {value_preview or ''}")
    return any(token in merged for token in HS1H_REGION_PRESSURE_HINT_TOKENS)


def _hs1h_is_offscreen_thread_candidate(
    *,
    thread_type: str,
    title: str,
    owner: str | None,
    target: str | None,
    details_preview: str | None,
) -> bool:
    thread_type_norm = _ea1h_normalize_text(thread_type)
    if thread_type_norm in HS1H_OFFSCREEN_THREAD_TYPES:
        return True

    merged = _ea1h_normalize_text(
        f"{title} {owner or ''} {target or ''} {details_preview or ''}"
    )
    return any(token in merged for token in HS1H_OFFSCREEN_HINT_TOKENS)


def _hs1i_project_world_items(*, raw_items: object, allowed_keys: tuple[str, ...], limit: int) -> list[dict]:
    if not isinstance(raw_items, list):
        return []

    projected: list[dict] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        item: dict = {}
        for key in allowed_keys:
            if key in raw_item:
                item[key] = raw_item.get(key)
        if item:
            projected.append(item)
        if len(projected) >= limit:
            break
    return projected


def _hs1i_project_world_current_state(*, raw_world: object, turn_index: int) -> dict:
    source = raw_world if isinstance(raw_world, dict) else {}

    world_rules = _hs1i_project_world_items(
        raw_items=source.get("world_rules"),
        allowed_keys=HS1I_WORLD_RULE_KEYS,
        limit=HS1H_WORLD_RULES_LIMIT,
    )
    faction_status = _hs1i_project_world_items(
        raw_items=source.get("faction_status"),
        allowed_keys=HS1I_FACTION_STATUS_KEYS,
        limit=HS1H_FACTION_STATUS_LIMIT,
    )
    region_pressure = _hs1i_project_world_items(
        raw_items=source.get("region_pressure"),
        allowed_keys=HS1I_REGION_PRESSURE_KEYS,
        limit=HS1H_REGION_PRESSURE_LIMIT,
    )
    offscreen_threads = _hs1i_project_world_items(
        raw_items=source.get("offscreen_threads"),
        allowed_keys=HS1I_OFFSCREEN_THREAD_KEYS,
        limit=HS1H_OFFSCREEN_THREAD_LIMIT,
    )

    projected = {
        "policy_version": HS1H_WORLD_CURRENT_STATE_POLICY_VERSION,
        "minimal_schema_policy_version": HS1I_REL_WORLD_MINIMAL_SCHEMA_POLICY_VERSION,
        "turn_index": _hs1e_parse_int(source.get("turn_index")) or turn_index,
        "world_rules": world_rules,
        "faction_status": faction_status,
        "region_pressure": region_pressure,
        "offscreen_threads": offscreen_threads,
        "counts": {
            "world_rules": len(world_rules),
            "faction_status": len(faction_status),
            "region_pressure": len(region_pressure),
            "offscreen_threads": len(offscreen_threads),
        },
    }

    build_status = str(source.get("build_status") or "").strip()
    if build_status:
        projected["build_status"] = build_status

    return projected


def _hs1h_build_world_current_state_snapshot(*, chat_session_id: str, turn_index: int) -> dict:
    db: Session = SessionLocal()
    try:
        wr_rows = (
            db.query(WorldRule)
            .filter(
                WorldRule.chat_session_id == chat_session_id,
                WorldRule.suppressed == False,  # noqa: E712
            )
            .order_by(
                WorldRule.pinned.desc(),
                WorldRule.source_turn.desc(),
                WorldRule.id.desc(),
            )
            .limit(HS1H_WORLD_RULES_LIMIT * 4)
            .all()
        )

        world_rules: list[dict] = []
        faction_status: list[dict] = []
        region_pressure: list[dict] = []

        for row in wr_rows:
            scope = str(row.scope or "").strip() or "root"
            scope_name = str(row.scope_name or "").strip() or None
            category = str(row.category or "").strip() or None
            key = str(row.key or "").strip()
            if not key:
                continue

            value_obj = _hs1h_parse_json_object(row.value_json)
            value_preview = _hs1h_preview_text(value_obj if value_obj else row.value_json)

            world_item = {
                "scope": scope,
                "scope_name": scope_name,
                "category": category,
                "key": key,
                "value_preview": value_preview,
                "source_turn": _hs1e_parse_int(row.source_turn),
            }
            world_rules.append(world_item)

            if _hs1h_is_faction_status_rule(scope=scope, key=key, value_preview=value_preview):
                faction_state = str(
                    value_obj.get("state")
                    or value_obj.get("status")
                    or value_obj.get("alignment")
                    or ""
                ).strip() or None
                faction_status.append({
                    "scope_name": scope_name,
                    "key": key,
                    "state": faction_state,
                    "value_preview": value_preview,
                    "source_turn": _hs1e_parse_int(row.source_turn),
                })

            pressure_signal = (
                value_obj.get("pressure")
                or value_obj.get("threat")
                or value_obj.get("risk")
                or value_obj.get("tension")
            )
            pressure_value = _hs1e_parse_float(pressure_signal)
            pressure_repr = pressure_value if pressure_value is not None else (str(pressure_signal).strip() or None)

            if _hs1h_is_region_pressure_rule(scope=scope, key=key, value_preview=value_preview):
                region_pressure.append({
                    "scope": scope,
                    "scope_name": scope_name,
                    "key": key,
                    "pressure": pressure_repr,
                    "value_preview": value_preview,
                    "source_turn": _hs1e_parse_int(row.source_turn),
                })

        thread_rows = (
            db.query(PendingThread)
            .filter(
                PendingThread.chat_session_id == chat_session_id,
                PendingThread.status.in_(["open", "paused"]),
                PendingThread.suppressed == False,  # noqa: E712
            )
            .order_by(
                PendingThread.pinned.desc(),
                PendingThread.last_seen_turn.desc(),
                PendingThread.id.desc(),
            )
            .limit(HS1H_OFFSCREEN_THREAD_LIMIT * 5)
            .all()
        )

        offscreen_threads: list[dict] = []
        for row in thread_rows:
            thread_type = str(row.thread_type or "").strip()
            title = str(row.title or "").strip()
            owner = str(row.owner or "").strip() or None
            target = str(row.target or "").strip() or None
            details_obj = _hs1h_parse_json_object(row.details_json)
            details_preview = _hs1h_preview_text(details_obj if details_obj else row.details_json)

            if not _hs1h_is_offscreen_thread_candidate(
                thread_type=thread_type,
                title=title,
                owner=owner,
                target=target,
                details_preview=details_preview,
            ):
                continue

            offscreen_threads.append({
                "thread_type": thread_type,
                "title": title,
                "owner": owner,
                "target": target,
                "status": str(row.status or "").strip() or "open",
                "last_seen_turn": _hs1e_parse_int(row.last_seen_turn),
                "confidence": _hs1e_parse_float(row.confidence),
                "details_preview": details_preview,
            })

        world_rules = world_rules[:HS1H_WORLD_RULES_LIMIT]
        faction_status = faction_status[:HS1H_FACTION_STATUS_LIMIT]
        region_pressure = region_pressure[:HS1H_REGION_PRESSURE_LIMIT]
        offscreen_threads = offscreen_threads[:HS1H_OFFSCREEN_THREAD_LIMIT]

        snapshot = {
            "policy_version": HS1H_WORLD_CURRENT_STATE_POLICY_VERSION,
            "turn_index": turn_index,
            "world_rules": world_rules,
            "faction_status": faction_status,
            "region_pressure": region_pressure,
            "offscreen_threads": offscreen_threads,
            "counts": {
                "world_rules": len(world_rules),
                "faction_status": len(faction_status),
                "region_pressure": len(region_pressure),
                "offscreen_threads": len(offscreen_threads),
            },
        }
        return _hs1i_project_world_current_state(raw_world=snapshot, turn_index=turn_index)
    finally:
        db.close()


def _hs1h_enrich_scene_state_with_world_current(
    *,
    chat_session_id: str,
    turn_index: int,
    scene_state: dict | None,
) -> dict:
    base_scene: dict = dict(scene_state) if isinstance(scene_state, dict) else {}

    try:
        world_snapshot = _hs1h_build_world_current_state_snapshot(
            chat_session_id=chat_session_id,
            turn_index=turn_index,
        )
    except Exception as exc:
        logger.warning("hs1h world current state build failed (session=%s, turn=%d): %s", chat_session_id, turn_index, exc)
        world_snapshot = {
            "policy_version": HS1H_WORLD_CURRENT_STATE_POLICY_VERSION,
            "turn_index": turn_index,
            "world_rules": [],
            "faction_status": [],
            "region_pressure": [],
            "offscreen_threads": [],
            "counts": {
                "world_rules": 0,
                "faction_status": 0,
                "region_pressure": 0,
                "offscreen_threads": 0,
            },
            "build_status": "failed",
        }

    existing_world = base_scene.get("world_current_state")
    if isinstance(existing_world, dict):
        merged_world = dict(existing_world)
        merged_world.update(world_snapshot)
        world_snapshot = merged_world

    world_snapshot = _hs1i_project_world_current_state(raw_world=world_snapshot, turn_index=turn_index)
    base_scene["world_current_state_policy_version"] = HS1H_WORLD_CURRENT_STATE_POLICY_VERSION
    base_scene["world_current_state_schema_policy_version"] = HS1I_REL_WORLD_MINIMAL_SCHEMA_POLICY_VERSION
    base_scene["world_current_state"] = world_snapshot
    return base_scene


def save_state_deltas(
    *,
    chat_session_id: str,
    turn_index: int,
    state_deltas: dict,
) -> dict:
    """Critic이 산출한 state_deltas를 파싱해 active_states에 append한다.

    state_deltas 구조:
      {
        "scene_state": { ... } | null,
        "relationship_changes": [ ... ],
        "unresolved_threads": { "opened": [...], "resolved": [...] }
      }

    각 항목이 존재하면 해당 state_type으로 append.
    파싱/저장 실패 시 개별 항목만 건너뛰고, 전체를 중단하지 않는다.
    Returns: { saved: [...], skipped: [...], errors: [...] }
    """
    lc1b_policy_version = "lc1b.v1"
    save_started_at = perf_counter()

    result = {
        "saved": [],
        "skipped": [],
        "errors": [],
        "canonical_state_upsert": {
            "policy_version": HS1B_CANONICAL_UPSERT_POLICY_VERSION,
            "canonical_write_cost_measurement_policy_version": lc1b_policy_version,
            "provenance_policy_version": HS1E_CANONICAL_PROVENANCE_POLICY_VERSION,
            "verified_only_promotion_policy_version": HS1F_CANONICAL_VERIFIED_ONLY_PROMOTION_POLICY_VERSION,
            "relationship_state_policy_version": HS1G_RELATIONSHIP_STATE_POLICY_VERSION,
            "world_current_state_policy_version": HS1H_WORLD_CURRENT_STATE_POLICY_VERSION,
            "relationship_world_schema_policy_version": HS1I_REL_WORLD_MINIMAL_SCHEMA_POLICY_VERSION,
            "relationship_v2_rollout_policy_version": HS1J_RELATIONSHIP_V2_ROLLOUT_POLICY_VERSION,
            "relationship_v2_backfill_policy_version": HS1K_RELATIONSHIP_V2_BACKFILL_POLICY_VERSION,
            "unverified_promotion_guard": "block_unverified_estimate_without_verified_source",
            "mode": "critic_delta_upsert_with_last_good_fallback",
            "status": "pending",
            "items": {},
            "cost_measurement": {
                "policy_version": lc1b_policy_version,
                "state_write_count": 0,
                "delta_update_count": 0,
                "full_rewrite_count": 0,
                "fallback_count": 0,
                "avg_write_latency_ms": 0.0,
                "p95_write_latency_ms": 0.0,
                "total_write_chars": 0,
                "total_elapsed_ms": 0.0,
            },
        },
    }

    canonical_upsert = result["canonical_state_upsert"]
    canonical_items: dict[str, dict] = canonical_upsert["items"]

    def _lc1b_similarity_ratio(previous_content: str, next_content: str) -> float:
        prev_tokens = {
            token for token in str(previous_content or "").lower().split()
            if token
        }
        next_tokens = {
            token for token in str(next_content or "").lower().split()
            if token
        }
        if not prev_tokens and not next_tokens:
            return 1.0
        if not prev_tokens or not next_tokens:
            return 0.0
        union = prev_tokens | next_tokens
        if not union:
            return 0.0
        return len(prev_tokens & next_tokens) / len(union)

    def _lc1b_finalize_elapsed() -> None:
        canonical_upsert["cost_measurement"]["total_elapsed_ms"] = round(
            (perf_counter() - save_started_at) * 1000.0,
            3,
        )

    def _record_last_good_fallback(
        state_type: str,
        reason: str,
        error: Exception | str | None = None,
        candidate_provenance: dict | None = None,
        state_metadata: dict | None = None,
    ) -> dict:
        latest_rows = get_all_canonical_state_layers_for_session(
            chat_session_id=sid,
            layer_type=state_type,
        )
        latest = latest_rows[0] if latest_rows else None

        payload = {
            "status": "last_good_retained" if latest else "empty_no_last_good",
            "reason": reason,
            "fallback_used": True,
        }
        if latest:
            payload.update({
                "last_good_id": latest.get("id"),
                "last_good_turn_index": latest.get("turn_index"),
                "last_good_source_turn": latest.get("source_turn"),
                "last_good_source_record": latest.get("source_record"),
                "last_good_last_verified_turn": latest.get("last_verified_turn"),
                "last_good_confidence": latest.get("confidence"),
                "last_good_created_at": latest.get("created_at"),
            })
        if isinstance(candidate_provenance, dict):
            payload["candidate_provenance"] = candidate_provenance
        if isinstance(state_metadata, dict):
            payload.update(state_metadata)
        if error is not None:
            payload["error"] = str(error)[:160]

        canonical_items[state_type] = payload
        return payload

    def _upsert_canonical_layer_or_fallback(
        state_type: str,
        content: str,
        state_metadata: dict | None = None,
    ) -> dict:
        candidate_provenance = _hs1e_build_candidate_provenance(
            state_type=state_type,
            turn_index=turn_index,
            content=content,
            direct_evidence_seed=direct_evidence_seed,
        )

        if _hs1f_should_block_unverified_estimate_promotion(provenance=candidate_provenance):
            logger.info(
                "save_state_deltas canonical promotion blocked by hs1f guard (%s, session=%s, turn=%d)",
                state_type,
                sid,
                turn_index,
            )
            return _record_last_good_fallback(
                state_type,
                reason="hs1f_unverified_estimate_blocked",
                candidate_provenance=candidate_provenance,
            )

        latest_rows = get_all_canonical_state_layers_for_session(
            chat_session_id=sid,
            layer_type=state_type,
        )
        latest_before = latest_rows[0] if latest_rows else None
        previous_content = str((latest_before or {}).get("content") or "")
        previous_chars = len(previous_content)
        next_chars = len(str(content or ""))
        char_delta = abs(next_chars - previous_chars)
        similarity = _lc1b_similarity_ratio(previous_content, content)
        if previous_chars <= 0:
            write_mode = "full_rewrite_bootstrap"
        elif similarity >= 0.55:
            write_mode = "delta_update"
        else:
            write_mode = "full_rewrite"

        write_started_at = perf_counter()

        try:
            row = append_canonical_state_layer(
                chat_session_id=sid,
                layer_type=state_type,
                content=content,
                turn_index=turn_index,
                source_state_type=state_type,
                source_turn=candidate_provenance.get("source_turn"),
                source_record=candidate_provenance.get("source_record"),
                last_verified_turn=candidate_provenance.get("last_verified_turn"),
                confidence=candidate_provenance.get("confidence"),
            )
            write_latency_ms = round((perf_counter() - write_started_at) * 1000.0, 3)
            payload = {
                "status": "upserted",
                "reason": "critic_delta_applied",
                "fallback_used": False,
                "row_id": row.get("id"),
                "turn_index": row.get("turn_index"),
                "source_turn": row.get("source_turn"),
                "source_record": row.get("source_record"),
                "last_verified_turn": row.get("last_verified_turn"),
                "confidence": row.get("confidence"),
                "provenance_policy_version": HS1E_CANONICAL_PROVENANCE_POLICY_VERSION,
                "verified_only_promotion_policy_version": HS1F_CANONICAL_VERIFIED_ONLY_PROMOTION_POLICY_VERSION,
                "canonical_write_cost_measurement_policy_version": lc1b_policy_version,
                "write_mode": write_mode,
                "write_latency_ms": write_latency_ms,
                "previous_content_chars": previous_chars,
                "new_content_chars": next_chars,
                "char_delta_abs": char_delta,
                "token_similarity": round(similarity, 4),
            }
            if isinstance(state_metadata, dict):
                payload.update(state_metadata)
            if state_type == "relationship_state":
                payload["relationship_state_policy_version"] = HS1G_RELATIONSHIP_STATE_POLICY_VERSION
                payload["relationship_state_schema_policy_version"] = HS1I_REL_WORLD_MINIMAL_SCHEMA_POLICY_VERSION
            if state_type == "scene_state":
                payload["world_current_state_policy_version"] = HS1H_WORLD_CURRENT_STATE_POLICY_VERSION
                payload["world_current_state_schema_policy_version"] = HS1I_REL_WORLD_MINIMAL_SCHEMA_POLICY_VERSION
            canonical_items[state_type] = payload
            return payload
        except Exception as mirror_exc:
            logger.warning("save_state_deltas canonical layer upsert failed (%s): %s", state_type, mirror_exc)
            return _record_last_good_fallback(
                state_type,
                reason="canonical_write_failed",
                error=mirror_exc,
                candidate_provenance=candidate_provenance,
            )

    def _finalize_canonical_upsert_status() -> None:
        for state_type in HS1B_CANONICAL_LAYER_ORDER:
            if state_type not in canonical_items:
                _record_last_good_fallback(state_type, reason="hs1b_internal_missing_state")

        statuses = [str((canonical_items.get(st) or {}).get("status") or "") for st in HS1B_CANONICAL_LAYER_ORDER]
        upserted_count = sum(1 for status in statuses if status == "upserted")
        fallback_count = sum(1 for status in statuses if status in {"last_good_retained", "empty_no_last_good"})
        empty_fallback_count = sum(1 for status in statuses if status == "empty_no_last_good")

        canonical_upsert["upserted_count"] = upserted_count
        canonical_upsert["fallback_count"] = fallback_count
        canonical_upsert["empty_fallback_count"] = empty_fallback_count

        write_latencies: list[float] = []
        delta_update_count = 0
        full_rewrite_count = 0
        total_write_chars = 0
        for state_type in HS1B_CANONICAL_LAYER_ORDER:
            item = canonical_items.get(state_type) or {}
            if str(item.get("status") or "") == "upserted":
                latency_ms = item.get("write_latency_ms")
                if isinstance(latency_ms, (int, float)):
                    write_latencies.append(float(latency_ms))
                write_mode = str(item.get("write_mode") or "")
                if write_mode == "delta_update":
                    delta_update_count += 1
                elif write_mode in {"full_rewrite", "full_rewrite_bootstrap"}:
                    full_rewrite_count += 1
                total_write_chars += int(item.get("new_content_chars") or 0)

        avg_latency_ms = round(sum(write_latencies) / len(write_latencies), 3) if write_latencies else 0.0
        p95_latency_ms = 0.0
        if write_latencies:
            sorted_latencies = sorted(write_latencies)
            p95_index = int((len(sorted_latencies) - 1) * 0.95)
            p95_latency_ms = round(sorted_latencies[p95_index], 3)

        canonical_upsert["cost_measurement"] = {
            "policy_version": lc1b_policy_version,
            "state_write_count": len(write_latencies),
            "delta_update_count": delta_update_count,
            "full_rewrite_count": full_rewrite_count,
            "fallback_count": fallback_count,
            "avg_write_latency_ms": avg_latency_ms,
            "p95_write_latency_ms": p95_latency_ms,
            "total_write_chars": total_write_chars,
            "total_elapsed_ms": round((perf_counter() - save_started_at) * 1000.0, 3),
        }

        if upserted_count == len(HS1B_CANONICAL_LAYER_ORDER):
            canonical_upsert["status"] = "ok"
        elif upserted_count > 0 and fallback_count > 0:
            canonical_upsert["status"] = "partial"
        elif upserted_count == 0 and fallback_count == len(HS1B_CANONICAL_LAYER_ORDER):
            canonical_upsert["status"] = "empty" if empty_fallback_count == len(HS1B_CANONICAL_LAYER_ORDER) else "fallback_last_good"
        else:
            canonical_upsert["status"] = "degraded"

    if not state_deltas or not isinstance(state_deltas, dict):
        canonical_upsert["status"] = "off"
        canonical_upsert["reason"] = "no_state_deltas"
        _lc1b_finalize_elapsed()
        return result

    sid = _safe_session_id(chat_session_id)
    if not sid:
        result["errors"].append("empty chat_session_id")
        canonical_upsert["status"] = "error"
        canonical_upsert["reason"] = "empty_chat_session_id"
        _lc1b_finalize_elapsed()
        return result

    # EA-1c canonical 승격 guard:
    # 동일 turn에 repair/rejected 신호가 있는데 gate 승인 direct evidence가 0건이면
    # active_state 승격을 보류한다.
    gate_snapshot = _get_direct_evidence_gate_snapshot_for_turn(
        chat_session_id=sid,
        turn_index=turn_index,
    )
    if gate_snapshot["blocked_count"] > 0 and gate_snapshot["approved_count"] == 0:
        result["skipped"].extend([
            "scene_state",
            "relationship_state",
            "unresolved_threads",
        ])
        result["errors"].append("direct_evidence_gate_blocked")
        for state_type in HS1B_CANONICAL_LAYER_ORDER:
            _record_last_good_fallback(state_type, reason="direct_evidence_gate_blocked")
        _finalize_canonical_upsert_status()
        logger.info(
            "save_state_deltas gated (session=%s, turn=%d, blocked=%d, approved=%d)",
            sid,
            turn_index,
            gate_snapshot["blocked_count"],
            gate_snapshot["approved_count"],
        )
        _lc1b_finalize_elapsed()
        return result

    direct_evidence_seed = _collect_gate_approved_direct_evidence_seed_for_turn(
        chat_session_id=sid,
        turn_index=turn_index,
    )

    rel_changes_raw = state_deltas.get("relationship_changes")
    ea1h_conflict_result = _ea1h_resolve_canonical_conflicts(
        chat_session_id=sid,
        turn_index=turn_index,
        relationship_changes=rel_changes_raw,
        direct_evidence_seed=direct_evidence_seed,
    )
    ea1h_conflict_summary = ea1h_conflict_result.get("summary")
    if isinstance(ea1h_conflict_summary, dict):
        result["conflict_resolution"] = ea1h_conflict_summary

    # 1. scene_state
    scene = _decorate_scene_state_with_direct_evidence_seed(
        scene_state=state_deltas.get("scene_state"),
        evidence_seed=direct_evidence_seed,
        conflict_resolution=ea1h_conflict_summary,
    )
    if isinstance(scene, dict):
        scene = _hs1h_enrich_scene_state_with_world_current(
            chat_session_id=sid,
            turn_index=turn_index,
            scene_state=scene,
        )
    if scene and isinstance(scene, dict):
        try:
            content = json.dumps(scene, ensure_ascii=False)
            append_active_state(
                chat_session_id=sid,
                state_type="scene_state",
                content=content,
                turn_index=turn_index,
            )
            _upsert_canonical_layer_or_fallback("scene_state", content)
            result["saved"].append("scene_state")
        except Exception as exc:
            logger.warning("save_state_deltas scene_state failed: %s", exc)
            result["errors"].append(f"scene_state: {exc}")
            _record_last_good_fallback("scene_state", reason="active_state_save_failed", error=exc)
    else:
        result["skipped"].append("scene_state")
        _record_last_good_fallback("scene_state", reason="critic_delta_missing_or_empty")

    # 2. relationship_state
    rel_changes = ea1h_conflict_result.get("promoted_relationship_changes")
    relationship_rollout_summary = _hs1j_summarize_rollout_status(rel_changes_v2=[])
    relationship_backfill_summary = _hs1k_summarize_backfill_status(rel_changes_v2=[])
    if isinstance(rel_changes, list) and rel_changes:
        try:
            rel_changes_v2 = _hs1g_enrich_relationship_changes_v2(
                rel_changes=rel_changes,
                turn_index=turn_index,
            )
            relationship_rollout_summary = _hs1j_summarize_rollout_status(rel_changes_v2=rel_changes_v2)
            relationship_backfill_summary = _hs1k_summarize_backfill_status(rel_changes_v2=rel_changes_v2)
            content = json.dumps(rel_changes_v2, ensure_ascii=False)
            append_active_state(
                chat_session_id=sid,
                state_type="relationship_state",
                content=content,
                turn_index=turn_index,
            )
            _upsert_canonical_layer_or_fallback(
                "relationship_state",
                content,
                state_metadata={
                    **relationship_rollout_summary,
                    **relationship_backfill_summary,
                },
            )
            result["saved"].append("relationship_state")
        except Exception as exc:
            logger.warning("save_state_deltas relationship_state failed: %s", exc)
            result["errors"].append(f"relationship_state: {exc}")
            _record_last_good_fallback(
                "relationship_state",
                reason="active_state_save_failed",
                error=exc,
                state_metadata={
                    **relationship_rollout_summary,
                    **relationship_backfill_summary,
                },
            )
    else:
        result["skipped"].append("relationship_state")
        _record_last_good_fallback(
            "relationship_state",
            reason="no_promoted_relationship_changes",
            state_metadata={
                **relationship_rollout_summary,
                **relationship_backfill_summary,
            },
        )
    result["relationship_v2_rollout"] = relationship_rollout_summary
    result["relationship_v2_backfill"] = relationship_backfill_summary

    # 3. unresolved_threads
    threads = _ea1h_merge_manual_review_threads(
        threads=state_deltas.get("unresolved_threads"),
        manual_review_threads=ea1h_conflict_result.get("manual_review_threads") or [],
        user_confirmation_threads=ea1h_conflict_result.get("user_confirmation_threads") or [],
    )
    if isinstance(threads, dict):
        opened = threads.get("opened", [])
        resolved = threads.get("resolved", [])
        has_content = (isinstance(opened, list) and opened) or (isinstance(resolved, list) and resolved)
        if has_content:
            try:
                content = json.dumps(threads, ensure_ascii=False)
                append_active_state(
                    chat_session_id=sid,
                    state_type="unresolved_threads",
                    content=content,
                    turn_index=turn_index,
                )
                _upsert_canonical_layer_or_fallback("unresolved_threads", content)
                result["saved"].append("unresolved_threads")
            except Exception as exc:
                logger.warning("save_state_deltas unresolved_threads failed: %s", exc)
                result["errors"].append(f"unresolved_threads: {exc}")
                _record_last_good_fallback("unresolved_threads", reason="active_state_save_failed", error=exc)
        else:
            result["skipped"].append("unresolved_threads")
            _record_last_good_fallback("unresolved_threads", reason="no_unresolved_thread_content")
    else:
        result["skipped"].append("unresolved_threads")
        _record_last_good_fallback("unresolved_threads", reason="unresolved_threads_invalid_shape")

    _finalize_canonical_upsert_status()
    _lc1b_finalize_elapsed()

    return result


# ------------------------------------------------------------------ #
#  Phase 3-1 — Episode Summary CRUD
# ------------------------------------------------------------------ #

def create_episode_summary(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    summary_text: str,
    key_entities: str | None = None,
    key_events: str | None = None,
    open_loops_json: str | None = None,
    relationship_changes_json: str | None = None,
    embedding_vector: str | None = None,
    embedding_model: str | None = None,
) -> dict:
    """episode_summaries에 1건 삽입하고 dict로 반환한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        raise ValueError("chat_session_id is required")

    db: Session = SessionLocal()
    try:
        row = EpisodeSummary(
            chat_session_id=sid,
            from_turn=from_turn,
            to_turn=to_turn,
            summary_text=summary_text,
            key_entities=key_entities,
            key_events=key_events,
            open_loops_json=open_loops_json,
            relationship_changes_json=relationship_changes_json,
            embedding_vector=embedding_vector,
            embedding_model=embedding_model,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _episode_summary_to_dict(row)
    finally:
        db.close()


def get_episode_summaries_for_session(
    *,
    chat_session_id: str,
    limit: int | None = None,
    from_turn: int | None = None,
    to_turn: int | None = None,
) -> list[dict]:
    """세션의 에피소드 요약 목록을 반환한다.

    from_turn / to_turn 필터를 제공하면 해당 범위와 겹치는 에피소드만 반환.
    """
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    lim = _clamp_limit(limit)

    db: Session = SessionLocal()
    try:
        q = db.query(EpisodeSummary).filter(EpisodeSummary.chat_session_id == sid)

        if from_turn is not None:
            q = q.filter(EpisodeSummary.to_turn >= from_turn)
        if to_turn is not None:
            q = q.filter(EpisodeSummary.from_turn <= to_turn)

        rows = q.order_by(EpisodeSummary.from_turn.asc()).limit(lim).all()
        return [_episode_summary_to_dict(r) for r in rows]
    finally:
        db.close()


def get_episode_summary_by_id(episode_id: int) -> dict | None:
    """id로 에피소드 요약 1건 조회."""
    db: Session = SessionLocal()
    try:
        row = db.query(EpisodeSummary).filter(EpisodeSummary.id == episode_id).first()
        return _episode_summary_to_dict(row) if row else None
    finally:
        db.close()


def _episode_summary_to_dict(row: EpisodeSummary) -> dict:
    payload = {
        "id": row.id,
        "chat_session_id": row.chat_session_id,
        "from_turn": row.from_turn,
        "to_turn": row.to_turn,
        "summary_text": row.summary_text,
        "key_entities": row.key_entities,
        "key_events": row.key_events,
        "open_loops_json": row.open_loops_json,
        "relationship_changes_json": row.relationship_changes_json,
        "episode_dense_summary_policy_version": "ds1a.v1",
        "embedding_vector": row.embedding_vector,
        "embedding_model": row.embedding_model,
        "created_at": to_kst(row.created_at),
    }
    payload.update(
        _build_dense_summary_role_split_ds1h(payload, tier="episode")
    )
    payload.update(
        _build_dense_summary_source_anchor_ds1f(
            source_record_id=row.id,
            source_record_type="episode_summary",
            from_turn=row.from_turn,
            to_turn=row.to_turn,
        )
    )
    return payload


# ------------------------------------------------------------------ #
#  Step 9 P-1a — Chapter Summary CRUD Contract
# ------------------------------------------------------------------ #

def create_chapter_summary(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    summary_text: str,
    chapter_index: int | None = None,
    chapter_title: str | None = None,
    open_loops_json: str | None = None,
    relationship_changes_json: str | None = None,
    world_changes_json: str | None = None,
    callback_candidates_json: str | None = None,
    resume_text: str | None = None,
    embedding_vector: str | None = None,
    embedding_model: str | None = None,
) -> dict:
    """chapter_summaries에 1건 삽입하고 dict로 반환한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        raise ValueError("chat_session_id is required")

    db: Session = SessionLocal()
    try:
        row = ChapterSummary(
            chat_session_id=sid,
            from_turn=from_turn,
            to_turn=to_turn,
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            summary_text=summary_text,
            open_loops_json=open_loops_json,
            relationship_changes_json=relationship_changes_json,
            world_changes_json=world_changes_json,
            callback_candidates_json=callback_candidates_json,
            resume_text=resume_text,
            embedding_vector=embedding_vector,
            embedding_model=embedding_model,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _chapter_summary_to_dict(row)
    finally:
        db.close()


def get_chapter_summaries_for_session(
    *,
    chat_session_id: str,
    limit: int | None = None,
    from_turn: int | None = None,
    to_turn: int | None = None,
) -> list[dict]:
    """세션의 chapter summary 목록을 반환한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    lim = _clamp_limit(limit)

    db: Session = SessionLocal()
    try:
        q = db.query(ChapterSummary).filter(ChapterSummary.chat_session_id == sid)

        if from_turn is not None:
            q = q.filter(ChapterSummary.to_turn >= from_turn)
        if to_turn is not None:
            q = q.filter(ChapterSummary.from_turn <= to_turn)

        rows = q.order_by(ChapterSummary.from_turn.asc()).limit(lim).all()
        return [_chapter_summary_to_dict(r) for r in rows]
    finally:
        db.close()


def get_chapter_summary_by_id(chapter_id: int) -> dict | None:
    """id로 chapter summary 1건 조회."""
    db: Session = SessionLocal()
    try:
        row = db.query(ChapterSummary).filter(ChapterSummary.id == chapter_id).first()
        return _chapter_summary_to_dict(row) if row else None
    finally:
        db.close()


def _chapter_summary_to_dict(row: ChapterSummary) -> dict:
    payload = {
        "id": row.id,
        "chat_session_id": row.chat_session_id,
        "from_turn": row.from_turn,
        "to_turn": row.to_turn,
        "chapter_index": row.chapter_index,
        "chapter_title": row.chapter_title,
        "summary_text": row.summary_text,
        "open_loops_json": row.open_loops_json,
        "relationship_changes_json": row.relationship_changes_json,
        "world_changes_json": row.world_changes_json,
        "callback_candidates_json": row.callback_candidates_json,
        "resume_text": row.resume_text,
        "embedding_vector": row.embedding_vector,
        "embedding_model": row.embedding_model,
        "created_at": to_kst(row.created_at),
    }
    payload.update(
        _build_dense_summary_role_split_ds1h(payload, tier="chapter")
    )
    payload.update(
        _build_dense_summary_source_anchor_ds1f(
            source_record_id=row.id,
            source_record_type="chapter_summary",
            from_turn=row.from_turn,
            to_turn=row.to_turn,
        )
    )
    return payload


# ------------------------------------------------------------------ #
#  Step 9 P-3a — Arc Summary CRUD Contract
# ------------------------------------------------------------------ #

def create_arc_summary(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    arc_status: str,
    arc_name: str | None = None,
    arc_index: int | None = None,
    core_conflict: str | None = None,
    key_turning_points_json: str | None = None,
    active_promises_json: str | None = None,
    unresolved_debts_json: str | None = None,
    resolved_payoffs_json: str | None = None,
    callback_candidates_json: str | None = None,
    irreversible_turns_json: str | None = None,
    callback_debts_json: str | None = None,
    relationship_pivots_json: str | None = None,
    future_payoff_candidates_json: str | None = None,
    arc_resume_text: str | None = None,
    embedding_vector: str | None = None,
    embedding_model: str | None = None,
) -> dict:
    """arc_summaries에 1건 삽입하고 dict로 반환한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        raise ValueError("chat_session_id is required")

    db: Session = SessionLocal()
    try:
        row = ArcSummary(
            chat_session_id=sid,
            from_turn=from_turn,
            to_turn=to_turn,
            arc_index=arc_index,
            arc_name=arc_name,
            arc_status=arc_status,
            core_conflict=core_conflict,
            key_turning_points_json=key_turning_points_json,
            active_promises_json=active_promises_json,
            unresolved_debts_json=unresolved_debts_json,
            resolved_payoffs_json=resolved_payoffs_json,
            callback_candidates_json=callback_candidates_json,
            irreversible_turns_json=irreversible_turns_json,
            callback_debts_json=callback_debts_json,
            relationship_pivots_json=relationship_pivots_json,
            future_payoff_candidates_json=future_payoff_candidates_json,
            arc_resume_text=arc_resume_text,
            embedding_vector=embedding_vector,
            embedding_model=embedding_model,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _arc_summary_to_dict(row)
    finally:
        db.close()


def get_arc_summaries_for_session(
    *,
    chat_session_id: str,
    limit: int | None = None,
    from_turn: int | None = None,
    to_turn: int | None = None,
) -> list[dict]:
    """세션의 arc summary 목록을 반환한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    lim = _clamp_limit(limit)

    db: Session = SessionLocal()
    try:
        q = db.query(ArcSummary).filter(ArcSummary.chat_session_id == sid)

        if from_turn is not None:
            q = q.filter(ArcSummary.to_turn >= from_turn)
        if to_turn is not None:
            q = q.filter(ArcSummary.from_turn <= to_turn)

        rows = q.order_by(ArcSummary.from_turn.asc()).limit(lim).all()
        return [_arc_summary_to_dict(r) for r in rows]
    finally:
        db.close()


def get_arc_summary_by_id(arc_id: int) -> dict | None:
    """id로 arc summary 1건 조회."""
    db: Session = SessionLocal()
    try:
        row = db.query(ArcSummary).filter(ArcSummary.id == arc_id).first()
        return _arc_summary_to_dict(row) if row else None
    finally:
        db.close()


def _arc_summary_to_dict(row: ArcSummary) -> dict:
    payload = {
        "id": row.id,
        "chat_session_id": row.chat_session_id,
        "from_turn": row.from_turn,
        "to_turn": row.to_turn,
        "arc_index": row.arc_index,
        "arc_name": row.arc_name,
        "arc_status": row.arc_status,
        "core_conflict": row.core_conflict,
        "key_turning_points_json": row.key_turning_points_json,
        "active_promises_json": row.active_promises_json,
        "unresolved_debts_json": row.unresolved_debts_json,
        "resolved_payoffs_json": row.resolved_payoffs_json,
        "callback_candidates_json": row.callback_candidates_json,
        "irreversible_turns_json": row.irreversible_turns_json,
        "callback_debts_json": row.callback_debts_json,
        "relationship_pivots_json": row.relationship_pivots_json,
        "future_payoff_candidates_json": row.future_payoff_candidates_json,
        "arc_resume_text": row.arc_resume_text,
        "embedding_vector": row.embedding_vector,
        "embedding_model": row.embedding_model,
        "created_at": to_kst(row.created_at),
    }
    payload.update(
        _build_dense_summary_role_split_ds1h(payload, tier="arc")
    )
    payload.update(
        _build_dense_summary_source_anchor_ds1f(
            source_record_id=row.id,
            source_record_type="arc_summary",
            from_turn=row.from_turn,
            to_turn=row.to_turn,
        )
    )
    return payload


# ------------------------------------------------------------------ #
#  Step 9 P-4a - Saga Digest CRUD Contract
# ------------------------------------------------------------------ #

def create_saga_digest(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    saga_summary: str,
    era_label: str | None = None,
    persistent_facts_json: str | None = None,
    never_drop_candidates_json: str | None = None,
    resume_pack_text: str | None = None,
    embedding_vector: str | None = None,
    embedding_model: str | None = None,
) -> dict:
    """saga_digests에 1건 삽입하고 dict로 반환한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        raise ValueError("chat_session_id is required")

    db: Session = SessionLocal()
    try:
        row = SagaDigest(
            chat_session_id=sid,
            from_turn=from_turn,
            to_turn=to_turn,
            era_label=era_label,
            saga_summary=saga_summary,
            persistent_facts_json=persistent_facts_json,
            never_drop_candidates_json=never_drop_candidates_json,
            resume_pack_text=resume_pack_text,
            embedding_vector=embedding_vector,
            embedding_model=embedding_model,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _saga_digest_to_dict(row)
    finally:
        db.close()


def get_saga_digests_for_session(
    *,
    chat_session_id: str,
    limit: int | None = None,
    from_turn: int | None = None,
    to_turn: int | None = None,
) -> list[dict]:
    """세션의 saga digest 목록을 반환한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    lim = _clamp_limit(limit)

    db: Session = SessionLocal()
    try:
        q = db.query(SagaDigest).filter(SagaDigest.chat_session_id == sid)

        if from_turn is not None:
            q = q.filter(SagaDigest.to_turn >= from_turn)
        if to_turn is not None:
            q = q.filter(SagaDigest.from_turn <= to_turn)

        rows = q.order_by(SagaDigest.from_turn.asc()).limit(lim).all()
        return [_saga_digest_to_dict(r) for r in rows]
    finally:
        db.close()


def get_saga_digest_by_id(saga_digest_id: int) -> dict | None:
    """id로 saga digest 1건 조회."""
    db: Session = SessionLocal()
    try:
        row = db.query(SagaDigest).filter(SagaDigest.id == saga_digest_id).first()
        return _saga_digest_to_dict(row) if row else None
    finally:
        db.close()


def _saga_digest_to_dict(row: SagaDigest) -> dict:
    payload = {
        "id": row.id,
        "chat_session_id": row.chat_session_id,
        "from_turn": row.from_turn,
        "to_turn": row.to_turn,
        "era_label": row.era_label,
        "saga_summary": row.saga_summary,
        "persistent_facts_json": row.persistent_facts_json,
        "never_drop_candidates_json": row.never_drop_candidates_json,
        "resume_pack_text": row.resume_pack_text,
        "embedding_vector": row.embedding_vector,
        "embedding_model": row.embedding_model,
        "created_at": to_kst(row.created_at),
    }
    payload.update(
        _build_dense_summary_role_split_ds1h(payload, tier="saga")
    )
    payload.update(
        _build_dense_summary_source_anchor_ds1f(
            source_record_id=row.id,
            source_record_type="saga_digest",
            from_turn=row.from_turn,
            to_turn=row.to_turn,
        )
    )
    return payload


# ------------------------------------------------------------------ #
#  Step 9 P-1b — Chapter Summary Prompt + JSON Contract
# ------------------------------------------------------------------ #

_CHAPTER_SYSTEM_PROMPT = """\
You are a chapter summarizer for a roleplay memory system.
Given a larger segment of conversation and the related episode summaries,
produce a structured JSON summary for one chapter of the story.

[Core Principles]
1. Summarize a medium-range chapter, not single turns.
2. Preserve continuity-critical changes: relationships, world state, unresolved promises, and payoff candidates.
3. Keep the output compact enough to be reused in long-horizon memory packets.
4. Do NOT fabricate events or motivations not present in the input.
5. Write in the SAME LANGUAGE as the input conversation.

[Required Fields]
- chapter_title: short title for the chapter
- summary_text: 3-6 sentence chapter summary
- open_loops: unresolved questions, promises, risks, emotional debts still active after this chapter
- relationship_changes: notable relationship or trust changes introduced/reinforced in this chapter
- world_changes: world facts, location shifts, faction/system changes that remain relevant later
- callback_candidates: older or current elements likely worth revisiting later as payoff or callback
- resume_text: 1-2 sentence compact reentry summary for long-gap resume

[Output Format]
Return ONLY a single valid JSON object:
```json
{
  "chapter_title": "short title",
  "summary_text": "3-6 sentence summary",
  "open_loops": ["item1", "item2"],
  "relationship_changes": ["item1", "item2"],
  "world_changes": ["item1", "item2"],
  "callback_candidates": ["item1", "item2"],
  "resume_text": "1-2 sentence compact resume"
}
```
Do NOT add commentary outside the JSON."""


_ARC_SYSTEM_PROMPT = """\
You are an arc summarizer for a roleplay memory system.
Given a larger story span and the related chapter summaries,
produce a structured JSON summary for one long-running story arc.

[Core Principles]
1. Summarize a long-range arc, not a single chapter.
2. Preserve continuity-critical direction: the arc's core conflict, turning points, unresolved promises, resolved payoffs, and future payoff candidates.
3. Keep the output compact enough to be reused in ultra-long-horizon memory packets.
4. Do NOT fabricate events, motives, or payoffs not present in the input.
5. Write in the SAME LANGUAGE as the input conversation.

[Required Fields]
- arc_name: short name for the arc
- arc_status: one of active, paused, resolved
- core_conflict: the central unresolved or defining conflict of this arc
- key_turning_points: major beats that changed the direction of the arc
- irreversible_turns: irreversible turns (deaths, vows, broken alliances, role-locks, world-rule commitments) that should not be silently rewritten
- active_promises: promises, debts, unresolved goals, or open obligations still active in this arc
- unresolved_debts: emotional debts, risks, unanswered questions, or unresolved burdens still carried by this arc
- resolved_payoffs: promises or long-running setups that this arc already paid off
- callback_candidates: concrete callback hooks from this arc worth resurfacing later
- callback_debts: callback hooks that are still unpaid and should be tracked as debt before payoff
- relationship_pivots: long-term relationship pivots that materially changed trust, loyalty, hostility, or dependency
- future_payoff_candidates: high-value callback or payoff candidates worth revisiting later
- arc_resume_text: 1-3 sentence compact reentry summary for long-gap resume

[Output Format]
Return ONLY a single valid JSON object:
```json
{
    "arc_name": "short name",
    "arc_status": "active",
    "core_conflict": "central conflict summary",
    "key_turning_points": ["item1", "item2"],
    "irreversible_turns": ["item1", "item2"],
    "active_promises": ["item1", "item2"],
    "unresolved_debts": ["item1", "item2"],
    "resolved_payoffs": ["item1", "item2"],
    "callback_candidates": ["item1", "item2"],
    "callback_debts": ["item1", "item2"],
    "relationship_pivots": ["item1", "item2"],
    "future_payoff_candidates": ["item1", "item2"],
    "arc_resume_text": "1-3 sentence compact resume"
}
```
Do NOT add commentary outside the JSON."""


_SAGA_SYSTEM_PROMPT = """\
You are a saga digest summarizer for a roleplay memory system.
Given a very long story span and the related arc summaries,
produce a structured JSON digest for one era-level saga segment.

[Core Principles]
1. Summarize an era-level saga span, not a single arc beat.
2. Preserve continuity-critical long-horizon material: persistent facts, never-drop callback material, and a compact resume pack.
3. Keep the output compact enough to anchor 1M+ and 2M+ resume packets.
4. Do NOT fabricate eras, facts, or payoffs not present in the input.
5. Write in the SAME LANGUAGE as the input conversation.

[Required Fields]
- era_label: short label for this era or saga phase
- saga_summary: 4-8 sentence long-horizon summary of the era
- persistent_facts: facts, alliances, wounds, promises, or world conditions that still matter after this era
- never_drop_candidates: continuity-critical hooks or facts that should almost never be omitted in long-gap resume
- resume_pack_text: 2-4 sentence compact reentry summary for ultra-long resume

[Output Format]
Return ONLY a single valid JSON object:
```json
{
    "era_label": "short era label",
    "saga_summary": "4-8 sentence saga summary",
    "persistent_facts": ["item1", "item2"],
    "never_drop_candidates": ["item1", "item2"],
    "resume_pack_text": "2-4 sentence compact resume"
}
```
Do NOT add commentary outside the JSON."""


def _normalize_chapter_list_field(raw: object, *, max_items: int = 8, max_item_len: int = 240) -> list[str]:
    """chapter contract용 문자열 목록 필드를 보수적으로 정규화한다."""
    if raw is None:
        return []

    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = " ".join(str(item or "").split()).strip()
        if not text:
            continue
        if len(text) > max_item_len:
            text = text[: max_item_len - 3] + "..."
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
        if len(normalized) >= max_items:
            break
    return normalized


def _normalize_chapter_summary_payload(parsed: dict) -> dict | None:
    """P-1b: chapter summary JSON payload를 저장 준비 형태로 정규화한다.

    반환 shape는 create_chapter_summary에 바로 전달 가능한 필드 subset이다.
    필수: summary_text.
    선택: chapter_title, resume_text, *_json list fields.
    """
    if not isinstance(parsed, dict):
        return None

    summary_text = " ".join(str(parsed.get("summary_text") or "").split()).strip()
    if not summary_text:
        return None
    if len(summary_text) > 1800:
        summary_text = summary_text[:1797] + "..."

    chapter_title = " ".join(str(parsed.get("chapter_title") or "").split()).strip() or None
    if chapter_title and len(chapter_title) > 255:
        chapter_title = chapter_title[:252] + "..."

    open_loops = _normalize_chapter_list_field(parsed.get("open_loops"))
    relationship_changes = _normalize_chapter_list_field(parsed.get("relationship_changes"))
    world_changes = _normalize_chapter_list_field(parsed.get("world_changes"))
    callback_candidates = _normalize_chapter_list_field(parsed.get("callback_candidates"))

    resume_text = " ".join(str(parsed.get("resume_text") or "").split()).strip()
    if not resume_text:
        resume_text = summary_text[:280]
    if len(resume_text) > 500:
        resume_text = resume_text[:497] + "..."

    return {
        "chapter_title": chapter_title,
        "summary_text": summary_text,
        "open_loops_json": json.dumps(open_loops, ensure_ascii=False) if open_loops else None,
        "relationship_changes_json": json.dumps(relationship_changes, ensure_ascii=False) if relationship_changes else None,
        "world_changes_json": json.dumps(world_changes, ensure_ascii=False) if world_changes else None,
        "callback_candidates_json": json.dumps(callback_candidates, ensure_ascii=False) if callback_candidates else None,
        "resume_text": resume_text or None,
    }


def _parse_chapter_json(raw: str) -> dict | None:
    """LLM 응답에서 chapter JSON을 파싱하고 P-1b contract로 정규화한다."""
    code_block = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = []
    if code_block:
        candidates.append(code_block.group(1).strip())
    candidates.append((raw or "").strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        normalized = _normalize_chapter_summary_payload(parsed)
        if normalized is not None:
            return normalized
    return None


def _normalize_arc_summary_payload(parsed: dict) -> dict | None:
    """P-3b: arc summary JSON payload를 저장 준비 형태로 정규화한다.

    반환 shape는 create_arc_summary에 바로 전달 가능한 필드 subset이다.
    필수: core_conflict.
    선택: arc_name, arc_status, arc_resume_text, *_json list fields.
    """
    if not isinstance(parsed, dict):
        return None

    core_conflict = " ".join(str(parsed.get("core_conflict") or "").split()).strip()
    if not core_conflict:
        return None
    if len(core_conflict) > 1200:
        core_conflict = core_conflict[:1197] + "..."

    arc_name = " ".join(str(parsed.get("arc_name") or "").split()).strip() or None
    if arc_name and len(arc_name) > 255:
        arc_name = arc_name[:252] + "..."

    arc_status = " ".join(str(parsed.get("arc_status") or "active").split()).strip().lower()
    if arc_status not in {"active", "paused", "resolved"}:
        arc_status = "active"

    key_turning_points = _normalize_chapter_list_field(parsed.get("key_turning_points"), max_items=10)
    irreversible_turns = _normalize_chapter_list_field(parsed.get("irreversible_turns"), max_items=10)
    active_promises = _normalize_chapter_list_field(parsed.get("active_promises"), max_items=10)
    unresolved_debts = _normalize_chapter_list_field(parsed.get("unresolved_debts"), max_items=10)
    resolved_payoffs = _normalize_chapter_list_field(parsed.get("resolved_payoffs"), max_items=10)
    callback_candidates = _normalize_chapter_list_field(parsed.get("callback_candidates"), max_items=10)
    callback_debts = _normalize_chapter_list_field(parsed.get("callback_debts"), max_items=10)
    relationship_pivots = _normalize_chapter_list_field(parsed.get("relationship_pivots"), max_items=10)
    future_payoff_candidates = _normalize_chapter_list_field(parsed.get("future_payoff_candidates"), max_items=10)

    arc_resume_text = " ".join(str(parsed.get("arc_resume_text") or "").split()).strip()
    if not arc_resume_text:
        arc_resume_text = core_conflict[:320]
    if len(arc_resume_text) > 600:
        arc_resume_text = arc_resume_text[:597] + "..."

    return {
        "arc_name": arc_name,
        "arc_status": arc_status,
        "core_conflict": core_conflict,
        "key_turning_points_json": json.dumps(key_turning_points, ensure_ascii=False) if key_turning_points else None,
        "irreversible_turns_json": json.dumps(irreversible_turns, ensure_ascii=False) if irreversible_turns else None,
        "active_promises_json": json.dumps(active_promises, ensure_ascii=False) if active_promises else None,
        "unresolved_debts_json": json.dumps(unresolved_debts, ensure_ascii=False) if unresolved_debts else None,
        "resolved_payoffs_json": json.dumps(resolved_payoffs, ensure_ascii=False) if resolved_payoffs else None,
        "callback_candidates_json": json.dumps(callback_candidates, ensure_ascii=False) if callback_candidates else None,
        "callback_debts_json": json.dumps(callback_debts, ensure_ascii=False) if callback_debts else None,
        "relationship_pivots_json": json.dumps(relationship_pivots, ensure_ascii=False) if relationship_pivots else None,
        "future_payoff_candidates_json": json.dumps(future_payoff_candidates, ensure_ascii=False) if future_payoff_candidates else None,
        "arc_resume_text": arc_resume_text or None,
    }


def _parse_arc_json(raw: str) -> dict | None:
    """LLM 응답에서 arc JSON을 파싱하고 P-3b contract로 정규화한다."""
    code_block = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = []
    if code_block:
        candidates.append(code_block.group(1).strip())
    candidates.append((raw or "").strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        normalized = _normalize_arc_summary_payload(parsed)
        if normalized is not None:
            return normalized
    return None


def _normalize_saga_digest_payload(parsed: dict) -> dict | None:
    """P-4b: saga digest JSON payload를 저장 준비 형태로 정규화한다.

    반환 shape는 create_saga_digest에 바로 전달 가능한 필드 subset이다.
    필수: saga_summary.
    선택: era_label, resume_pack_text, *_json list fields.
    """
    if not isinstance(parsed, dict):
        return None

    saga_summary = " ".join(str(parsed.get("saga_summary") or "").split()).strip()
    if not saga_summary:
        return None
    if len(saga_summary) > 2200:
        saga_summary = saga_summary[:2197] + "..."

    era_label = " ".join(str(parsed.get("era_label") or "").split()).strip() or None
    if era_label and len(era_label) > 255:
        era_label = era_label[:252] + "..."

    persistent_facts = _normalize_chapter_list_field(parsed.get("persistent_facts"), max_items=12, max_item_len=280)
    never_drop_candidates = _normalize_chapter_list_field(parsed.get("never_drop_candidates"), max_items=12, max_item_len=280)

    resume_pack_text = " ".join(str(parsed.get("resume_pack_text") or "").split()).strip()
    if not resume_pack_text:
        resume_pack_text = saga_summary[:420]
    if len(resume_pack_text) > 800:
        resume_pack_text = resume_pack_text[:797] + "..."

    return {
        "era_label": era_label,
        "saga_summary": saga_summary,
        "persistent_facts_json": json.dumps(persistent_facts, ensure_ascii=False) if persistent_facts else None,
        "never_drop_candidates_json": json.dumps(never_drop_candidates, ensure_ascii=False) if never_drop_candidates else None,
        "resume_pack_text": resume_pack_text or None,
    }


def _parse_saga_json(raw: str) -> dict | None:
    """LLM 응답에서 saga JSON을 파싱하고 P-4b contract로 정규화한다."""
    code_block = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = []
    if code_block:
        candidates.append(code_block.group(1).strip())
    candidates.append((raw or "").strip())

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        normalized = _normalize_saga_digest_payload(parsed)
        if normalized is not None:
            return normalized
    return None


def _parse_json_string_list(raw: str | None) -> list[str]:
    """JSON 문자열 목록 필드를 보수적으로 파싱한다."""
    if not raw or not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []

    normalized: list[str] = []
    for item in parsed:
        text = " ".join(str(item or "").split()).strip()
        if text:
            normalized.append(text)
    return normalized


def _check_arc_exists(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
) -> bool:
    """동일 session + 동일 turn range의 arc가 이미 존재하는지 확인한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return False

    db: Session = SessionLocal()
    try:
        count = (
            db.query(ArcSummary)
            .filter(
                ArcSummary.chat_session_id == sid,
                ArcSummary.from_turn == from_turn,
                ArcSummary.to_turn == to_turn,
            )
            .count()
        )
        return count > 0
    finally:
        db.close()


def _next_arc_index(*, chat_session_id: str) -> int:
    """세션 내 다음 arc_index를 계산한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return 1

    db: Session = SessionLocal()
    try:
        max_index = (
            db.query(func.max(ArcSummary.arc_index))
            .filter(ArcSummary.chat_session_id == sid)
            .scalar()
        )
        if isinstance(max_index, int) and max_index > 0:
            return max_index + 1

        count = (
            db.query(ArcSummary)
            .filter(ArcSummary.chat_session_id == sid)
            .count()
        )
        return count + 1
    finally:
        db.close()


def _resolve_arc_generation_status(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    proposed_status: str,
    active_promises_json: str | None,
    unresolved_debts_json: str | None,
) -> dict:
    """P-3c: arc lifecycle 규칙으로 generation status를 정규화한다."""
    sid = _safe_session_id(chat_session_id)
    final_status = proposed_status if proposed_status in {"active", "paused", "resolved"} else "active"
    status_reason = "llm_status"
    active_promises = _parse_json_string_list(active_promises_json)
    unresolved_debts = _parse_json_string_list(unresolved_debts_json)

    if final_status == "resolved" and (active_promises or unresolved_debts):
        final_status = "active"
        status_reason = "unresolved_material_override"

    newer_active_exists = False
    if sid and final_status == "active":
        db: Session = SessionLocal()
        try:
            newer_active_exists = (
                db.query(ArcSummary.id)
                .filter(ArcSummary.chat_session_id == sid)
                .filter(ArcSummary.arc_status == "active")
                .filter(ArcSummary.to_turn > to_turn)
                .first()
            ) is not None
        finally:
            db.close()

    if final_status == "active" and newer_active_exists:
        final_status = "paused"
        status_reason = "newer_active_arc_exists"

    return {
        "final_status": final_status,
        "status_reason": status_reason,
        "coerced": final_status != proposed_status,
    }


def _pause_older_active_arcs(
    *,
    chat_session_id: str,
    current_arc_id: int,
    current_to_turn: int,
) -> int:
    """P-3c: 더 최신 active arc가 저장되면 이전 active arc를 paused로 내린다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return 0

    db: Session = SessionLocal()
    try:
        updated = (
            db.query(ArcSummary)
            .filter(ArcSummary.chat_session_id == sid)
            .filter(ArcSummary.id != current_arc_id)
            .filter(ArcSummary.arc_status == "active")
            .filter(ArcSummary.to_turn < current_to_turn)
            .update({ArcSummary.arc_status: "paused"}, synchronize_session=False)
        )
        db.commit()
        return int(updated or 0)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _collect_arc_input(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    max_chapter_chars: int = 9000,
    include_direct_evidence: bool = True,
    max_direct_evidence_chars: int = 2000,
) -> dict:
    """지정 구간의 arc 생성을 위한 chapter summary 입력을 수집한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return {
            "chapter_lines": [],
            "direct_evidence_lines": [],
            "chapter_count": 0,
            "direct_evidence_count": 0,
        }

    chapters = get_chapter_summaries_for_session(
        chat_session_id=sid,
        limit=200,
        from_turn=from_turn,
        to_turn=to_turn,
    )

    chapter_lines = []
    total_chars = 0
    for chapter in chapters:
        summary_text = " ".join(str(chapter.get("summary_text") or "").split()).strip()
        resume_text = " ".join(str(chapter.get("resume_text") or "").split()).strip()

        open_loops = _json_list_preview_q2b(chapter.get("open_loops_json"), max_items=4)
        relationship_changes = _json_list_preview_q2b(chapter.get("relationship_changes_json"), max_items=4)
        world_changes = _json_list_preview_q2b(chapter.get("world_changes_json"), max_items=4)
        callback_candidates = _json_list_preview_q2b(chapter.get("callback_candidates_json"), max_items=4)

        anchor_parts: list[str] = []
        if open_loops:
            anchor_parts.append("open loops: " + ", ".join(open_loops))
        if relationship_changes:
            anchor_parts.append("relationships: " + ", ".join(relationship_changes))
        if world_changes:
            anchor_parts.append("world: " + ", ".join(world_changes))
        if callback_candidates:
            anchor_parts.append("callbacks: " + ", ".join(callback_candidates))
        if resume_text:
            anchor_parts.append(f"resume: {resume_text}")

        if not anchor_parts and not summary_text:
            continue

        title = " ".join(str(chapter.get("chapter_title") or "").split()).strip()
        heading = title or f"chapter {chapter.get('from_turn')}-{chapter.get('to_turn')}"
        line = f"[{heading} | turn {chapter.get('from_turn')}-{chapter.get('to_turn')}]"

        if anchor_parts:
            line += " " + " | ".join(anchor_parts)
            if summary_text:
                line += f" | summary: {summary_text}"
        else:
            line += f" summary: {summary_text}"

        if total_chars + len(line) > max_chapter_chars:
            remaining = max_chapter_chars - total_chars
            if remaining > 50:
                chapter_lines.append(line[:remaining] + "...(절단됨)")
            break

        chapter_lines.append(line)
        total_chars += len(line)

    direct_evidence_lines = []
    direct_evidence_count = 0
    if include_direct_evidence:
        db: Session = SessionLocal()
        try:
            evidence_rows = (
                db.query(DirectEvidenceRecord)
                .filter(
                    DirectEvidenceRecord.chat_session_id == sid,
                    DirectEvidenceRecord.source_turn_end >= from_turn,
                    DirectEvidenceRecord.source_turn_start <= to_turn,
                )
                .order_by(DirectEvidenceRecord.source_turn_start.asc(), DirectEvidenceRecord.id.asc())
                .all()
            )

            evidence_chars = 0
            for evidence_row in evidence_rows:
                if not is_gate_approved_current_truth(
                    archive_state=evidence_row.archive_state,
                    capture_verification=evidence_row.capture_verification,
                    repair_needed=evidence_row.repair_needed,
                    committed_gate=evidence_row.committed_gate,
                    tombstoned=evidence_row.tombstoned,
                    superseded_by_id=evidence_row.superseded_by_id,
                ):
                    continue

                if not _is_direct_evidence_retained_for_consumption(
                    evidence_row,
                    latest_turn_index=to_turn,
                ):
                    continue

                normalized_committed_gate = resolve_committed_gate(
                    evidence_row.capture_verification,
                    evidence_row.repair_needed,
                    evidence_row.committed_gate,
                )

                evidence_text = " ".join(str(evidence_row.evidence_text or "").split()).strip()
                if not evidence_text:
                    continue

                source_turn_start = evidence_row.source_turn_start if evidence_row.source_turn_start is not None else "?"
                source_turn_end = evidence_row.source_turn_end if evidence_row.source_turn_end is not None else "?"
                line = f"[evidence turn {source_turn_start}-{source_turn_end}][gate:{normalized_committed_gate}] {evidence_text}"
                if evidence_chars + len(line) > max_direct_evidence_chars:
                    break

                direct_evidence_lines.append(line)
                evidence_chars += len(line)
                direct_evidence_count += 1
        finally:
            db.close()

    return {
        "chapter_lines": chapter_lines,
        "direct_evidence_lines": direct_evidence_lines,
        "chapter_count": len(chapter_lines),
        "direct_evidence_count": direct_evidence_count,
    }


def _call_arc_llm(user_prompt: str) -> str | None:
    """Critic LLM 설정을 재사용하여 arc 요약 LLM을 호출한다."""
    endpoint = settings.PROJECT_CRITIC_ENDPOINT
    api_key = settings.PROJECT_CRITIC_API_KEY
    model = settings.PROJECT_CRITIC_MODEL
    timeout = getattr(settings, "CRITIC_TIMEOUT", 90) or 90

    if not endpoint or not api_key or not model:
        logger.warning("Arc LLM 설정 없음 (Critic 설정 재사용): 요약 생성 건너뜀")
        return None

    try:
        url = endpoint.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _ARC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
            return "\n".join(parts).strip() if parts else None
        return None
    except Exception as exc:
        logger.warning("Arc LLM 호출 실패: %s", exc)
        return None


def generate_arc_summary(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    force: bool = False,
) -> dict:
    """지정된 turn range의 arc summary를 생성하고 저장한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return {"status": "error", "reason": "invalid chat_session_id"}

    if from_turn < 0 or to_turn < from_turn:
        return {"status": "error", "reason": f"invalid turn range: {from_turn}-{to_turn}"}

    if not force and _check_arc_exists(chat_session_id=sid, from_turn=from_turn, to_turn=to_turn):
        return {
            "status": "skipped",
            "reason": "already_exists",
            "from_turn": from_turn,
            "to_turn": to_turn,
        }

    arc_input = _collect_arc_input(
        chat_session_id=sid,
        from_turn=from_turn,
        to_turn=to_turn,
    )

    if not arc_input["chapter_lines"]:
        return {
            "status": "skipped",
            "reason": "no_chapter_summaries",
            "from_turn": from_turn,
            "to_turn": to_turn,
        }

    turn_span = (to_turn - from_turn) + 1
    warnings: list[str] = []
    if not (3 <= arc_input["chapter_count"] <= 6):
        warnings.append("chapter_count_outside_recommended_window")
    if not (200 <= turn_span <= 400):
        warnings.append("turn_span_outside_recommended_window")

    user_prompt_parts = [
        f"Arc range: turn {from_turn} to {to_turn}",
    ]
    if arc_input["direct_evidence_lines"]:
        user_prompt_parts.extend([
            "",
            "=== Verified Direct Evidence (gate-approved) ===",
            *arc_input["direct_evidence_lines"],
        ])
    user_prompt_parts.extend([
        "",
        "=== Chapter Summaries ===",
        *arc_input["chapter_lines"],
    ])
    user_prompt = "\n".join(user_prompt_parts)

    raw_response = _call_arc_llm(user_prompt)
    if not raw_response:
        return {
            "status": "error",
            "reason": "llm_call_failed",
            "from_turn": from_turn,
            "to_turn": to_turn,
            "warnings": warnings,
            "input_stats": {
                "chapter_count": arc_input["chapter_count"],
                "direct_evidence_count": arc_input["direct_evidence_count"],
                "direct_evidence_first_applied": bool(arc_input["direct_evidence_lines"]),
                "turn_span": turn_span,
                "chapter_dense_summary_injection_policy_version": "ds1b.v1",
                "arc_dense_summary_policy_version": "ds1c.v1",
            },
        }

    parsed = _parse_arc_json(raw_response)
    if not parsed:
        return {
            "status": "error",
            "reason": "json_parse_failed",
            "from_turn": from_turn,
            "to_turn": to_turn,
            "warnings": warnings,
            "raw_preview": raw_response[:300],
        }

    lifecycle = _resolve_arc_generation_status(
        chat_session_id=sid,
        from_turn=from_turn,
        to_turn=to_turn,
        proposed_status=parsed.get("arc_status") or "active",
        active_promises_json=parsed.get("active_promises_json"),
        unresolved_debts_json=parsed.get("unresolved_debts_json"),
    )
    if lifecycle.get("coerced"):
        warnings.append(f"arc_status_coerced:{lifecycle.get('status_reason')}")

    embedding_vector_json = None
    embedding_model_name = None
    try:
        archive_bridge = ArchiveBridge()
        embed_text = parsed.get("arc_resume_text") or parsed["core_conflict"]
        vec = archive_bridge._get_embedding(embed_text)
        if vec:
            embedding_vector_json = json.dumps(vec)
            embedding_model_name = settings.PROJECT_EMBEDDING_MODEL or None
    except Exception as exc:
        logger.warning("Arc embedding 생성 실패 (계속 진행): %s", exc)

    try:
        saved = create_arc_summary(
            chat_session_id=sid,
            from_turn=from_turn,
            to_turn=to_turn,
            arc_index=_next_arc_index(chat_session_id=sid),
            arc_name=parsed.get("arc_name"),
            arc_status=lifecycle["final_status"],
            core_conflict=parsed["core_conflict"],
            key_turning_points_json=parsed.get("key_turning_points_json"),
            active_promises_json=parsed.get("active_promises_json"),
            unresolved_debts_json=parsed.get("unresolved_debts_json"),
            resolved_payoffs_json=parsed.get("resolved_payoffs_json"),
            callback_candidates_json=parsed.get("callback_candidates_json"),
            irreversible_turns_json=parsed.get("irreversible_turns_json"),
            callback_debts_json=parsed.get("callback_debts_json"),
            relationship_pivots_json=parsed.get("relationship_pivots_json"),
            future_payoff_candidates_json=parsed.get("future_payoff_candidates_json"),
            arc_resume_text=parsed.get("arc_resume_text"),
            embedding_vector=embedding_vector_json,
            embedding_model=embedding_model_name,
        )
        paused_other_active_count = 0
        if saved.get("arc_status") == "active":
            paused_other_active_count = _pause_older_active_arcs(
                chat_session_id=sid,
                current_arc_id=int(saved.get("id") or 0),
                current_to_turn=to_turn,
            )
        return {
            "status": "generated",
            "arc": saved,
            "warnings": warnings,
            "lifecycle": {
                **lifecycle,
                "paused_other_active_count": paused_other_active_count,
            },
            "input_stats": {
                "chapter_count": arc_input["chapter_count"],
                "direct_evidence_count": arc_input["direct_evidence_count"],
                "direct_evidence_first_applied": bool(arc_input["direct_evidence_lines"]),
                "turn_span": turn_span,
                "chapter_dense_summary_injection_policy_version": "ds1b.v1",
                "arc_dense_summary_policy_version": "ds1c.v1",
            },
        }
    except Exception as exc:
        logger.error("generate_arc_summary 저장 실패: %s", exc)
        return {
            "status": "error",
            "reason": "db_insert_failed",
            "detail": str(exc),
            "from_turn": from_turn,
            "to_turn": to_turn,
        }


def _check_saga_exists(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
) -> bool:
    """동일 session + 동일 turn range의 saga digest가 이미 존재하는지 확인한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return False

    db: Session = SessionLocal()
    try:
        count = (
            db.query(SagaDigest)
            .filter(
                SagaDigest.chat_session_id == sid,
                SagaDigest.from_turn == from_turn,
                SagaDigest.to_turn == to_turn,
            )
            .count()
        )
        return count > 0
    finally:
        db.close()


def _collect_saga_input(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    max_arc_chars: int = 12000,
    include_direct_evidence: bool = True,
    max_direct_evidence_chars: int = 2500,
) -> dict:
    """지정 구간의 saga digest 생성을 위한 arc summary 입력을 수집한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return {
            "arc_lines": [],
            "direct_evidence_lines": [],
            "arc_count": 0,
            "direct_evidence_count": 0,
        }

    arcs = get_arc_summaries_for_session(
        chat_session_id=sid,
        limit=100,
        from_turn=from_turn,
        to_turn=to_turn,
    )

    arc_lines = []
    total_chars = 0
    for arc in arcs:
        conflict = " ".join(str(arc.get("core_conflict") or "").split()).strip()
        resume_text = " ".join(str(arc.get("arc_resume_text") or "").split()).strip()

        irreversible_turns = _parse_json_string_list(arc.get("irreversible_turns_json"))
        callback_debts = _parse_json_string_list(arc.get("callback_debts_json"))
        relationship_pivots = _parse_json_string_list(arc.get("relationship_pivots_json"))
        active_promises = _parse_json_string_list(arc.get("active_promises_json"))
        unresolved_debts = _parse_json_string_list(arc.get("unresolved_debts_json"))
        callback_candidates = _parse_json_string_list(arc.get("callback_candidates_json"))

        anchor_parts: list[str] = []
        if irreversible_turns:
            anchor_parts.append("irreversible: " + ", ".join(irreversible_turns[:4]))
        if callback_debts:
            anchor_parts.append("callback debt: " + ", ".join(callback_debts[:4]))
        if relationship_pivots:
            anchor_parts.append("relationship pivots: " + ", ".join(relationship_pivots[:4]))
        if active_promises:
            anchor_parts.append("promises: " + ", ".join(active_promises[:4]))
        if unresolved_debts:
            anchor_parts.append("debts: " + ", ".join(unresolved_debts[:4]))
        if callback_candidates:
            anchor_parts.append("callbacks: " + ", ".join(callback_candidates[:4]))
        if resume_text:
            anchor_parts.append(f"resume: {resume_text}")

        if not conflict and not anchor_parts:
            continue

        name = " ".join(str(arc.get("arc_name") or "").split()).strip()
        status = " ".join(str(arc.get("arc_status") or "active").split()).strip().lower() or "active"
        heading = name or f"arc {arc.get('from_turn')}-{arc.get('to_turn')}"
        line = f"[{heading} | {status} | turn {arc.get('from_turn')}-{arc.get('to_turn')}]"
        if anchor_parts:
            line += " " + " | ".join(anchor_parts)
            if conflict:
                line += f" | core: {conflict}"
        else:
            line += f" core: {conflict}"

        if total_chars + len(line) > max_arc_chars:
            remaining = max_arc_chars - total_chars
            if remaining > 50:
                arc_lines.append(line[:remaining] + "...(절단됨)")
            break

        arc_lines.append(line)
        total_chars += len(line)

    direct_evidence_lines = []
    direct_evidence_count = 0
    if include_direct_evidence:
        db: Session = SessionLocal()
        try:
            evidence_rows = (
                db.query(DirectEvidenceRecord)
                .filter(
                    DirectEvidenceRecord.chat_session_id == sid,
                    DirectEvidenceRecord.source_turn_end >= from_turn,
                    DirectEvidenceRecord.source_turn_start <= to_turn,
                )
                .order_by(DirectEvidenceRecord.source_turn_start.asc(), DirectEvidenceRecord.id.asc())
                .all()
            )

            evidence_chars = 0
            for evidence_row in evidence_rows:
                if not is_gate_approved_current_truth(
                    archive_state=evidence_row.archive_state,
                    capture_verification=evidence_row.capture_verification,
                    repair_needed=evidence_row.repair_needed,
                    committed_gate=evidence_row.committed_gate,
                    tombstoned=evidence_row.tombstoned,
                    superseded_by_id=evidence_row.superseded_by_id,
                ):
                    continue

                if not _is_direct_evidence_retained_for_consumption(
                    evidence_row,
                    latest_turn_index=to_turn,
                ):
                    continue

                normalized_committed_gate = resolve_committed_gate(
                    evidence_row.capture_verification,
                    evidence_row.repair_needed,
                    evidence_row.committed_gate,
                )

                evidence_text = " ".join(str(evidence_row.evidence_text or "").split()).strip()
                if not evidence_text:
                    continue

                source_turn_start = evidence_row.source_turn_start if evidence_row.source_turn_start is not None else "?"
                source_turn_end = evidence_row.source_turn_end if evidence_row.source_turn_end is not None else "?"
                line = f"[evidence turn {source_turn_start}-{source_turn_end}][gate:{normalized_committed_gate}] {evidence_text}"
                if evidence_chars + len(line) > max_direct_evidence_chars:
                    break

                direct_evidence_lines.append(line)
                evidence_chars += len(line)
                direct_evidence_count += 1
        finally:
            db.close()

    return {
        "arc_lines": arc_lines,
        "direct_evidence_lines": direct_evidence_lines,
        "arc_count": len(arc_lines),
        "direct_evidence_count": direct_evidence_count,
    }


def _call_saga_llm(user_prompt: str) -> str | None:
    """Critic LLM 설정을 재사용하여 saga digest LLM을 호출한다."""
    endpoint = settings.PROJECT_CRITIC_ENDPOINT
    api_key = settings.PROJECT_CRITIC_API_KEY
    model = settings.PROJECT_CRITIC_MODEL
    timeout = getattr(settings, "CRITIC_TIMEOUT", 90) or 90

    if not endpoint or not api_key or not model:
        logger.warning("Saga LLM 설정 없음 (Critic 설정 재사용): 요약 생성 건너뜀")
        return None

    try:
        url = endpoint.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SAGA_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
            return "\n".join(parts).strip() if parts else None
        return None
    except Exception as exc:
        logger.warning("Saga LLM 호출 실패: %s", exc)
        return None


def generate_saga_digest(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    force: bool = False,
) -> dict:
    """지정된 turn range의 saga digest를 생성하고 저장한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return {"status": "error", "reason": "invalid chat_session_id"}

    if from_turn < 0 or to_turn < from_turn:
        return {"status": "error", "reason": f"invalid turn range: {from_turn}-{to_turn}"}

    if not force and _check_saga_exists(chat_session_id=sid, from_turn=from_turn, to_turn=to_turn):
        return {
            "status": "skipped",
            "reason": "already_exists",
            "from_turn": from_turn,
            "to_turn": to_turn,
        }

    saga_input = _collect_saga_input(
        chat_session_id=sid,
        from_turn=from_turn,
        to_turn=to_turn,
    )

    if not saga_input["arc_lines"]:
        return {
            "status": "skipped",
            "reason": "no_arc_summaries",
            "from_turn": from_turn,
            "to_turn": to_turn,
        }

    turn_span = (to_turn - from_turn) + 1
    warnings: list[str] = []
    if not (2 <= saga_input["arc_count"] <= 6):
        warnings.append("arc_count_outside_recommended_window")
    if not (400 <= turn_span <= 2000):
        warnings.append("turn_span_outside_recommended_window")

    user_prompt_parts = [
        f"Saga range: turn {from_turn} to {to_turn}",
    ]
    if saga_input["direct_evidence_lines"]:
        user_prompt_parts.extend([
            "",
            "=== Verified Direct Evidence (gate-approved) ===",
            *saga_input["direct_evidence_lines"],
        ])
    user_prompt_parts.extend([
        "",
        "=== Arc Summaries ===",
        *saga_input["arc_lines"],
    ])
    user_prompt = "\n".join(user_prompt_parts)

    raw_response = _call_saga_llm(user_prompt)
    if not raw_response:
        return {
            "status": "error",
            "reason": "llm_call_failed",
            "from_turn": from_turn,
            "to_turn": to_turn,
            "warnings": warnings,
            "input_stats": {
                "arc_count": saga_input["arc_count"],
                "direct_evidence_count": saga_input["direct_evidence_count"],
                "direct_evidence_first_applied": bool(saga_input["direct_evidence_lines"]),
                "turn_span": turn_span,
            },
        }

    parsed = _parse_saga_json(raw_response)
    if not parsed:
        return {
            "status": "error",
            "reason": "json_parse_failed",
            "from_turn": from_turn,
            "to_turn": to_turn,
            "warnings": warnings,
            "raw_preview": raw_response[:300],
        }

    if not parsed.get("era_label"):
        parsed["era_label"] = f"Era {from_turn}-{to_turn}"

    embedding_vector_json = None
    embedding_model_name = None
    try:
        archive_bridge = ArchiveBridge()
        embed_text = parsed.get("resume_pack_text") or parsed["saga_summary"]
        vec = archive_bridge._get_embedding(embed_text)
        if vec:
            embedding_vector_json = json.dumps(vec)
            embedding_model_name = settings.PROJECT_EMBEDDING_MODEL or None
    except Exception as exc:
        logger.warning("Saga embedding 생성 실패 (계속 진행): %s", exc)

    try:
        saved = create_saga_digest(
            chat_session_id=sid,
            from_turn=from_turn,
            to_turn=to_turn,
            era_label=parsed.get("era_label"),
            saga_summary=parsed["saga_summary"],
            persistent_facts_json=parsed.get("persistent_facts_json"),
            never_drop_candidates_json=parsed.get("never_drop_candidates_json"),
            resume_pack_text=parsed.get("resume_pack_text"),
            embedding_vector=embedding_vector_json,
            embedding_model=embedding_model_name,
        )
        return {
            "status": "generated",
            "saga_digest": saved,
            "warnings": warnings,
            "input_stats": {
                "arc_count": saga_input["arc_count"],
                "direct_evidence_count": saga_input["direct_evidence_count"],
                "direct_evidence_first_applied": bool(saga_input["direct_evidence_lines"]),
                "turn_span": turn_span,
            },
        }
    except Exception as exc:
        logger.error("generate_saga_digest 저장 실패: %s", exc)
        return {
            "status": "error",
            "reason": "db_insert_failed",
            "detail": str(exc),
            "from_turn": from_turn,
            "to_turn": to_turn,
        }


def _check_chapter_exists(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
) -> bool:
    """동일 session + 동일 turn range의 chapter가 이미 존재하는지 확인한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return False

    db: Session = SessionLocal()
    try:
        count = (
            db.query(ChapterSummary)
            .filter(
                ChapterSummary.chat_session_id == sid,
                ChapterSummary.from_turn == from_turn,
                ChapterSummary.to_turn == to_turn,
            )
            .count()
        )
        return count > 0
    finally:
        db.close()


def _check_chapter_interval(turn_index: int, interval: int) -> dict:
    """P-1c helper: chapter 생성 구간 경계 체크. 실제 생성 없이 boundary 여부만 반환."""
    interval = max(10, min(200, int(interval or 60)))
    info: dict = {"checked": True, "triggered": False, "range": None, "reason": ""}
    if turn_index < interval or turn_index % interval != 0:
        info["reason"] = "not_interval_boundary"
        return info
    to_turn = turn_index
    from_turn = turn_index - interval + 1
    info["range"] = [from_turn, to_turn]
    info["reason"] = "interval_boundary"
    return info


def preview_chapter_generation(
    *,
    chat_session_id: str,
    turn_index: int,
    interval: int = 60,
    force: bool = False,
    preview_limit: int = 8,
) -> dict:
    """Step 9 P-1e: chapter generation 전 read-only dry-run 정보를 반환한다.

    실제 LLM 호출이나 DB write 없이 아래를 점검한다.
    - interval boundary 여부
    - 동일 range chapter 중복 여부
    - overlapping episode summary 입력 개수/preview
    - 권장 window(episode 4~8개, turn span 40~80) 충족 여부
    """
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return {"status": "error", "reason": "invalid chat_session_id"}
    if turn_index < 0:
        return {"status": "error", "reason": "invalid turn_index"}

    normalized_interval = max(10, min(200, int(interval or 60)))
    preview_limit = max(1, min(int(preview_limit or 8), 12))
    interval_info = _check_chapter_interval(turn_index, normalized_interval)

    result = {
        "status": "ok",
        "mode": "dry_run",
        "chat_session_id": sid,
        "turn_index": turn_index,
        "interval": normalized_interval,
        "force": bool(force),
        "triggered": bool(interval_info.get("range")),
        "interval_check": interval_info,
        "candidate_range": None,
        "already_exists": False,
        "ready": False,
        "blocking_reasons": [],
        "warnings": [],
        "input_stats": {
            "episode_count": 0,
            "episode_count_recommended": False,
            "turn_span": None,
            "turn_span_recommended": False,
        },
        "episode_inputs": [],
    }

    if not interval_info.get("range"):
        result["blocking_reasons"].append(interval_info.get("reason") or "not_interval_boundary")
        return result

    from_turn, to_turn = interval_info["range"]
    turn_span = (to_turn - from_turn) + 1
    result["candidate_range"] = {
        "from_turn": from_turn,
        "to_turn": to_turn,
        "turn_span": turn_span,
    }

    already_exists = _check_chapter_exists(
        chat_session_id=sid,
        from_turn=from_turn,
        to_turn=to_turn,
    )
    result["already_exists"] = already_exists
    if already_exists and not force:
        result["blocking_reasons"].append("already_exists")

    episodes = get_episode_summaries_for_session(
        chat_session_id=sid,
        limit=preview_limit * 2,
        from_turn=from_turn,
        to_turn=to_turn,
    )
    episode_count = len(episodes)
    episode_count_recommended = 4 <= episode_count <= 8
    turn_span_recommended = 40 <= turn_span <= 80

    if episode_count == 0:
        result["blocking_reasons"].append("no_episode_summaries")
    elif not episode_count_recommended:
        result["warnings"].append("episode_count_outside_recommended_window")

    if not turn_span_recommended:
        result["warnings"].append("turn_span_outside_recommended_window")

    result["input_stats"] = {
        "episode_count": episode_count,
        "episode_count_recommended": episode_count_recommended,
        "turn_span": turn_span,
        "turn_span_recommended": turn_span_recommended,
    }
    result["episode_inputs"] = [
        {
            "id": ep.get("id"),
            "from_turn": ep.get("from_turn"),
            "to_turn": ep.get("to_turn"),
            "summary_preview": ((ep.get("summary_text") or "")[:160] + ("..." if len(ep.get("summary_text") or "") > 160 else "")),
        }
        for ep in episodes[:preview_limit]
    ]
    result["ready"] = len(result["blocking_reasons"]) == 0
    return result


def _next_chapter_index(*, chat_session_id: str) -> int:
    """세션 내 다음 chapter_index를 계산한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return 1

    db: Session = SessionLocal()
    try:
        max_index = (
            db.query(func.max(ChapterSummary.chapter_index))
            .filter(ChapterSummary.chat_session_id == sid)
            .scalar()
        )
        if isinstance(max_index, int) and max_index > 0:
            return max_index + 1

        count = (
            db.query(ChapterSummary)
            .filter(ChapterSummary.chat_session_id == sid)
            .count()
        )
        return count + 1
    finally:
        db.close()


def _collect_chapter_input(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    max_episode_chars: int = 6000,
    include_direct_evidence: bool = True,
    max_direct_evidence_chars: int = 2000,
) -> dict:
    """지정 구간의 chapter 생성을 위한 episode summary 입력을 수집한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return {
            "episode_lines": [],
            "direct_evidence_lines": [],
            "episode_count": 0,
            "direct_evidence_count": 0,
        }

    episodes = get_episode_summaries_for_session(
        chat_session_id=sid,
        limit=200,
        from_turn=from_turn,
        to_turn=to_turn,
    )

    episode_lines = []
    total_chars = 0
    for ep in episodes:
        summary_text = " ".join(str(ep.get("summary_text") or "").split()).strip()
        if not summary_text:
            continue

        line = f"[episode {ep.get('from_turn')}-{ep.get('to_turn')}] {summary_text}"

        entities = _json_list_preview_q2b(ep.get("key_entities"), max_items=5)
        if entities:
            entity_preview = ", ".join(entities)
            line += f" | entities: {entity_preview}"

        events = _json_list_preview_q2b(ep.get("key_events"), max_items=5)
        if events:
            event_preview = ", ".join(events)
            line += f" | events: {event_preview}"

        relationship_changes = _json_list_preview_q2b(ep.get("relationship_changes_json"), max_items=4)
        if relationship_changes:
            line += " | rel: " + ", ".join(relationship_changes)

        open_loops = _json_list_preview_q2b(ep.get("open_loops_json"), max_items=4)
        if open_loops:
            line += " | loops: " + ", ".join(open_loops)

        if total_chars + len(line) > max_episode_chars:
            remaining = max_episode_chars - total_chars
            if remaining > 50:
                episode_lines.append(line[:remaining] + "...(절단됨)")
            break

        episode_lines.append(line)
        total_chars += len(line)

    direct_evidence_lines = []
    direct_evidence_count = 0
    if include_direct_evidence:
        db: Session = SessionLocal()
        try:
            evidence_rows = (
                db.query(DirectEvidenceRecord)
                .filter(
                    DirectEvidenceRecord.chat_session_id == sid,
                    DirectEvidenceRecord.source_turn_end >= from_turn,
                    DirectEvidenceRecord.source_turn_start <= to_turn,
                )
                .order_by(DirectEvidenceRecord.source_turn_start.asc(), DirectEvidenceRecord.id.asc())
                .all()
            )

            evidence_chars = 0
            for evidence_row in evidence_rows:
                if not is_gate_approved_current_truth(
                    archive_state=evidence_row.archive_state,
                    capture_verification=evidence_row.capture_verification,
                    repair_needed=evidence_row.repair_needed,
                    committed_gate=evidence_row.committed_gate,
                    tombstoned=evidence_row.tombstoned,
                    superseded_by_id=evidence_row.superseded_by_id,
                ):
                    continue

                if not _is_direct_evidence_retained_for_consumption(
                    evidence_row,
                    latest_turn_index=to_turn,
                ):
                    continue

                normalized_committed_gate = resolve_committed_gate(
                    evidence_row.capture_verification,
                    evidence_row.repair_needed,
                    evidence_row.committed_gate,
                )

                evidence_text = " ".join(str(evidence_row.evidence_text or "").split()).strip()
                if not evidence_text:
                    continue

                source_turn_start = evidence_row.source_turn_start if evidence_row.source_turn_start is not None else "?"
                source_turn_end = evidence_row.source_turn_end if evidence_row.source_turn_end is not None else "?"
                line = f"[evidence turn {source_turn_start}-{source_turn_end}][gate:{normalized_committed_gate}] {evidence_text}"
                if evidence_chars + len(line) > max_direct_evidence_chars:
                    break

                direct_evidence_lines.append(line)
                evidence_chars += len(line)
                direct_evidence_count += 1
        finally:
            db.close()

    return {
        "episode_lines": episode_lines,
        "direct_evidence_lines": direct_evidence_lines,
        "episode_count": len(episode_lines),
        "direct_evidence_count": direct_evidence_count,
    }


def _call_chapter_llm(user_prompt: str) -> str | None:
    """Critic LLM 설정을 재사용하여 chapter 요약 LLM을 호출한다."""
    endpoint = settings.PROJECT_CRITIC_ENDPOINT
    api_key = settings.PROJECT_CRITIC_API_KEY
    model = settings.PROJECT_CRITIC_MODEL
    timeout = getattr(settings, "CRITIC_TIMEOUT", 90) or 90

    if not endpoint or not api_key or not model:
        logger.warning("Chapter LLM 설정 없음 (Critic 설정 재사용): 요약 생성 건너뜀")
        return None

    try:
        url = endpoint.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _CHAPTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
            return "\n".join(parts).strip() if parts else None
        return None
    except Exception as exc:
        logger.warning("Chapter LLM 호출 실패: %s", exc)
        return None


def generate_chapter_summary(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    force: bool = False,
) -> dict:
    """지정된 turn range의 chapter summary를 생성하고 저장한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return {"status": "error", "reason": "invalid chat_session_id"}

    if from_turn < 0 or to_turn < from_turn:
        return {"status": "error", "reason": f"invalid turn range: {from_turn}-{to_turn}"}

    if not force and _check_chapter_exists(chat_session_id=sid, from_turn=from_turn, to_turn=to_turn):
        return {
            "status": "skipped",
            "reason": "already_exists",
            "from_turn": from_turn,
            "to_turn": to_turn,
        }

    chapter_input = _collect_chapter_input(
        chat_session_id=sid,
        from_turn=from_turn,
        to_turn=to_turn,
    )

    if not chapter_input["episode_lines"]:
        return {
            "status": "skipped",
            "reason": "no_episode_summaries",
            "from_turn": from_turn,
            "to_turn": to_turn,
        }

    turn_span = (to_turn - from_turn) + 1
    warnings: list[str] = []
    if not (4 <= chapter_input["episode_count"] <= 8):
        warnings.append("episode_count_outside_recommended_window")
    if not (40 <= turn_span <= 80):
        warnings.append("turn_span_outside_recommended_window")

    user_prompt_parts = [
        f"Chapter range: turn {from_turn} to {to_turn}",
    ]
    if chapter_input["direct_evidence_lines"]:
        user_prompt_parts.extend([
            "",
            "=== Verified Direct Evidence (gate-approved) ===",
            *chapter_input["direct_evidence_lines"],
        ])
    user_prompt_parts.extend([
        "",
        "=== Episode Summaries ===",
        *chapter_input["episode_lines"],
    ])
    user_prompt = "\n".join(user_prompt_parts)

    raw_response = _call_chapter_llm(user_prompt)
    if not raw_response:
        return {
            "status": "error",
            "reason": "llm_call_failed",
            "from_turn": from_turn,
            "to_turn": to_turn,
            "warnings": warnings,
            "input_stats": {
                "episode_count": chapter_input["episode_count"],
                "direct_evidence_count": chapter_input["direct_evidence_count"],
                "direct_evidence_first_applied": bool(chapter_input["direct_evidence_lines"]),
                "turn_span": turn_span,
            },
        }

    parsed = _parse_chapter_json(raw_response)
    if not parsed:
        return {
            "status": "error",
            "reason": "json_parse_failed",
            "from_turn": from_turn,
            "to_turn": to_turn,
            "warnings": warnings,
            "raw_preview": raw_response[:300],
        }

    embedding_vector_json = None
    embedding_model_name = None
    try:
        archive_bridge = ArchiveBridge()
        vec = archive_bridge._get_embedding(parsed["summary_text"])
        if vec:
            embedding_vector_json = json.dumps(vec)
            embedding_model_name = settings.PROJECT_EMBEDDING_MODEL or None
    except Exception as exc:
        logger.warning("Chapter embedding 생성 실패 (계속 진행): %s", exc)

    try:
        saved = create_chapter_summary(
            chat_session_id=sid,
            from_turn=from_turn,
            to_turn=to_turn,
            chapter_index=_next_chapter_index(chat_session_id=sid),
            chapter_title=parsed.get("chapter_title"),
            summary_text=parsed["summary_text"],
            open_loops_json=parsed.get("open_loops_json"),
            relationship_changes_json=parsed.get("relationship_changes_json"),
            world_changes_json=parsed.get("world_changes_json"),
            callback_candidates_json=parsed.get("callback_candidates_json"),
            resume_text=parsed.get("resume_text"),
            embedding_vector=embedding_vector_json,
            embedding_model=embedding_model_name,
        )
        return {
            "status": "generated",
            "chapter": saved,
            "warnings": warnings,
            "input_stats": {
                "episode_count": chapter_input["episode_count"],
                "direct_evidence_count": chapter_input["direct_evidence_count"],
                "direct_evidence_first_applied": bool(chapter_input["direct_evidence_lines"]),
                "turn_span": turn_span,
            },
        }
    except Exception as exc:
        logger.error("generate_chapter_summary 저장 실패: %s", exc)
        return {
            "status": "error",
            "reason": "db_insert_failed",
            "detail": str(exc),
            "from_turn": from_turn,
            "to_turn": to_turn,
        }


# ------------------------------------------------------------------ #
#  Phase 3-2 — Episode Summary Generation
# ------------------------------------------------------------------ #

# 에피소드 요약 생성 전용 시스템 프롬프트
_EPISODE_SYSTEM_PROMPT = """\
You are an episode summarizer for a roleplay memory system.
Given a segment of conversation (chat_logs) and extracted memories from that segment,
produce a structured JSON summary of the episode.

[Core Principles]
1. Summarize the segment's narrative arc, not individual turns.
2. Focus on: key events, relationship changes, emotional shifts, setting changes, unresolved threads.
3. key_entities: important characters, places, organizations mentioned.
4. key_events: 3-7 bullet-point key event anchors (who did what, and why it matters next).
5. relationship_changes: 0-6 concise relationship anchors (who changed with whom and how).
6. open_loops: 0-6 unresolved anchors (promise, risk, question, debt) still active after this episode.
7. summary_text: 2-4 sentence narrative summary of the entire segment.
8. Do NOT fabricate events not present in the input.
9. Write in the SAME LANGUAGE as the input conversation.

[Output Format]
Return ONLY a single valid JSON object:
```json
{
  "summary_text": "2-4 sentence narrative summary",
  "key_entities": ["entity1", "entity2"],
    "key_events": ["event anchor 1", "event anchor 2"],
    "relationship_changes": ["A started trusting B after ..."],
    "open_loops": ["Return before dawn promise still unresolved"]
}
```
Do NOT add commentary outside the JSON."""


def _collect_episode_input(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    include_memories: bool = True,
    include_direct_evidence: bool = True,
    max_chat_chars: int = 6000,
    max_memory_chars: int = 2000,
    max_direct_evidence_chars: int = 2000,
) -> dict:
    """에피소드 요약 생성을 위해 해당 구간의 chat_logs + memories를 수집한다."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return {
            "chat_lines": [],
            "memory_lines": [],
            "direct_evidence_lines": [],
            "chat_count": 0,
            "memory_count": 0,
            "direct_evidence_count": 0,
        }

    db: Session = SessionLocal()
    try:
        # chat_logs: 해당 turn range
        chat_rows = (
            db.query(ChatLog)
            .filter(
                ChatLog.chat_session_id == sid,
                ChatLog.turn_index >= from_turn,
                ChatLog.turn_index <= to_turn,
            )
            .order_by(ChatLog.turn_index.asc(), ChatLog.id.asc())
            .all()
        )

        chat_lines = []
        total_chars = 0
        for r in chat_rows:
            content = (r.content or "").strip()
            if not content:
                continue
            line = f"[turn {r.turn_index}][{r.role}] {content}"
            if total_chars + len(line) > max_chat_chars:
                # 남은 공간에 절단하여 추가
                remaining = max_chat_chars - total_chars
                if remaining > 50:
                    chat_lines.append(line[:remaining] + "...(절단됨)")
                break
            chat_lines.append(line)
            total_chars += len(line)

        memory_lines = []
        memory_count = 0
        if include_memories:
            mem_rows = (
                db.query(Memory)
                .filter(
                    Memory.chat_session_id == sid,
                    Memory.turn_index >= from_turn,
                    Memory.turn_index <= to_turn,
                )
                .order_by(Memory.turn_index.asc())
                .all()
            )
            mem_chars = 0
            for m in mem_rows:
                summary = m.summary_json or ""
                if isinstance(summary, str) and summary.strip():
                    # JSON인 경우 turn_summary만 추출 시도
                    try:
                        parsed = json.loads(summary)
                        ts = parsed.get("turn_summary", "")
                        if ts:
                            summary = ts
                    except Exception:
                        pass
                line = f"[mem turn {m.turn_index}] {summary}"
                if mem_chars + len(line) > max_memory_chars:
                    break
                memory_lines.append(line)
                mem_chars += len(line)
                memory_count += 1

        direct_evidence_lines = []
        direct_evidence_count = 0
        if include_direct_evidence:
            evidence_rows = (
                db.query(DirectEvidenceRecord)
                .filter(
                    DirectEvidenceRecord.chat_session_id == sid,
                    DirectEvidenceRecord.source_turn_end >= from_turn,
                    DirectEvidenceRecord.source_turn_start <= to_turn,
                )
                .order_by(DirectEvidenceRecord.source_turn_start.asc(), DirectEvidenceRecord.id.asc())
                .all()
            )

            evidence_chars = 0
            for evidence_row in evidence_rows:
                if not is_gate_approved_current_truth(
                    archive_state=evidence_row.archive_state,
                    capture_verification=evidence_row.capture_verification,
                    repair_needed=evidence_row.repair_needed,
                    committed_gate=evidence_row.committed_gate,
                    tombstoned=evidence_row.tombstoned,
                    superseded_by_id=evidence_row.superseded_by_id,
                ):
                    continue

                if not _is_direct_evidence_retained_for_consumption(
                    evidence_row,
                    latest_turn_index=to_turn,
                ):
                    continue

                normalized_committed_gate = resolve_committed_gate(
                    evidence_row.capture_verification,
                    evidence_row.repair_needed,
                    evidence_row.committed_gate,
                )

                evidence_text = " ".join(str(evidence_row.evidence_text or "").split()).strip()
                if not evidence_text:
                    continue

                source_turn_start = evidence_row.source_turn_start if evidence_row.source_turn_start is not None else "?"
                source_turn_end = evidence_row.source_turn_end if evidence_row.source_turn_end is not None else "?"
                line = f"[evidence turn {source_turn_start}-{source_turn_end}][gate:{normalized_committed_gate}] {evidence_text}"
                if evidence_chars + len(line) > max_direct_evidence_chars:
                    break

                direct_evidence_lines.append(line)
                evidence_chars += len(line)
                direct_evidence_count += 1

        return {
            "chat_lines": chat_lines,
            "memory_lines": memory_lines,
            "direct_evidence_lines": direct_evidence_lines,
            "chat_count": len(chat_rows),
            "memory_count": memory_count,
            "direct_evidence_count": direct_evidence_count,
        }
    finally:
        db.close()


def _check_episode_exists(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
) -> bool:
    """동일 session + 동일 turn range의 에피소드가 이미 존재하는지 확인."""
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return False

    db: Session = SessionLocal()
    try:
        count = (
            db.query(EpisodeSummary)
            .filter(
                EpisodeSummary.chat_session_id == sid,
                EpisodeSummary.from_turn == from_turn,
                EpisodeSummary.to_turn == to_turn,
            )
            .count()
        )
        return count > 0
    finally:
        db.close()


def _call_episode_llm(user_prompt: str) -> str | None:
    """Critic LLM 설정을 재사용하여 에피소드 요약 LLM을 호출한다."""
    endpoint = settings.PROJECT_CRITIC_ENDPOINT
    api_key = settings.PROJECT_CRITIC_API_KEY
    model = settings.PROJECT_CRITIC_MODEL
    timeout = getattr(settings, "CRITIC_TIMEOUT", 90) or 90

    if not endpoint or not api_key or not model:
        logger.warning("Episode LLM 설정 없음 (Critic 설정 재사용): 요약 생성 건너뜀")
        return None

    try:
        url = endpoint.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _EPISODE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            return None
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
            return "\n".join(parts).strip() if parts else None
        return None
    except Exception as exc:
        logger.warning("Episode LLM 호출 실패: %s", exc)
        return None


def _normalize_episode_summary_payload(parsed: dict) -> dict | None:
    if not isinstance(parsed, dict):
        return None

    summary_text = " ".join(str(parsed.get("summary_text") or "").split()).strip()
    if not summary_text:
        return None
    if len(summary_text) > 1800:
        summary_text = summary_text[:1797] + "..."

    key_entities = _normalize_chapter_list_field(parsed.get("key_entities"), max_items=8, max_item_len=120)
    key_events = _normalize_chapter_list_field(parsed.get("key_events"), max_items=10, max_item_len=220)
    open_loops = _normalize_chapter_list_field(parsed.get("open_loops"), max_items=8, max_item_len=240)
    relationship_changes = _normalize_chapter_list_field(parsed.get("relationship_changes"), max_items=8, max_item_len=240)

    return {
        "summary_text": summary_text,
        "key_entities": json.dumps(key_entities, ensure_ascii=False) if key_entities else None,
        "key_events": json.dumps(key_events, ensure_ascii=False) if key_events else None,
        "open_loops_json": json.dumps(open_loops, ensure_ascii=False) if open_loops else None,
        "relationship_changes_json": json.dumps(relationship_changes, ensure_ascii=False) if relationship_changes else None,
        "episode_dense_summary_policy_version": "ds1a.v1",
    }


def _parse_episode_json(raw: str) -> dict | None:
    """LLM 응답에서 episode JSON을 파싱하고 DS-1a 계약으로 정규화한다."""
    # ```json ... ``` 블록 우선
    code_block = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = []
    if code_block:
        candidates.append(code_block.group(1).strip())
    candidates.append(raw.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            normalized = _normalize_episode_summary_payload(parsed)
            if normalized is not None:
                return normalized
        except Exception:
            continue
    return None


def generate_episode_summary(
    *,
    chat_session_id: str,
    from_turn: int,
    to_turn: int,
    force: bool = False,
) -> dict:
    """지정된 turn range의 에피소드 요약을 생성하고 저장한다.

    Returns:
        {"status": "generated"|"skipped"|"error", ...}
    """
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return {"status": "error", "reason": "invalid chat_session_id"}

    if from_turn < 0 or to_turn < from_turn:
        return {"status": "error", "reason": f"invalid turn range: {from_turn}-{to_turn}"}

    # 중복 체크
    if not force and _check_episode_exists(chat_session_id=sid, from_turn=from_turn, to_turn=to_turn):
        return {"status": "skipped", "reason": "already_exists", "from_turn": from_turn, "to_turn": to_turn}

    # 입력 수집
    episode_input = _collect_episode_input(
        chat_session_id=sid,
        from_turn=from_turn,
        to_turn=to_turn,
    )

    if not episode_input["chat_lines"]:
        return {"status": "skipped", "reason": "no_chat_logs", "from_turn": from_turn, "to_turn": to_turn}

    # LLM 프롬프트 조립
    user_prompt_parts = [
        f"Episode range: turn {from_turn} to {to_turn}",
    ]
    if episode_input["direct_evidence_lines"]:
        user_prompt_parts.extend([
            "",
            "=== Verified Direct Evidence (gate-approved) ===",
            *episode_input["direct_evidence_lines"],
        ])
    user_prompt_parts.extend([
        "",
        "=== Conversation ===",
        *episode_input["chat_lines"],
    ])
    if episode_input["memory_lines"]:
        user_prompt_parts.extend([
            "",
            "=== Extracted Memories ===",
            *episode_input["memory_lines"],
        ])
    user_prompt = "\n".join(user_prompt_parts)

    # LLM 호출
    raw_response = _call_episode_llm(user_prompt)
    if not raw_response:
        return {
            "status": "error",
            "reason": "llm_call_failed",
            "from_turn": from_turn,
            "to_turn": to_turn,
            "input_stats": {
                "chat_count": episode_input["chat_count"],
                "memory_count": episode_input["memory_count"],
                "direct_evidence_count": episode_input["direct_evidence_count"],
                "direct_evidence_first_applied": bool(episode_input["direct_evidence_lines"]),
            },
        }

    # 파싱
    parsed = _parse_episode_json(raw_response)
    if not parsed:
        return {
            "status": "error",
            "reason": "json_parse_failed",
            "from_turn": from_turn,
            "to_turn": to_turn,
            "raw_preview": raw_response[:300],
        }

    summary_text = str(parsed.get("summary_text", "")).strip()
    if not summary_text:
        return {"status": "error", "reason": "empty_summary_text", "from_turn": from_turn, "to_turn": to_turn}

    entities_json = parsed.get("key_entities")
    events_json = parsed.get("key_events")
    open_loops_json = parsed.get("open_loops_json")
    relationship_changes_json = parsed.get("relationship_changes_json")

    # embedding 생성 (실패해도 episode는 저장)
    embedding_vector_json = None
    embedding_model_name = None
    try:
        archive_bridge = ArchiveBridge()
        vec = archive_bridge._get_embedding(summary_text)
        if vec:
            embedding_vector_json = json.dumps(vec)
            embedding_model_name = settings.PROJECT_EMBEDDING_MODEL or None
    except Exception as exc:
        logger.warning("Episode embedding 생성 실패 (계속 진행): %s", exc)

    # 저장
    try:
        saved = create_episode_summary(
            chat_session_id=sid,
            from_turn=from_turn,
            to_turn=to_turn,
            summary_text=summary_text,
            key_entities=entities_json,
            key_events=events_json,
            open_loops_json=open_loops_json,
            relationship_changes_json=relationship_changes_json,
            embedding_vector=embedding_vector_json,
            embedding_model=embedding_model_name,
        )
        return {
            "status": "generated",
            "episode": saved,
            "input_stats": {
                "chat_count": episode_input["chat_count"],
                "memory_count": episode_input["memory_count"],
                "direct_evidence_count": episode_input["direct_evidence_count"],
                "direct_evidence_first_applied": bool(episode_input["direct_evidence_lines"]),
                "episode_dense_summary_policy_version": "ds1a.v1",
            },
            "has_embedding": embedding_vector_json is not None,
        }
    except Exception as exc:
        logger.error("Episode 저장 실패: %s", exc)
        return {"status": "error", "reason": f"save_failed: {exc}", "from_turn": from_turn, "to_turn": to_turn}


# ------------------------------------------------------------------ #
#  Phase 3-3 — 에피소드 검색 (embedding 기반)
# ------------------------------------------------------------------ #

_Q2A_ANN_VERSION = "q2a.v1"
_Q2A_ANN_TIERS = ("episode", "chapter", "arc", "saga")
_Q2A_ANN_SIGNATURE_BITS = 12
_Q2A_ANN_CANDIDATE_MULTIPLIER = 4
_Q2A_ANN_MAX_CANDIDATES_PER_TIER = 12
_Q2D_BENCHMARK_TIMEOUT_MS = 250
_Q2D_BENCHMARK_STRATEGY = "full_scan_rerank_merge_v1"
_Q2B_ANN_RECENCY_FLOOR = 0.35
_Q2B_ANN_RERANK_WEIGHTS = {
    "similarity": 0.6,
    "recency": 0.18,
    "importance": 0.12,
    "pinned": 0.07,
    "status": 0.03,
}

_DS1D_DENSE_SUMMARY_POLICY_VERSION = "ds1d.v1"
_DS1D_DENSE_SUMMARY_SCORE_WEIGHTS = {
    "similarity": 0.62,
    "recency": 0.06,
    "importance": 0.2,
    "relationship": 0.08,
    "world": 0.04,
}
_DS1G_IMPORTANT_FACT_RETENTION_POLICY_VERSION = "ds1g.v1"
_DS1G_RETENTION_MIN_SIMILARITY = 0.2
_DS1G_RETENTION_IMPORTANCE_FLOOR = 0.75
_DS1G_RETENTION_ANCHOR_FLOOR = 0.55
_DS1F_DENSE_SUMMARY_SOURCE_ANCHOR_POLICY_VERSION = "ds1f.v1"
_DS1H_DENSE_SUMMARY_ROLE_SPLIT_POLICY_VERSION = "ds1h.v1"
_DS1I_DIRECT_EVIDENCE_PROMOTION_POLICY_VERSION = "ds1i.v1"
_DS1I_DIRECT_EVIDENCE_PROMOTION_WEIGHT = 0.12


def _build_dense_summary_source_anchor_ds1f(
    *,
    source_record_id: int | None,
    source_record_type: str,
    from_turn: int | None,
    to_turn: int | None,
) -> dict:
    return {
        "dense_source_anchor_policy_version": _DS1F_DENSE_SUMMARY_SOURCE_ANCHOR_POLICY_VERSION,
        "source_record_id": source_record_id if isinstance(source_record_id, int) else None,
        "source_record_type": str(source_record_type or "").strip() or "unknown",
        "source_turn_range": {
            "from_turn": from_turn if isinstance(from_turn, int) else None,
            "to_turn": to_turn if isinstance(to_turn, int) else None,
        },
    }


def _build_dense_summary_role_split_ds1h(item: dict, *, tier: str) -> dict:
    narrative_text = ""
    if tier in {"episode", "chapter"}:
        narrative_text = " ".join(str(item.get("summary_text") or "").split()).strip()
    elif tier == "arc":
        narrative_text = " ".join(str(item.get("arc_resume_text") or item.get("core_conflict") or "").split()).strip()
    elif tier == "saga":
        narrative_text = " ".join(str(item.get("resume_pack_text") or item.get("saga_summary") or "").split()).strip()

    structured_payload: dict = {}
    if tier == "episode":
        structured_payload = {
            "key_entities": _json_list_preview_q2b(item.get("key_entities"), max_items=8),
            "key_events": _json_list_preview_q2b(item.get("key_events"), max_items=8),
            "open_loops": _json_list_preview_q2b(item.get("open_loops_json"), max_items=8),
            "relationship_changes": _json_list_preview_q2b(item.get("relationship_changes_json"), max_items=8),
        }
    elif tier == "chapter":
        structured_payload = {
            "open_loops": _json_list_preview_q2b(item.get("open_loops_json"), max_items=8),
            "relationship_changes": _json_list_preview_q2b(item.get("relationship_changes_json"), max_items=8),
            "world_changes": _json_list_preview_q2b(item.get("world_changes_json"), max_items=8),
            "callback_candidates": _json_list_preview_q2b(item.get("callback_candidates_json"), max_items=8),
            "resume": _json_list_preview_q2b(item.get("resume_text"), max_items=1),
        }
    elif tier == "arc":
        structured_payload = {
            "active_promises": _json_list_preview_q2b(item.get("active_promises_json"), max_items=8),
            "unresolved_debts": _json_list_preview_q2b(item.get("unresolved_debts_json"), max_items=8),
            "callback_debts": _json_list_preview_q2b(item.get("callback_debts_json"), max_items=8),
            "relationship_pivots": _json_list_preview_q2b(item.get("relationship_pivots_json"), max_items=8),
            "irreversible_turns": _json_list_preview_q2b(item.get("irreversible_turns_json"), max_items=8),
            "future_payoff_candidates": _json_list_preview_q2b(item.get("future_payoff_candidates_json"), max_items=8),
            "arc_status": " ".join(str(item.get("arc_status") or "").split()).strip(),
        }
    elif tier == "saga":
        structured_payload = {
            "persistent_facts": _json_list_preview_q2b(item.get("persistent_facts_json"), max_items=8),
            "never_drop_candidates": _json_list_preview_q2b(item.get("never_drop_candidates_json"), max_items=8),
            "era_label": " ".join(str(item.get("era_label") or "").split()).strip(),
        }

    return {
        "dense_role_split_policy_version": _DS1H_DENSE_SUMMARY_ROLE_SPLIT_POLICY_VERSION,
        "dense_narrative_text": narrative_text,
        "dense_narrative_usage": "read_only",
        "dense_structured_usage": "adjudication_retrieval",
        "dense_structured_payload": structured_payload,
    }


def _normalize_embedding_vector_q2a(raw: object) -> list[float] | None:
    if raw is None:
        return None

    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return None

    if not isinstance(parsed, list) or not parsed:
        return None

    normalized: list[float] = []
    for value in parsed:
        try:
            normalized.append(float(value))
        except Exception:
            return None
    return normalized or None


def _build_ann_signature_q2a(vec: list[float], *, bits: int = _Q2A_ANN_SIGNATURE_BITS) -> tuple[int, ...]:
    if not vec:
        return ()

    sample_count = max(1, min(bits, len(vec)))
    if sample_count == 1:
        indexes = [0]
    elif sample_count >= len(vec):
        indexes = list(range(len(vec)))
    else:
        last_index = len(vec) - 1
        indexes: list[int] = []
        for idx in range(sample_count):
            sampled = round((last_index * idx) / (sample_count - 1))
            if indexes and sampled <= indexes[-1]:
                sampled = min(last_index, indexes[-1] + 1)
            indexes.append(sampled)

    return tuple(1 if vec[index] >= 0.0 else 0 for index in indexes)


def _hamming_distance_q2a(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    length = min(len(left), len(right))
    distance = sum(1 for idx in range(length) if left[idx] != right[idx])
    distance += abs(len(left) - len(right))
    return distance


def _json_list_count_q2b(raw: object) -> int:
    if raw is None:
        return 0
    parsed = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return 0
        try:
            parsed = json.loads(text)
        except Exception:
            return 1

    if isinstance(parsed, list):
        return sum(1 for item in parsed if str(item or "").strip())
    if isinstance(parsed, dict):
        return sum(1 for value in parsed.values() if str(value or "").strip())
    return 1 if str(parsed or "").strip() else 0


def _json_list_preview_q2b(raw: object, *, max_items: int = 5) -> list[str]:
    if raw is None:
        return []
    parsed = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return [text[:220]]

    if isinstance(parsed, list):
        result: list[str] = []
        for item in parsed:
            text = " ".join(str(item or "").split()).strip()
            if not text:
                continue
            result.append(text[:220])
            if len(result) >= max_items:
                break
        return result

    text = " ".join(str(parsed or "").split()).strip()
    return [text[:220]] if text else []


def _turn_value_q2b(item: dict) -> int | None:
    for key in ("to_turn", "turn_index", "from_turn"):
        value = item.get(key)
        if isinstance(value, int):
            return value
    return None


def _recency_score_q2b(item: dict, *, min_turn: int | None, max_turn: int | None) -> float:
    turn_value = _turn_value_q2b(item)
    if turn_value is None or min_turn is None or max_turn is None:
        return 0.5
    if max_turn <= min_turn:
        return 1.0

    normalized = max(0.0, min(1.0, (turn_value - min_turn) / max(1, max_turn - min_turn)))
    return round(_Q2B_ANN_RECENCY_FLOOR + ((1.0 - _Q2B_ANN_RECENCY_FLOOR) * normalized), 4)


def _resolve_ann_metadata_q2b(
    item: dict,
    *,
    tier: str,
    min_turn: int | None,
    max_turn: int | None,
) -> dict:
    recency_score = _recency_score_q2b(item, min_turn=min_turn, max_turn=max_turn)
    importance_score = 0.0
    pinned_hint = False
    status = None
    status_score = 1.0
    filter_passed = True
    filter_reason = None

    if tier == "episode":
        signal_count = (
            _json_list_count_q2b(item.get("key_entities"))
            + _json_list_count_q2b(item.get("key_events"))
            + _json_list_count_q2b(item.get("open_loops_json"))
            + _json_list_count_q2b(item.get("relationship_changes_json"))
        )
        importance_score = min(1.0, signal_count / 6.0)
    elif tier == "chapter":
        open_loop_count = _json_list_count_q2b(item.get("open_loops_json"))
        relationship_count = _json_list_count_q2b(item.get("relationship_changes_json"))
        world_change_count = _json_list_count_q2b(item.get("world_changes_json"))
        callback_count = _json_list_count_q2b(item.get("callback_candidates_json"))
        resume_score = 1 if " ".join(str(item.get("resume_text") or "").split()).strip() else 0
        signal_count = open_loop_count + relationship_count + world_change_count + callback_count + resume_score
        importance_score = min(1.0, signal_count / 8.0)
        pinned_hint = callback_count > 0 and (open_loop_count > 0 or relationship_count > 0 or world_change_count > 0)
    elif tier == "arc":
        active_promise_count = _json_list_count_q2b(item.get("active_promises_json"))
        unresolved_debt_count = _json_list_count_q2b(item.get("unresolved_debts_json"))
        callback_count = _json_list_count_q2b(item.get("callback_candidates_json"))
        callback_debt_count = _json_list_count_q2b(item.get("callback_debts_json"))
        irreversible_turn_count = _json_list_count_q2b(item.get("irreversible_turns_json"))
        relationship_pivot_count = _json_list_count_q2b(item.get("relationship_pivots_json"))
        future_payoff_count = _json_list_count_q2b(item.get("future_payoff_candidates_json"))
        signal_count = (
            active_promise_count
            + unresolved_debt_count
            + callback_count
            + callback_debt_count
            + irreversible_turn_count
            + relationship_pivot_count
            + future_payoff_count
        )
        importance_score = min(1.0, signal_count / 10.0)
        status = str(item.get("arc_status") or "active").strip().lower() or "active"
        status_score = {
            "active": 1.0,
            "paused": 0.7,
            "resolved": 0.35,
        }.get(status, 0.5)
        pinned_hint = (
            unresolved_debt_count > 0
            or callback_debt_count > 0
            or irreversible_turn_count > 0
            or relationship_pivot_count > 0
            or future_payoff_count > 0
            or callback_count >= 2
        )
        if status == "resolved" and signal_count == 0:
            filter_passed = False
            filter_reason = "resolved_without_callback_material"
        elif status not in {"active", "paused", "resolved"}:
            filter_passed = False
            filter_reason = "unsupported_arc_status"
    elif tier == "saga":
        persistent_fact_count = _json_list_count_q2b(item.get("persistent_facts_json"))
        never_drop_count = _json_list_count_q2b(item.get("never_drop_candidates_json"))
        signal_count = persistent_fact_count + never_drop_count
        importance_score = min(1.0, signal_count / 8.0)
        pinned_hint = never_drop_count > 0

    return {
        "recency_score": round(float(recency_score), 4),
        "importance_score": round(float(importance_score), 4),
        "pinned_hint": bool(pinned_hint),
        "pinned_score": 1.0 if pinned_hint else 0.0,
        "status": status,
        "status_score": round(float(status_score), 4),
        "filter_passed": bool(filter_passed),
        "filter_reason": filter_reason,
    }


def _compute_rerank_score_q2b(similarity: float, metadata: dict) -> float:
    return round(
        (_Q2B_ANN_RERANK_WEIGHTS["similarity"] * float(similarity))
        + (_Q2B_ANN_RERANK_WEIGHTS["recency"] * float(metadata.get("recency_score") or 0.0))
        + (_Q2B_ANN_RERANK_WEIGHTS["importance"] * float(metadata.get("importance_score") or 0.0))
        + (_Q2B_ANN_RERANK_WEIGHTS["pinned"] * float(metadata.get("pinned_score") or 0.0))
        + (_Q2B_ANN_RERANK_WEIGHTS["status"] * float(metadata.get("status_score") or 0.0)),
        4,
    )


def _resolve_ds1d_priority_signals(item: dict, *, tier: str) -> tuple[float, float, float]:
    """DS-1d: age보다 importance/relationship/world anchor를 우선하는 점수 신호를 계산한다."""
    importance_score = float(_resolve_ann_metadata_q2b(item, tier=tier, min_turn=None, max_turn=None).get("importance_score") or 0.0)
    relationship_score = 0.0
    world_score = 0.0

    if tier == "episode":
        relationship_score = min(1.0, _json_list_count_q2b(item.get("relationship_changes_json")) / 3.0)
        world_score = min(1.0, _json_list_count_q2b(item.get("key_events")) / 4.0)
    elif tier == "chapter":
        relationship_score = min(1.0, _json_list_count_q2b(item.get("relationship_changes_json")) / 3.0)
        world_score = min(1.0, _json_list_count_q2b(item.get("world_changes_json")) / 3.0)
    elif tier == "arc":
        relationship_score = min(1.0, _json_list_count_q2b(item.get("relationship_pivots_json")) / 3.0)
        world_score = min(
            1.0,
            (
                _json_list_count_q2b(item.get("irreversible_turns_json"))
                + _json_list_count_q2b(item.get("callback_debts_json"))
            ) / 4.0,
        )
    elif tier == "saga":
        relationship_score = 0.0
        world_score = min(1.0, _json_list_count_q2b(item.get("persistent_facts_json")) / 4.0)

    return (
        round(max(0.0, min(1.0, importance_score)), 4),
        round(max(0.0, min(1.0, relationship_score)), 4),
        round(max(0.0, min(1.0, world_score)), 4),
    )


def _compute_dense_summary_priority_score_ds1d(
    *,
    similarity: float,
    recency_score: float,
    importance_score: float,
    relationship_score: float,
    world_score: float,
    direct_evidence_promotion_score: float = 0.0,
) -> float:
    return round(
        (_DS1D_DENSE_SUMMARY_SCORE_WEIGHTS["similarity"] * float(similarity))
        + (_DS1D_DENSE_SUMMARY_SCORE_WEIGHTS["recency"] * float(recency_score))
        + (_DS1D_DENSE_SUMMARY_SCORE_WEIGHTS["importance"] * float(importance_score))
        + (_DS1D_DENSE_SUMMARY_SCORE_WEIGHTS["relationship"] * float(relationship_score))
        + (_DS1D_DENSE_SUMMARY_SCORE_WEIGHTS["world"] * float(world_score))
        + (_DS1I_DIRECT_EVIDENCE_PROMOTION_WEIGHT * float(direct_evidence_promotion_score or 0.0)),
        4,
    )



def _normalize_text_ds1i(value: object) -> str:
    text = " ".join(str(value or "").split()).strip().lower()
    if not text:
        return ""
    filtered = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            filtered.append(ch)
    return " ".join("".join(filtered).split()).strip()


def _extract_structured_signals_ds1i(item: dict, *, tier: str) -> dict[str, list[str]]:
    if tier == "episode":
        relationship = _json_list_preview_q2b(item.get("relationship_changes_json"), max_items=12)
        world = _json_list_preview_q2b(item.get("key_events"), max_items=12)
        promises = _json_list_preview_q2b(item.get("open_loops_json"), max_items=12)
    elif tier == "chapter":
        relationship = _json_list_preview_q2b(item.get("relationship_changes_json"), max_items=12)
        world = _json_list_preview_q2b(item.get("world_changes_json"), max_items=12)
        promises = (
            _json_list_preview_q2b(item.get("open_loops_json"), max_items=8)
            + _json_list_preview_q2b(item.get("callback_candidates_json"), max_items=8)
        )
    elif tier == "arc":
        relationship = _json_list_preview_q2b(item.get("relationship_pivots_json"), max_items=12)
        world = (
            _json_list_preview_q2b(item.get("irreversible_turns_json"), max_items=8)
            + _json_list_preview_q2b(item.get("callback_debts_json"), max_items=8)
        )
        promises = (
            _json_list_preview_q2b(item.get("active_promises_json"), max_items=8)
            + _json_list_preview_q2b(item.get("future_payoff_candidates_json"), max_items=8)
        )
    elif tier == "saga":
        relationship = []
        world = _json_list_preview_q2b(item.get("persistent_facts_json"), max_items=12)
        promises = _json_list_preview_q2b(item.get("never_drop_candidates_json"), max_items=12)
    else:
        relationship = []
        world = []
        promises = []

    return {
        "relationship": [text for text in relationship if _normalize_text_ds1i(text)],
        "world": [text for text in world if _normalize_text_ds1i(text)],
        "promise": [text for text in promises if _normalize_text_ds1i(text)],
    }


def _signal_supported_by_evidence_ds1i(signal_text: str, evidence_texts: list[str]) -> bool:
    signal = _normalize_text_ds1i(signal_text)
    if not signal or not evidence_texts:
        return False

    for evidence in evidence_texts:
        normalized_evidence = _normalize_text_ds1i(evidence)
        if not normalized_evidence:
            continue
        if signal in normalized_evidence or normalized_evidence in signal:
            return True
    return False


def _load_direct_evidence_support_cache_ds1i(db: Session, *, chat_session_id: str, latest_turn: int | None) -> list[dict]:
    rows = (
        db.query(DirectEvidenceRecord)
        .filter(DirectEvidenceRecord.chat_session_id == chat_session_id)
        .order_by(DirectEvidenceRecord.source_turn_start.asc(), DirectEvidenceRecord.id.asc())
        .all()
    )

    results: list[dict] = []
    for row in rows:
        if not is_gate_approved_current_truth(
            archive_state=row.archive_state,
            capture_verification=row.capture_verification,
            repair_needed=row.repair_needed,
            committed_gate=row.committed_gate,
            tombstoned=row.tombstoned,
            superseded_by_id=row.superseded_by_id,
        ):
            continue

        if latest_turn is not None and not _is_direct_evidence_retained_for_consumption(row, latest_turn_index=latest_turn):
            continue

        evidence_text = " ".join(str(row.evidence_text or "").split()).strip()
        if not evidence_text:
            continue

        results.append({
            "source_turn_start": int(row.source_turn_start),
            "source_turn_end": int(row.source_turn_end),
            "evidence_text": evidence_text,
        })
    return results


def _resolve_direct_evidence_promotion_ds1i(
    *,
    item: dict,
    tier: str,
    evidence_cache: list[dict],
) -> dict:
    from_turn = item.get("from_turn") if isinstance(item.get("from_turn"), int) else None
    to_turn = item.get("to_turn") if isinstance(item.get("to_turn"), int) else None
    if from_turn is None or to_turn is None:
        return {
            "promotion_score": 0.0,
            "promoted_relationship_count": 0,
            "promoted_world_count": 0,
            "promoted_promise_count": 0,
            "promoted_signal_count": 0,
            "structured_precedence_applied": False,
        }

    evidence_texts = [
        row.get("evidence_text")
        for row in evidence_cache
        if int(row.get("source_turn_start") or 0) <= to_turn and int(row.get("source_turn_end") or 0) >= from_turn
    ]
    evidence_texts = [str(text) for text in evidence_texts if str(text or "").strip()]
    if not evidence_texts:
        return {
            "promotion_score": 0.0,
            "promoted_relationship_count": 0,
            "promoted_world_count": 0,
            "promoted_promise_count": 0,
            "promoted_signal_count": 0,
            "structured_precedence_applied": False,
        }

    structured = _extract_structured_signals_ds1i(item, tier=tier)
    rel_count = sum(1 for signal in structured["relationship"] if _signal_supported_by_evidence_ds1i(signal, evidence_texts))
    world_count = sum(1 for signal in structured["world"] if _signal_supported_by_evidence_ds1i(signal, evidence_texts))
    promise_count = sum(1 for signal in structured["promise"] if _signal_supported_by_evidence_ds1i(signal, evidence_texts))

    rel_score = min(1.0, rel_count / 2.0)
    world_score = min(1.0, world_count / 2.0)
    promise_score = min(1.0, promise_count / 2.0)
    promotion_score = round((0.45 * rel_score) + (0.25 * world_score) + (0.30 * promise_score), 4)

    total_promoted = int(rel_count + world_count + promise_count)
    return {
        "promotion_score": promotion_score,
        "promoted_relationship_count": int(rel_count),
        "promoted_world_count": int(world_count),
        "promoted_promise_count": int(promise_count),
        "promoted_signal_count": total_promoted,
        "structured_precedence_applied": total_promoted > 0,
    }

def _passes_ds1d_similarity_gate(
    *,
    similarity: float,
    importance_score: float,
    relationship_score: float,
    world_score: float,
) -> bool:
    if similarity >= 0.5:
        return True
    if similarity < 0.4:
        return False
    return (
        importance_score >= 0.65
        or relationship_score >= 0.55
        or world_score >= 0.55
    )


def _passes_ds1g_important_fact_retention(
    *,
    similarity: float,
    importance_score: float,
    relationship_score: float,
    world_score: float,
    metadata: dict | None,
) -> tuple[bool, bool, str | None]:
    if similarity < _DS1G_RETENTION_MIN_SIMILARITY:
        return False, False, None

    pinned_hint = bool((metadata or {}).get("pinned_hint") or float((metadata or {}).get("pinned_score") or 0.0) >= 1.0)
    has_dense_anchor = (
        relationship_score >= _DS1G_RETENTION_ANCHOR_FLOOR
        or world_score >= _DS1G_RETENTION_ANCHOR_FLOOR
    )

    if importance_score >= _DS1G_RETENTION_IMPORTANCE_FLOOR and (has_dense_anchor or pinned_hint):
        return True, True, "important_fact_retention"

    return False, False, None


def _passes_dense_summary_gate_ds1g(
    *,
    similarity: float,
    importance_score: float,
    relationship_score: float,
    world_score: float,
    metadata: dict | None,
) -> tuple[bool, bool, str | None]:
    if _passes_ds1d_similarity_gate(
        similarity=similarity,
        importance_score=importance_score,
        relationship_score=relationship_score,
        world_score=world_score,
    ):
        return True, False, None

    return _passes_ds1g_important_fact_retention(
        similarity=similarity,
        importance_score=importance_score,
        relationship_score=relationship_score,
        world_score=world_score,
        metadata=metadata,
    )


def _ann_merge_key_q2c(item: dict, *, fallback_tier: str | None = None) -> tuple[str, int | str]:
    tier = str(item.get("source_type") or fallback_tier or "unknown")
    row_id = item.get("id")
    if isinstance(row_id, int):
        return (tier, row_id)
    return (tier, str(item.get("document_id") or item.get("title") or ""))


def _ann_merge_sort_key_q2c(item: dict) -> tuple[float, float, float, float, int, int, int]:
    return (
        float(item.get("ann_rerank_score") or item.get("similarity") or 0.0),
        float(item.get("similarity") or 0.0),
        float(item.get("ann_pinned_score") or 0.0),
        float(item.get("ann_recency_score") or 0.0),
        -int(item.get("ann_bucket_distance") or 0),
        -int(item.get("ann_candidate_rank") or 0),
        int(item.get("id") or -1),
    )


def _merge_ann_candidates_q2c(tier_items: dict[str, list[dict]]) -> dict:
    ordered_tiers = [tier for tier in _Q2A_ANN_TIERS if tier_items.get(tier)]
    if not ordered_tiers:
        return {
            "applied": False,
            "policy": "tier_head_then_rerank_v1",
            "items": [],
            "tier_sequence": [],
        }

    merged_items: list[dict] = []
    seen_keys: set[tuple[str, int | str]] = set()

    tier_heads = sorted(
        [tier_items[tier][0] for tier in ordered_tiers if tier_items.get(tier)],
        key=_ann_merge_sort_key_q2c,
        reverse=True,
    )
    for item in tier_heads:
        key = _ann_merge_key_q2c(item)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_items.append({**item, "ann_merge_pass": "tier_head"})

    fill_candidates: list[dict] = []
    for tier in ordered_tiers:
        for item in tier_items.get(tier) or []:
            key = _ann_merge_key_q2c(item, fallback_tier=tier)
            if key in seen_keys:
                continue
            fill_candidates.append(item)

    fill_candidates.sort(key=_ann_merge_sort_key_q2c, reverse=True)
    for item in fill_candidates:
        key = _ann_merge_key_q2c(item)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        merged_items.append({**item, "ann_merge_pass": "rerank_fill"})

    finalized_items = [
        {
            **item,
            "ann_merge_rank": rank,
        }
        for rank, item in enumerate(merged_items, start=1)
    ]
    return {
        "applied": True,
        "policy": "tier_head_then_rerank_v1",
        "items": finalized_items,
        "tier_sequence": [str(item.get("source_type") or "unknown") for item in finalized_items],
    }


def _empty_ann_benchmark_q2d(
    *,
    status: str = "empty",
    reason: str | None = None,
    timeout_budget_ms: int = _Q2D_BENCHMARK_TIMEOUT_MS,
) -> dict:
    benchmark = {
        "status": status,
        "strategy": _Q2D_BENCHMARK_STRATEGY,
        "timeout_budget_ms": int(timeout_budget_ms),
        "shadow_selection_latency_ms": 0.0,
        "baseline_selection_latency_ms": 0.0,
        "latency_delta_ms": 0.0,
        "latency_improved": False,
        "shadow_candidate_count": 0,
        "baseline_candidate_count": 0,
        "shadow_tier_diversity": 0,
        "baseline_tier_diversity": 0,
        "diversity_delta": 0,
        "diversity_guard_ok": True,
        "overlap_count": 0,
        "overlap_ratio": 0.0,
        "top_hit_match": False,
        "shadow_tier_sequence": [],
        "baseline_tier_sequence": [],
    }
    if reason:
        benchmark["reason"] = reason
    return benchmark


def _raise_if_benchmark_timed_out_q2d(started_at: float, timeout_budget_ms: int) -> None:
    if timeout_budget_ms <= 0:
        return
    elapsed_ms = (perf_counter() - started_at) * 1000.0
    if elapsed_ms > float(timeout_budget_ms):
        raise TimeoutError("benchmark_timeout")


def _select_full_scan_candidates_q2d(
    rows: list[dict],
    query_vec: list[float],
    *,
    tier: str,
    top_k: int,
    started_at: float,
    timeout_budget_ms: int,
) -> dict:
    archive_bridge = ArchiveBridge()
    desired_pool = max(
        top_k,
        min(_Q2A_ANN_MAX_CANDIDATES_PER_TIER, top_k * _Q2A_ANN_CANDIDATE_MULTIPLIER),
    )

    normalized_entries: list[dict] = []
    for row in rows:
        _raise_if_benchmark_timed_out_q2d(started_at, timeout_budget_ms)
        row_vec = _normalize_embedding_vector_q2a(row.get("embedding_vector"))
        if row_vec is None or len(row_vec) != len(query_vec):
            continue
        similarity = archive_bridge._cosine(query_vec, row_vec)
        if similarity <= 0.0:
            continue
        normalized_entries.append({
            "row": row,
            "vector": row_vec,
            "similarity": similarity,
        })

    if not normalized_entries:
        return {
            "items": [],
            "filtered_out_count": 0,
            "filtered_out_reasons": {},
        }

    turn_values = [value for value in (_turn_value_q2b(entry["row"]) for entry in normalized_entries) if value is not None]
    min_turn = min(turn_values) if turn_values else None
    max_turn = max(turn_values) if turn_values else None

    filtered_out_count = 0
    filtered_out_reasons: dict[str, int] = {}
    scored: list[tuple[float, float, float, int, int, dict, dict]] = []
    for entry in normalized_entries:
        _raise_if_benchmark_timed_out_q2d(started_at, timeout_budget_ms)
        metadata = _resolve_ann_metadata_q2b(
            entry["row"],
            tier=tier,
            min_turn=min_turn,
            max_turn=max_turn,
        )
        if not metadata["filter_passed"]:
            filtered_out_count += 1
            reason = str(metadata.get("filter_reason") or "metadata_filtered")
            filtered_out_reasons[reason] = int(filtered_out_reasons.get(reason, 0)) + 1
            continue
        rerank_score = _compute_rerank_score_q2b(float(entry["similarity"]), metadata)
        scored.append(
            (
                rerank_score,
                float(entry["similarity"]),
                float(metadata.get("recency_score") or 0.0),
                0,
                int((entry["row"] or {}).get("id") or -1),
                entry,
                metadata,
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
            float(item[6].get("pinned_score") or 0.0),
            item[2],
            item[3],
            int((item[5]["row"] or {}).get("to_turn") or (item[5]["row"] or {}).get("turn_index") or -1),
            item[4],
        ),
        reverse=True,
    )

    items: list[dict] = []
    for rank, (rerank_score, similarity, _recency_score, _bucket_tiebreak, _row_id, entry, metadata) in enumerate(scored[:desired_pool], start=1):
        _raise_if_benchmark_timed_out_q2d(started_at, timeout_budget_ms)
        items.append(
            _sanitize_ann_candidate_item_q2a(
                entry["row"],
                tier=tier,
                similarity=similarity,
                rank=rank,
                bucket_distance=0,
                rerank_score=rerank_score,
                metadata=metadata,
            )
        )
    return {
        "items": items,
        "filtered_out_count": filtered_out_count,
        "filtered_out_reasons": filtered_out_reasons,
    }


def _build_ann_benchmark_q2d(
    snapshot: dict,
    tier_rows: dict[str, list[dict]],
    query_vec: list[float],
    *,
    top_k: int,
    shadow_selection_latency_ms: float,
    timeout_budget_ms: int = _Q2D_BENCHMARK_TIMEOUT_MS,
) -> dict:
    started_at = perf_counter()
    baseline_tier_items: dict[str, list[dict]] = {tier: [] for tier in _Q2A_ANN_TIERS}

    for tier in _Q2A_ANN_TIERS:
        _raise_if_benchmark_timed_out_q2d(started_at, timeout_budget_ms)
        selected = _select_full_scan_candidates_q2d(
            tier_rows.get(tier) or [],
            query_vec,
            tier=tier,
            top_k=top_k,
            started_at=started_at,
            timeout_budget_ms=timeout_budget_ms,
        )
        baseline_tier_items[tier] = list(selected.get("items") or [])

    baseline_merge = _merge_ann_candidates_q2c(baseline_tier_items)
    baseline_tier_sequence = list(baseline_merge.get("tier_sequence") or [])
    shadow_tier_sequence = list(snapshot.get("merged_tier_sequence") or [])
    baseline_candidate_count = sum(len(items) for items in baseline_tier_items.values())
    shadow_candidate_count = int(snapshot.get("candidate_count") or 0)

    baseline_keys = {_ann_merge_key_q2c(item) for item in (baseline_merge.get("items") or [])}
    shadow_keys = {_ann_merge_key_q2c(item) for item in (snapshot.get("merged_items") or [])}
    overlap_count = len(baseline_keys & shadow_keys)
    overlap_ratio = round(overlap_count / max(1, len(baseline_keys)), 4) if baseline_keys else 1.0
    baseline_tier_diversity = len(set(baseline_tier_sequence))
    shadow_tier_diversity = len(set(shadow_tier_sequence))
    diversity_delta = int(shadow_tier_diversity - baseline_tier_diversity)
    diversity_guard_ok = shadow_tier_diversity >= max(1, baseline_tier_diversity - 1) if baseline_tier_diversity > 0 else True
    baseline_latency_ms = round((perf_counter() - started_at) * 1000.0, 3)
    shadow_latency_ms = round(float(shadow_selection_latency_ms), 3)
    latency_delta_ms = round(baseline_latency_ms - shadow_latency_ms, 3)

    baseline_items = list(baseline_merge.get("items") or [])
    shadow_items = list(snapshot.get("merged_items") or [])
    return {
        "status": "ready",
        "strategy": _Q2D_BENCHMARK_STRATEGY,
        "timeout_budget_ms": int(timeout_budget_ms),
        "shadow_selection_latency_ms": shadow_latency_ms,
        "baseline_selection_latency_ms": baseline_latency_ms,
        "latency_delta_ms": latency_delta_ms,
        "latency_improved": baseline_latency_ms >= shadow_latency_ms,
        "shadow_candidate_count": shadow_candidate_count,
        "baseline_candidate_count": baseline_candidate_count,
        "shadow_tier_diversity": shadow_tier_diversity,
        "baseline_tier_diversity": baseline_tier_diversity,
        "diversity_delta": diversity_delta,
        "diversity_guard_ok": diversity_guard_ok,
        "overlap_count": overlap_count,
        "overlap_ratio": overlap_ratio,
        "top_hit_match": bool(
            baseline_items
            and shadow_items
            and _ann_merge_key_q2c(baseline_items[0]) == _ann_merge_key_q2c(shadow_items[0])
        ),
        "shadow_tier_sequence": shadow_tier_sequence,
        "baseline_tier_sequence": baseline_tier_sequence,
    }


def _sanitize_ann_candidate_item_q2a(
    item: dict,
    *,
    tier: str,
    similarity: float,
    rank: int,
    bucket_distance: int,
    rerank_score: float,
    metadata: dict,
) -> dict:
    normalized = dict(item)
    normalized.pop("embedding_vector", None)
    normalized.pop("embedding_model", None)
    normalized["source_type"] = normalized.get("source_type") or tier
    normalized["similarity"] = round(float(similarity), 4)
    normalized["ann_base_similarity"] = round(float(similarity), 4)
    normalized["ann_candidate_rank"] = rank
    normalized["ann_bucket_distance"] = bucket_distance
    normalized["ann_rerank_score"] = round(float(rerank_score), 4)
    normalized["ann_recency_score"] = round(float(metadata.get("recency_score") or 0.0), 4)
    normalized["ann_importance_score"] = round(float(metadata.get("importance_score") or 0.0), 4)
    normalized["ann_pinned_hint"] = bool(metadata.get("pinned_hint"))
    normalized["ann_pinned_score"] = round(float(metadata.get("pinned_score") or 0.0), 4)
    normalized["ann_status_score"] = round(float(metadata.get("status_score") or 0.0), 4)
    if metadata.get("status"):
        normalized["ann_status"] = metadata.get("status")
    return normalized


def _select_ann_candidates_q2a(
    rows: list[dict],
    query_vec: list[float],
    *,
    tier: str,
    top_k: int,
) -> dict:
    archive_bridge = ArchiveBridge()
    normalized_rows: list[dict] = []
    buckets: dict[tuple[int, ...], list[dict]] = {}

    for row in rows:
        row_vec = _normalize_embedding_vector_q2a(row.get("embedding_vector"))
        if row_vec is None or len(row_vec) != len(query_vec):
            continue
        signature = _build_ann_signature_q2a(row_vec)
        entry = {
            "row": row,
            "vector": row_vec,
            "signature": signature,
        }
        normalized_rows.append(entry)
        buckets.setdefault(signature, []).append(entry)

    indexed_rows = len(normalized_rows)
    if indexed_rows == 0:
        return {
            "indexed_rows": 0,
            "bucket_count": 0,
            "fallback_used": False,
            "filtered_out_count": 0,
            "filtered_out_reasons": {},
            "rerank_applied": False,
            "items": [],
        }

    desired_pool = max(
        top_k,
        min(_Q2A_ANN_MAX_CANDIDATES_PER_TIER, top_k * _Q2A_ANN_CANDIDATE_MULTIPLIER),
    )
    query_signature = _build_ann_signature_q2a(query_vec)
    ordered_buckets = sorted(
        buckets.items(),
        key=lambda item: (
            _hamming_distance_q2a(query_signature, item[0]),
            -len(item[1]),
            str(item[0]),
        ),
    )

    selected_entries: list[dict] = []
    selected_ids: set[int] = set()
    for signature, entries in ordered_buckets:
        for entry in entries:
            row_id = entry["row"].get("id")
            if not isinstance(row_id, int) or row_id in selected_ids:
                continue
            selected_ids.add(row_id)
            selected_entries.append({**entry, "bucket_distance": _hamming_distance_q2a(query_signature, signature)})
            if len(selected_entries) >= desired_pool:
                break
        if len(selected_entries) >= desired_pool:
            break

    fallback_used = len(selected_entries) < min(top_k, indexed_rows)
    if fallback_used:
        for entry in normalized_rows:
            row_id = entry["row"].get("id")
            if not isinstance(row_id, int) or row_id in selected_ids:
                continue
            selected_ids.add(row_id)
            selected_entries.append({
                **entry,
                "bucket_distance": _hamming_distance_q2a(query_signature, entry["signature"]),
            })
            if len(selected_entries) >= desired_pool:
                break

    turn_values = [value for value in (_turn_value_q2b(entry["row"]) for entry in selected_entries) if value is not None]
    min_turn = min(turn_values) if turn_values else None
    max_turn = max(turn_values) if turn_values else None

    filtered_out_count = 0
    filtered_out_reasons: dict[str, int] = {}
    scored: list[tuple[float, float, float, int, int, dict, dict]] = []
    for entry in selected_entries:
        similarity = archive_bridge._cosine(query_vec, entry["vector"])
        if similarity <= 0.0:
            continue
        metadata = _resolve_ann_metadata_q2b(
            entry["row"],
            tier=tier,
            min_turn=min_turn,
            max_turn=max_turn,
        )
        if not metadata["filter_passed"]:
            filtered_out_count += 1
            reason = str(metadata.get("filter_reason") or "metadata_filtered")
            filtered_out_reasons[reason] = int(filtered_out_reasons.get(reason, 0)) + 1
            continue

        rerank_score = _compute_rerank_score_q2b(similarity, metadata)
        scored.append(
            (
                rerank_score,
                float(similarity),
                float(metadata.get("recency_score") or 0.0),
                -int(entry["bucket_distance"]),
                int((entry["row"] or {}).get("id") or -1),
                entry,
                metadata,
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
            float(item[6].get("pinned_score") or 0.0),
            item[2],
            item[3],
            int((item[5]["row"] or {}).get("to_turn") or (item[5]["row"] or {}).get("turn_index") or -1),
            item[4],
        ),
        reverse=True,
    )

    items: list[dict] = []
    for rank, (rerank_score, similarity, _recency_score, _bucket_tiebreak, _row_id, entry, metadata) in enumerate(scored[:desired_pool], start=1):
        items.append(
            _sanitize_ann_candidate_item_q2a(
                entry["row"],
                tier=tier,
                similarity=similarity,
                rank=rank,
                bucket_distance=int(entry["bucket_distance"]),
                rerank_score=rerank_score,
                metadata=metadata,
            )
        )

    return {
        "indexed_rows": indexed_rows,
        "bucket_count": len(buckets),
        "fallback_used": fallback_used,
        "filtered_out_count": filtered_out_count,
        "filtered_out_reasons": filtered_out_reasons,
        "rerank_applied": True,
        "items": items,
    }


def preview_summary_ann_candidates(
    query: str,
    chat_session_id: str,
    top_k: int = 3,
) -> dict:
    """Q-2a: summary tier ANN candidate preview를 read-only shadow path로 계산한다."""
    query_text = str(query or "").strip()
    sid = _safe_session_id(chat_session_id)
    top_k = max(1, min(10, int(top_k or 3)))

    base_snapshot = {
        "status": "empty",
        "version": _Q2A_ANN_VERSION,
        "chat_session_id": sid or "",
        "source": "summary_ann_shadow",
        "query_embedding_ready": False,
        "benchmark": _empty_ann_benchmark_q2d(),
        "rerank_applied": False,
        "rerank_strategy": "metadata_hint_v1",
        "merge_applied": False,
        "merge_policy": "tier_head_then_rerank_v1",
        "merged_tier_sequence": [],
        "merged_items": [],
        "candidate_limit_per_tier": max(
            top_k,
            min(_Q2A_ANN_MAX_CANDIDATES_PER_TIER, top_k * _Q2A_ANN_CANDIDATE_MULTIPLIER),
        ),
        "candidate_count": 0,
        "tier_counts": {tier: 0 for tier in _Q2A_ANN_TIERS},
        "indexed_row_counts": {tier: 0 for tier in _Q2A_ANN_TIERS},
        "bucket_counts": {tier: 0 for tier in _Q2A_ANN_TIERS},
        "filtered_out_counts": {tier: 0 for tier in _Q2A_ANN_TIERS},
        "filtered_out_reasons": {tier: {} for tier in _Q2A_ANN_TIERS},
        "filtered_out_total": 0,
        "fallback_tiers": [],
        "tier_items": {tier: [] for tier in _Q2A_ANN_TIERS},
    }
    if not query_text or not sid:
        return base_snapshot

    archive_bridge = ArchiveBridge()
    query_vec = archive_bridge._get_embedding(query_text)
    if query_vec is None:
        return {
            **base_snapshot,
            "status": "degraded",
            "benchmark": _empty_ann_benchmark_q2d(status="skipped", reason="embedding_unavailable"),
            "reason": "embedding_unavailable",
        }

    base_snapshot["query_embedding_ready"] = True

    db: Session = SessionLocal()
    try:
        tier_rows = {
            "episode": [
                _episode_summary_to_dict(row)
                for row in db.query(EpisodeSummary)
                .filter(EpisodeSummary.chat_session_id == sid)
                .all()
            ],
            "chapter": [
                _chapter_summary_to_dict(row)
                for row in db.query(ChapterSummary)
                .filter(ChapterSummary.chat_session_id == sid)
                .all()
            ],
            "arc": [
                _arc_summary_to_dict(row)
                for row in db.query(ArcSummary)
                .filter(ArcSummary.chat_session_id == sid)
                .all()
            ],
            "saga": [
                _saga_digest_to_dict(row)
                for row in db.query(SagaDigest)
                .filter(SagaDigest.chat_session_id == sid)
                .all()
            ],
        }
    except Exception as exc:
        logger.error("preview_summary_ann_candidates 실패: %s", exc)
        return {
            **base_snapshot,
            "status": "degraded",
            "query_embedding_ready": True,
            "reason": "ann_candidate_query_failed",
        }
    finally:
        db.close()

    shadow_started_at = perf_counter()
    for tier, rows in tier_rows.items():
        selected = _select_ann_candidates_q2a(rows, query_vec, tier=tier, top_k=top_k)
        base_snapshot["tier_items"][tier] = selected["items"]
        base_snapshot["tier_counts"][tier] = len(selected["items"])
        base_snapshot["indexed_row_counts"][tier] = int(selected["indexed_rows"])
        base_snapshot["bucket_counts"][tier] = int(selected["bucket_count"])
        base_snapshot["filtered_out_counts"][tier] = int(selected.get("filtered_out_count") or 0)
        base_snapshot["filtered_out_reasons"][tier] = dict(selected.get("filtered_out_reasons") or {})
        if selected.get("rerank_applied"):
            base_snapshot["rerank_applied"] = True
        if selected["fallback_used"]:
            base_snapshot["fallback_tiers"].append(tier)

    merged = _merge_ann_candidates_q2c(base_snapshot["tier_items"])
    base_snapshot["merge_applied"] = bool(merged.get("applied"))
    base_snapshot["merge_policy"] = str(merged.get("policy") or "tier_head_then_rerank_v1")
    base_snapshot["merged_items"] = list(merged.get("items") or [])
    base_snapshot["merged_tier_sequence"] = list(merged.get("tier_sequence") or [])
    base_snapshot["candidate_count"] = sum(base_snapshot["tier_counts"].values())
    base_snapshot["filtered_out_total"] = sum(base_snapshot["filtered_out_counts"].values())
    shadow_selection_latency_ms = round((perf_counter() - shadow_started_at) * 1000.0, 3)
    try:
        base_snapshot["benchmark"] = _build_ann_benchmark_q2d(
            base_snapshot,
            tier_rows,
            query_vec,
            top_k=top_k,
            shadow_selection_latency_ms=shadow_selection_latency_ms,
        )
    except TimeoutError:
        base_snapshot["benchmark"] = _empty_ann_benchmark_q2d(status="timeout", reason="benchmark_timeout")
        base_snapshot["benchmark"]["shadow_selection_latency_ms"] = shadow_selection_latency_ms
        base_snapshot["benchmark"]["shadow_candidate_count"] = int(base_snapshot.get("candidate_count") or 0)
        base_snapshot["benchmark"]["shadow_tier_diversity"] = len(set(base_snapshot.get("merged_tier_sequence") or []))
    except Exception as exc:
        logger.warning("preview_summary_ann_candidates benchmark 실패: %s", exc)
        base_snapshot["benchmark"] = _empty_ann_benchmark_q2d(status="degraded", reason="benchmark_failed")
        base_snapshot["benchmark"]["shadow_selection_latency_ms"] = shadow_selection_latency_ms
        base_snapshot["benchmark"]["shadow_candidate_count"] = int(base_snapshot.get("candidate_count") or 0)
        base_snapshot["benchmark"]["shadow_tier_diversity"] = len(set(base_snapshot.get("merged_tier_sequence") or []))
    base_snapshot["status"] = "ready" if base_snapshot["candidate_count"] > 0 else "empty"
    return base_snapshot


def search_episode_summaries(
    query: str,
    chat_session_id: str,
    top_k: int = 3,
) -> list[dict]:
    """에피소드 요약을 embedding cosine similarity로 검색한다.

    Args:
        query: 검색 쿼리 텍스트
        chat_session_id: 세션 ID (필수 — 세션 격리)
        top_k: 반환할 최대 결과 수

    Returns:
        score 내림차순으로 정렬된 에피소드 dict 리스트.
        각 dict에 similarity 점수와 source_type: "episode" 태그 포함.
    """
    if not query or not str(query).strip():
        return []

    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    top_k = max(1, min(10, top_k))

    archive_bridge = ArchiveBridge()
    query_vec = archive_bridge._get_embedding(str(query).strip())

    db: Session = SessionLocal()
    try:
        rows = (
            db.query(EpisodeSummary)
            .filter(EpisodeSummary.chat_session_id == sid)
            .all()
        )

        if not rows:
            return []

        turn_values = [row.to_turn for row in rows if isinstance(row.to_turn, int)]
        min_turn = min(turn_values) if turn_values else None
        max_turn = max(turn_values) if turn_values else None
        evidence_cache = _load_direct_evidence_support_cache_ds1i(db, chat_session_id=sid, latest_turn=max_turn)

        scored: list[tuple[float, float, float, float, float, float, bool, str | None, dict, EpisodeSummary]] = []
        for row in rows:
            if not row.embedding_vector:
                continue
            try:
                row_vec = json.loads(row.embedding_vector)
            except Exception:
                continue

            similarity = archive_bridge._cosine(query_vec, row_vec) if query_vec is not None else 0.0
            item_dict = _episode_summary_to_dict(row)
            recency_score = _recency_score_q2b(item_dict, min_turn=min_turn, max_turn=max_turn)
            metadata = _resolve_ann_metadata_q2b(item_dict, tier="episode", min_turn=min_turn, max_turn=max_turn)
            importance_score, relationship_score, world_score = _resolve_ds1d_priority_signals(item_dict, tier="episode")
            promotion = _resolve_direct_evidence_promotion_ds1i(item=item_dict, tier="episode", evidence_cache=evidence_cache)
            gate_passed, retention_applied, retention_reason = _passes_dense_summary_gate_ds1g(
                similarity=similarity,
                importance_score=importance_score,
                relationship_score=relationship_score,
                world_score=world_score,
                metadata=metadata,
            )
            if not gate_passed:
                continue

            dense_priority_score = _compute_dense_summary_priority_score_ds1d(
                similarity=similarity,
                recency_score=recency_score,
                importance_score=importance_score,
                relationship_score=relationship_score,
                world_score=world_score,
                direct_evidence_promotion_score=float(promotion.get("promotion_score") or 0.0),
            )
            scored.append((dense_priority_score, similarity, importance_score, relationship_score, world_score, float(promotion.get("promotion_score") or 0.0), retention_applied, retention_reason, promotion, row))

        # DS-1d: age보다 importance/relationship/world priority를 우선한다.
        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]), reverse=True)
        scored = scored[:top_k]

        results = []
        for dense_priority_score, sim, importance_score, relationship_score, world_score, promotion_score, retention_applied, retention_reason, promotion, row in scored:
            ep = _episode_summary_to_dict(row)
            ep["similarity"] = round(sim, 4)
            ep["dense_priority_score"] = round(dense_priority_score, 4)
            ep["dense_summary_policy_version"] = _DS1D_DENSE_SUMMARY_POLICY_VERSION
            ep["dense_retention_policy_version"] = _DS1G_IMPORTANT_FACT_RETENTION_POLICY_VERSION
            ep["dense_retention_applied"] = bool(retention_applied)
            ep["dense_retention_reason"] = retention_reason
            ep["dense_importance_score"] = round(importance_score, 4)
            ep["dense_relationship_score"] = round(relationship_score, 4)
            ep["dense_world_score"] = round(world_score, 4)
            ep["dense_direct_evidence_promotion_policy_version"] = _DS1I_DIRECT_EVIDENCE_PROMOTION_POLICY_VERSION
            ep["dense_direct_evidence_promotion_score"] = round(promotion_score, 4)
            ep["dense_direct_evidence_promoted_relationship_count"] = int(promotion.get("promoted_relationship_count") or 0)
            ep["dense_direct_evidence_promoted_world_count"] = int(promotion.get("promoted_world_count") or 0)
            ep["dense_direct_evidence_promoted_promise_count"] = int(promotion.get("promoted_promise_count") or 0)
            ep["dense_direct_evidence_promoted_signal_count"] = int(promotion.get("promoted_signal_count") or 0)
            ep["dense_structured_precedence_applied"] = bool(promotion.get("structured_precedence_applied"))
            ep["source_type"] = "episode"
            results.append(ep)

        return results

    except Exception as exc:
        logger.error("search_episode_summaries 실패: %s", exc)
        return []
    finally:
        db.close()


def search_chapter_summaries(
    query: str,
    chat_session_id: str,
    top_k: int = 3,
) -> list[dict]:
    """chapter summary를 embedding cosine similarity로 검색한다.

    Returns:
        score 내림차순으로 정렬된 chapter dict 리스트.
        각 dict에 similarity 점수와 source_type: "chapter" 태그 포함.
        chapter가 전혀 없는 legacy session이면 chat_log keyword fallback 결과를
        source_type: "chat_log_fallback" 태그와 함께 반환한다.
    """
    if not query or not str(query).strip():
        return []

    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    top_k = max(1, min(10, top_k))

    archive_bridge = ArchiveBridge()
    query_vec = archive_bridge._get_embedding(str(query).strip())

    db: Session = SessionLocal()
    try:
        rows = (
            db.query(ChapterSummary)
            .filter(ChapterSummary.chat_session_id == sid)
            .all()
        )

        if not rows:
            query_text = str(query).strip()
            matched_logs = (
                db.query(ChatLog)
                .filter(
                    ChatLog.chat_session_id == sid,
                    ChatLog.content.is_not(None),
                    ChatLog.content.contains(query_text),
                )
                .order_by(desc(ChatLog.turn_index), desc(ChatLog.id))
                .limit(top_k)
                .all()
            )

            fallback_results: list[dict] = []
            query_folded = query_text.casefold()
            for row in matched_logs:
                content = row.content or ""
                content_folded = content.casefold()
                match_index = content_folded.find(query_folded)
                if match_index < 0:
                    excerpt = content[:220].strip()
                else:
                    start = max(0, match_index - 80)
                    end = min(len(content), match_index + len(query_text) + 80)
                    excerpt = content[start:end].strip()
                    if start > 0:
                        excerpt = f"...{excerpt}"
                    if end < len(content):
                        excerpt = f"{excerpt}..."
                excerpt = excerpt.replace("\r", " ").replace("\n", " ")

                fallback_results.append({
                    "id": row.id,
                    "chat_session_id": row.chat_session_id,
                    "from_turn": row.turn_index,
                    "to_turn": row.turn_index,
                    "chapter_index": None,
                    "chapter_title": f"Legacy chat log match @ turn {row.turn_index}",
                    "summary_text": excerpt,
                    "open_loops_json": None,
                    "relationship_changes_json": None,
                    "world_changes_json": None,
                    "callback_candidates_json": None,
                    "resume_text": excerpt,
                    "embedding_vector": None,
                    "embedding_model": None,
                    "created_at": to_kst(row.created_at),
                    "similarity": 1.0,
                    "source_type": "chat_log_fallback",
                    "match_type": "keyword",
                    "role": row.role,
                    "turn_index": row.turn_index,
                    "dense_source_anchor_policy_version": _DS1F_DENSE_SUMMARY_SOURCE_ANCHOR_POLICY_VERSION,
                    "source_record_id": row.id,
                    "source_record_type": "chat_log",
                    "source_turn_range": {
                        "from_turn": row.turn_index,
                        "to_turn": row.turn_index,
                    },
                    "dense_role_split_policy_version": _DS1H_DENSE_SUMMARY_ROLE_SPLIT_POLICY_VERSION,
                    "dense_narrative_text": excerpt,
                    "dense_narrative_usage": "read_only",
                    "dense_structured_usage": "adjudication_retrieval",
                    "dense_structured_payload": {},
                })

            return fallback_results

        turn_values = [row.to_turn for row in rows if isinstance(row.to_turn, int)]
        min_turn = min(turn_values) if turn_values else None
        max_turn = max(turn_values) if turn_values else None
        evidence_cache = _load_direct_evidence_support_cache_ds1i(db, chat_session_id=sid, latest_turn=max_turn)

        scored: list[tuple[float, float, float, float, float, float, bool, str | None, dict, ChapterSummary]] = []
        for row in rows:
            if not row.embedding_vector:
                continue
            try:
                row_vec = json.loads(row.embedding_vector)
            except Exception:
                continue

            similarity = archive_bridge._cosine(query_vec, row_vec) if query_vec is not None else 0.0
            item_dict = _chapter_summary_to_dict(row)
            recency_score = _recency_score_q2b(item_dict, min_turn=min_turn, max_turn=max_turn)
            metadata = _resolve_ann_metadata_q2b(item_dict, tier="chapter", min_turn=min_turn, max_turn=max_turn)
            importance_score, relationship_score, world_score = _resolve_ds1d_priority_signals(item_dict, tier="chapter")
            promotion = _resolve_direct_evidence_promotion_ds1i(item=item_dict, tier="chapter", evidence_cache=evidence_cache)
            gate_passed, retention_applied, retention_reason = _passes_dense_summary_gate_ds1g(
                similarity=similarity,
                importance_score=importance_score,
                relationship_score=relationship_score,
                world_score=world_score,
                metadata=metadata,
            )
            if not gate_passed:
                continue

            dense_priority_score = _compute_dense_summary_priority_score_ds1d(
                similarity=similarity,
                recency_score=recency_score,
                importance_score=importance_score,
                relationship_score=relationship_score,
                world_score=world_score,
                direct_evidence_promotion_score=float(promotion.get("promotion_score") or 0.0),
            )
            scored.append((dense_priority_score, similarity, importance_score, relationship_score, world_score, float(promotion.get("promotion_score") or 0.0), retention_applied, retention_reason, promotion, row))

        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]), reverse=True)
        scored = scored[:top_k]

        results = []
        for dense_priority_score, sim, importance_score, relationship_score, world_score, promotion_score, retention_applied, retention_reason, promotion, row in scored:
            chapter = _chapter_summary_to_dict(row)
            chapter["similarity"] = round(sim, 4)
            chapter["dense_priority_score"] = round(dense_priority_score, 4)
            chapter["dense_summary_policy_version"] = _DS1D_DENSE_SUMMARY_POLICY_VERSION
            chapter["dense_retention_policy_version"] = _DS1G_IMPORTANT_FACT_RETENTION_POLICY_VERSION
            chapter["dense_retention_applied"] = bool(retention_applied)
            chapter["dense_retention_reason"] = retention_reason
            chapter["dense_importance_score"] = round(importance_score, 4)
            chapter["dense_relationship_score"] = round(relationship_score, 4)
            chapter["dense_world_score"] = round(world_score, 4)
            chapter["dense_direct_evidence_promotion_policy_version"] = _DS1I_DIRECT_EVIDENCE_PROMOTION_POLICY_VERSION
            chapter["dense_direct_evidence_promotion_score"] = round(promotion_score, 4)
            chapter["dense_direct_evidence_promoted_relationship_count"] = int(promotion.get("promoted_relationship_count") or 0)
            chapter["dense_direct_evidence_promoted_world_count"] = int(promotion.get("promoted_world_count") or 0)
            chapter["dense_direct_evidence_promoted_promise_count"] = int(promotion.get("promoted_promise_count") or 0)
            chapter["dense_direct_evidence_promoted_signal_count"] = int(promotion.get("promoted_signal_count") or 0)
            chapter["dense_structured_precedence_applied"] = bool(promotion.get("structured_precedence_applied"))
            chapter["source_type"] = "chapter"
            results.append(chapter)

        return results

    except Exception as exc:
        logger.error("search_chapter_summaries 실패: %s", exc)
        return []
    finally:
        db.close()


def search_arc_summaries(
    query: str,
    chat_session_id: str,
    top_k: int = 3,
) -> list[dict]:
    """arc summary를 embedding cosine similarity로 검색한다.

    Returns:
        score 내림차순으로 정렬된 arc dict 리스트.
        각 dict에 similarity 점수와 source_type: "arc" 태그 포함.
    """
    if not query or not str(query).strip():
        return []

    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    top_k = max(1, min(10, top_k))

    archive_bridge = ArchiveBridge()
    query_vec = archive_bridge._get_embedding(str(query).strip())

    db: Session = SessionLocal()
    try:
        rows = (
            db.query(ArcSummary)
            .filter(ArcSummary.chat_session_id == sid)
            .all()
        )

        if not rows:
            return []

        turn_values = [row.to_turn for row in rows if isinstance(row.to_turn, int)]
        min_turn = min(turn_values) if turn_values else None
        max_turn = max(turn_values) if turn_values else None
        evidence_cache = _load_direct_evidence_support_cache_ds1i(db, chat_session_id=sid, latest_turn=max_turn)

        scored: list[tuple[float, float, float, float, float, float, bool, str | None, dict, ArcSummary]] = []
        for row in rows:
            if not row.embedding_vector:
                continue
            try:
                row_vec = json.loads(row.embedding_vector)
            except Exception:
                continue

            similarity = archive_bridge._cosine(query_vec, row_vec) if query_vec is not None else 0.0
            item_dict = _arc_summary_to_dict(row)
            recency_score = _recency_score_q2b(item_dict, min_turn=min_turn, max_turn=max_turn)
            metadata = _resolve_ann_metadata_q2b(item_dict, tier="arc", min_turn=min_turn, max_turn=max_turn)
            importance_score, relationship_score, world_score = _resolve_ds1d_priority_signals(item_dict, tier="arc")
            promotion = _resolve_direct_evidence_promotion_ds1i(item=item_dict, tier="arc", evidence_cache=evidence_cache)
            gate_passed, retention_applied, retention_reason = _passes_dense_summary_gate_ds1g(
                similarity=similarity,
                importance_score=importance_score,
                relationship_score=relationship_score,
                world_score=world_score,
                metadata=metadata,
            )
            if not gate_passed:
                continue

            dense_priority_score = _compute_dense_summary_priority_score_ds1d(
                similarity=similarity,
                recency_score=recency_score,
                importance_score=importance_score,
                relationship_score=relationship_score,
                world_score=world_score,
                direct_evidence_promotion_score=float(promotion.get("promotion_score") or 0.0),
            )
            scored.append((dense_priority_score, similarity, importance_score, relationship_score, world_score, float(promotion.get("promotion_score") or 0.0), retention_applied, retention_reason, promotion, row))

        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]), reverse=True)
        scored = scored[:top_k]

        results = []
        for dense_priority_score, sim, importance_score, relationship_score, world_score, promotion_score, retention_applied, retention_reason, promotion, row in scored:
            arc = _arc_summary_to_dict(row)
            arc["similarity"] = round(sim, 4)
            arc["dense_priority_score"] = round(dense_priority_score, 4)
            arc["dense_summary_policy_version"] = _DS1D_DENSE_SUMMARY_POLICY_VERSION
            arc["dense_retention_policy_version"] = _DS1G_IMPORTANT_FACT_RETENTION_POLICY_VERSION
            arc["dense_retention_applied"] = bool(retention_applied)
            arc["dense_retention_reason"] = retention_reason
            arc["dense_importance_score"] = round(importance_score, 4)
            arc["dense_relationship_score"] = round(relationship_score, 4)
            arc["dense_world_score"] = round(world_score, 4)
            arc["dense_direct_evidence_promotion_policy_version"] = _DS1I_DIRECT_EVIDENCE_PROMOTION_POLICY_VERSION
            arc["dense_direct_evidence_promotion_score"] = round(promotion_score, 4)
            arc["dense_direct_evidence_promoted_relationship_count"] = int(promotion.get("promoted_relationship_count") or 0)
            arc["dense_direct_evidence_promoted_world_count"] = int(promotion.get("promoted_world_count") or 0)
            arc["dense_direct_evidence_promoted_promise_count"] = int(promotion.get("promoted_promise_count") or 0)
            arc["dense_direct_evidence_promoted_signal_count"] = int(promotion.get("promoted_signal_count") or 0)
            arc["dense_structured_precedence_applied"] = bool(promotion.get("structured_precedence_applied"))
            arc["source_type"] = "arc"
            results.append(arc)

        return results

    except Exception as exc:
        logger.error("search_arc_summaries 실패: %s", exc)
        return []
    finally:
        db.close()


def search_saga_digests(
    query: str,
    chat_session_id: str,
    top_k: int = 3,
) -> list[dict]:
    """saga digest를 embedding cosine similarity로 검색한다.

    Returns:
        score 내림차순으로 정렬된 saga dict 리스트.
        각 dict에 similarity 점수와 source_type: "saga" 태그 포함.
    """
    if not query or not str(query).strip():
        return []

    sid = _safe_session_id(chat_session_id)
    if not sid:
        return []

    top_k = max(1, min(10, top_k))

    archive_bridge = ArchiveBridge()
    query_vec = archive_bridge._get_embedding(str(query).strip())

    db: Session = SessionLocal()
    try:
        rows = (
            db.query(SagaDigest)
            .filter(SagaDigest.chat_session_id == sid)
            .all()
        )

        if not rows:
            return []

        turn_values = [row.to_turn for row in rows if isinstance(row.to_turn, int)]
        min_turn = min(turn_values) if turn_values else None
        max_turn = max(turn_values) if turn_values else None
        evidence_cache = _load_direct_evidence_support_cache_ds1i(db, chat_session_id=sid, latest_turn=max_turn)

        scored: list[tuple[float, float, float, float, float, float, bool, str | None, dict, SagaDigest]] = []
        for row in rows:
            if not row.embedding_vector:
                continue
            try:
                row_vec = json.loads(row.embedding_vector)
            except Exception:
                continue

            similarity = archive_bridge._cosine(query_vec, row_vec) if query_vec is not None else 0.0
            item_dict = _saga_digest_to_dict(row)
            recency_score = _recency_score_q2b(item_dict, min_turn=min_turn, max_turn=max_turn)
            metadata = _resolve_ann_metadata_q2b(item_dict, tier="saga", min_turn=min_turn, max_turn=max_turn)
            importance_score, relationship_score, world_score = _resolve_ds1d_priority_signals(item_dict, tier="saga")
            promotion = _resolve_direct_evidence_promotion_ds1i(item=item_dict, tier="saga", evidence_cache=evidence_cache)
            gate_passed, retention_applied, retention_reason = _passes_dense_summary_gate_ds1g(
                similarity=similarity,
                importance_score=importance_score,
                relationship_score=relationship_score,
                world_score=world_score,
                metadata=metadata,
            )
            if not gate_passed:
                continue

            dense_priority_score = _compute_dense_summary_priority_score_ds1d(
                similarity=similarity,
                recency_score=recency_score,
                importance_score=importance_score,
                relationship_score=relationship_score,
                world_score=world_score,
                direct_evidence_promotion_score=float(promotion.get("promotion_score") or 0.0),
            )
            scored.append((dense_priority_score, similarity, importance_score, relationship_score, world_score, float(promotion.get("promotion_score") or 0.0), retention_applied, retention_reason, promotion, row))

        scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]), reverse=True)
        scored = scored[:top_k]

        results = []
        for dense_priority_score, sim, importance_score, relationship_score, world_score, promotion_score, retention_applied, retention_reason, promotion, row in scored:
            saga = _saga_digest_to_dict(row)
            saga["similarity"] = round(sim, 4)
            saga["dense_priority_score"] = round(dense_priority_score, 4)
            saga["dense_summary_policy_version"] = _DS1D_DENSE_SUMMARY_POLICY_VERSION
            saga["dense_retention_policy_version"] = _DS1G_IMPORTANT_FACT_RETENTION_POLICY_VERSION
            saga["dense_retention_applied"] = bool(retention_applied)
            saga["dense_retention_reason"] = retention_reason
            saga["dense_importance_score"] = round(importance_score, 4)
            saga["dense_relationship_score"] = round(relationship_score, 4)
            saga["dense_world_score"] = round(world_score, 4)
            saga["dense_direct_evidence_promotion_policy_version"] = _DS1I_DIRECT_EVIDENCE_PROMOTION_POLICY_VERSION
            saga["dense_direct_evidence_promotion_score"] = round(promotion_score, 4)
            saga["dense_direct_evidence_promoted_relationship_count"] = int(promotion.get("promoted_relationship_count") or 0)
            saga["dense_direct_evidence_promoted_world_count"] = int(promotion.get("promoted_world_count") or 0)
            saga["dense_direct_evidence_promoted_promise_count"] = int(promotion.get("promoted_promise_count") or 0)
            saga["dense_direct_evidence_promoted_signal_count"] = int(promotion.get("promoted_signal_count") or 0)
            saga["dense_structured_precedence_applied"] = bool(promotion.get("structured_precedence_applied"))
            saga["source_type"] = "saga"
            results.append(saga)

        return results

    except Exception as exc:
        logger.error("search_saga_digests 실패: %s", exc)
        return []
    finally:
        db.close()


# ------------------------------------------------------------------ #
#  Phase 3-4 — 에피소드 병합
# ------------------------------------------------------------------ #

def merge_episodes(
    *,
    chat_session_id: str,
    episode_ids: list[int],
    force: bool = False,
) -> dict:
    """여러 에피소드를 하나의 큰 구간으로 병합한다.

    선택된 에피소드들의 min(from_turn)~max(to_turn)로 새 에피소드를 생성한다.
    기존 에피소드는 삭제하지 않는다.
    """
    sid = _safe_session_id(chat_session_id)
    if not sid:
        return {"status": "error", "reason": "invalid chat_session_id"}

    if not episode_ids or len(episode_ids) < 2:
        return {"status": "error", "reason": "at_least_2_episodes_required"}

    # 에피소드 조회 + 세션 검증
    db: Session = SessionLocal()
    try:
        rows = (
            db.query(EpisodeSummary)
            .filter(
                EpisodeSummary.id.in_(episode_ids),
                EpisodeSummary.chat_session_id == sid,
            )
            .all()
        )
    finally:
        db.close()

    if len(rows) != len(episode_ids):
        found_ids = {r.id for r in rows}
        missing = [eid for eid in episode_ids if eid not in found_ids]
        return {"status": "error", "reason": "episodes_not_found", "missing_ids": missing}

    # 병합 구간 계산
    merged_from = min(r.from_turn for r in rows)
    merged_to = max(r.to_turn for r in rows)

    # 새 에피소드 생성 (기존 generate_episode_summary 재사용)
    result = generate_episode_summary(
        chat_session_id=sid,
        from_turn=merged_from,
        to_turn=merged_to,
        force=force,
    )

    result["merged_from_episodes"] = episode_ids
    result["merged_range"] = {"from_turn": merged_from, "to_turn": merged_to}

    return result
