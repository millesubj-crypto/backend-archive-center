"""Step 11 EA-1b direct evidence state contract.

This module defines the local, repository-specific state buckets used by the
direct evidence archive. It is intentionally small and does not copy any
external naming, code paths, or constants beyond the conceptual state split
documented in Step 11.
"""

from __future__ import annotations

import json
from collections.abc import Iterable


DIRECT_EVIDENCE_ARCHIVE_STATES = (
    "pending_capture",
    "verified_direct",
    "previous_archive",
    "repair_queue",
)

DIRECT_EVIDENCE_CAPTURE_VERIFICATIONS = (
    "pending",
    "verified",
    "rejected",
    "needs_review",
)

DIRECT_EVIDENCE_COMMITTED_GATES = (
    "finalize",
    "recovery",
    "manual",
)

# EA-1d: deleted turn tombstone는 일정 turn window 동안 보존한다.
DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS = 240

# EA-1l: 중요도/lineage 기반 retention policy.
DIRECT_EVIDENCE_RETENTION_POLICY_VERSION = "ea1l.v1"
DIRECT_EVIDENCE_RETENTION_IMPORTANCE_TIERS = (
    "critical",
    "high",
    "medium",
    "low",
)
DIRECT_EVIDENCE_RETENTION_WINDOWS_TURNS = {
    "direct_evidence": {
        "critical": 720,
        "high": 480,
        "medium": 320,
        "low": 180,
    },
    "previous_archive": {
        "critical": 540,
        "high": 360,
        "medium": 240,
        "low": 160,
    },
    "tombstone": {
        "critical": 480,
        "high": 320,
        "medium": DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS,
        "low": DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS,
    },
}


def normalize_archive_state(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in DIRECT_EVIDENCE_ARCHIVE_STATES:
        return normalized
    return "pending_capture"


def normalize_capture_verification(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in DIRECT_EVIDENCE_CAPTURE_VERIFICATIONS:
        return normalized
    return "pending"


def normalize_committed_gate(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in DIRECT_EVIDENCE_COMMITTED_GATES:
        return normalized
    return None


def resolve_committed_gate(
    capture_verification: str | None,
    repair_needed: bool | None = False,
    committed_gate: str | None = None,
) -> str | None:
    normalized_gate = normalize_committed_gate(committed_gate)
    if normalized_gate is not None:
        return normalized_gate

    normalized_verification = normalize_capture_verification(capture_verification)
    if repair_needed or normalized_verification in {"rejected", "needs_review"}:
        return "recovery"
    if normalized_verification == "verified":
        return "finalize"
    return None


def resolve_archive_bucket(
    archive_state: str | None,
    capture_verification: str | None,
    repair_needed: bool | None = False,
) -> str:
    normalized_state = normalize_archive_state(archive_state)
    normalized_verification = normalize_capture_verification(capture_verification)
    if repair_needed or normalized_verification in {"rejected", "needs_review"}:
        return "repair_queue"
    return normalized_state


def _normalize_importance_tier(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in DIRECT_EVIDENCE_RETENTION_IMPORTANCE_TIERS:
        return normalized
    return None


def _coerce_lineage_dict(lineage: dict | str | None) -> dict:
    if isinstance(lineage, dict):
        return lineage
    if isinstance(lineage, str):
        raw = lineage.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def resolve_retention_importance_tier(
    *,
    archive_state: str | None,
    capture_verification: str | None,
    repair_needed: bool | None = False,
    committed_gate: str | None = None,
    tombstoned: bool | None = False,
    lineage: dict | str | None = None,
) -> str:
    """EA-1l 중요도 tier를 결정한다.

    우선순위:
    1) lineage explicit tier
    2) lineage critical marker
    3) manual gate 기반 high
    4) archive/capture 상태 기반 fallback
    """
    lineage_map = _coerce_lineage_dict(lineage)
    for key in ("importance_tier", "retention_tier", "importance"):
        explicit_tier = _normalize_importance_tier(lineage_map.get(key))
        if explicit_tier:
            return explicit_tier

    if any(bool(lineage_map.get(marker)) for marker in (
        "force_retain",
        "high_impact",
        "user_confirmation_candidate",
        "manual_review_required",
    )):
        return "critical"

    normalized_archive_state = normalize_archive_state(archive_state)
    normalized_verification = normalize_capture_verification(capture_verification)
    normalized_gate = normalize_committed_gate(committed_gate)

    if bool(tombstoned) and normalized_gate == "manual":
        return "high"

    if (
        normalized_archive_state == "verified_direct"
        and normalized_verification == "verified"
        and not bool(repair_needed)
    ):
        if normalized_gate == "manual":
            return "high"
        return "medium"

    if normalized_archive_state == "previous_archive":
        if normalized_verification == "verified":
            return "medium"
        return "low"

    return "low"


def resolve_retention_ttl_turns(
    *,
    archive_state: str | None,
    capture_verification: str | None,
    repair_needed: bool | None = False,
    committed_gate: str | None = None,
    tombstoned: bool | None = False,
    lineage: dict | str | None = None,
) -> int:
    tier = resolve_retention_importance_tier(
        archive_state=archive_state,
        capture_verification=capture_verification,
        repair_needed=repair_needed,
        committed_gate=committed_gate,
        tombstoned=tombstoned,
        lineage=lineage,
    )
    normalized_archive_state = normalize_archive_state(archive_state)

    if bool(tombstoned):
        group = "tombstone"
    elif normalized_archive_state == "previous_archive":
        group = "previous_archive"
    else:
        group = "direct_evidence"

    tier_windows = DIRECT_EVIDENCE_RETENTION_WINDOWS_TURNS.get(group) or {}
    ttl = tier_windows.get(tier)
    if isinstance(ttl, int):
        return max(ttl, 0)
    return max(DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS, 0)


def is_retention_expired(
    *,
    source_turn_end: int | None,
    latest_turn_index: int | None,
    archive_state: str | None,
    capture_verification: str | None,
    repair_needed: bool | None = False,
    committed_gate: str | None = None,
    tombstoned: bool | None = False,
    lineage: dict | str | None = None,
) -> bool:
    """EA-1l: 레코드의 retention window 만료 여부를 판정한다."""
    if latest_turn_index is None or source_turn_end is None:
        return False

    try:
        latest_turn = int(latest_turn_index)
        end_turn = int(source_turn_end)
    except Exception:
        return False

    if latest_turn < end_turn:
        return False

    ttl_turns = resolve_retention_ttl_turns(
        archive_state=archive_state,
        capture_verification=capture_verification,
        repair_needed=repair_needed,
        committed_gate=committed_gate,
        tombstoned=tombstoned,
        lineage=lineage,
    )
    return (latest_turn - end_turn) > ttl_turns


def is_within_retention_window(
    *,
    source_turn_end: int | None,
    latest_turn_index: int | None,
    archive_state: str | None,
    capture_verification: str | None,
    repair_needed: bool | None = False,
    committed_gate: str | None = None,
    tombstoned: bool | None = False,
    lineage: dict | str | None = None,
) -> bool:
    return not is_retention_expired(
        source_turn_end=source_turn_end,
        latest_turn_index=latest_turn_index,
        archive_state=archive_state,
        capture_verification=capture_verification,
        repair_needed=repair_needed,
        committed_gate=committed_gate,
        tombstoned=tombstoned,
        lineage=lineage,
    )


def is_current_truth_excluded(
    *,
    tombstoned: bool | None = False,
    superseded_by_id: int | None = None,
) -> bool:
    """현재 진실 판정에서 제외해야 하는 stale/invalid record 여부."""
    return bool(tombstoned) or superseded_by_id is not None


def is_deleted_turn_tombstone_retained(
    *,
    tombstoned: bool | None,
    source_turn_end: int | None,
    latest_turn_index: int | None,
    retention_window_turns: int = DIRECT_EVIDENCE_DELETED_TURN_TOMBSTONE_RETENTION_WINDOW_TURNS,
) -> bool:
    """deleted turn tombstone가 retention window 안에 있는지 확인한다."""
    if not bool(tombstoned):
        return False

    # latest/source turn을 모르는 경우는 보수적으로 retained로 취급한다.
    if latest_turn_index is None or source_turn_end is None:
        return True

    try:
        latest_turn = int(latest_turn_index)
        end_turn = int(source_turn_end)
        window = max(int(retention_window_turns), 0)
    except Exception:
        return True

    return (latest_turn - end_turn) <= window


def is_gate_approved_current_truth(
    *,
    archive_state: str | None,
    capture_verification: str | None,
    repair_needed: bool | None = False,
    committed_gate: str | None = None,
    tombstoned: bool | None = False,
    superseded_by_id: int | None = None,
) -> bool:
    """gate 승인된 current truth 후보인지 판정한다."""
    if is_current_truth_excluded(tombstoned=tombstoned, superseded_by_id=superseded_by_id):
        return False

    normalized_capture_verification = normalize_capture_verification(capture_verification)
    normalized_archive_state = normalize_archive_state(archive_state)
    normalized_committed_gate = resolve_committed_gate(
        normalized_capture_verification,
        repair_needed,
        committed_gate,
    )

    return (
        normalized_archive_state == "verified_direct"
        and normalized_capture_verification == "verified"
        and not bool(repair_needed)
        and normalized_committed_gate in {"finalize", "manual"}
    )


def is_gate_blocked_signal(
    *,
    capture_verification: str | None,
    archive_state: str | None,
    repair_needed: bool | None = False,
    committed_gate: str | None = None,
    tombstoned: bool | None = False,
    superseded_by_id: int | None = None,
) -> bool:
    """EA-1c canonical guard용 blocked signal 판정.

    superseded record는 stale exclusion 대상이지만 canonical guard를 과잉 차단하지 않기 위해
    blocked signal로는 보지 않는다.
    """
    if bool(tombstoned):
        return True
    if superseded_by_id is not None:
        return False

    normalized_capture_verification = normalize_capture_verification(capture_verification)
    normalized_archive_state = normalize_archive_state(archive_state)
    normalized_committed_gate = resolve_committed_gate(
        normalized_capture_verification,
        repair_needed,
        committed_gate,
    )

    return (
        normalized_archive_state == "repair_queue"
        or normalized_capture_verification in {"needs_review", "rejected"}
        or bool(repair_needed)
        or normalized_committed_gate == "recovery"
    )


def build_state_counts(rows: Iterable[object]) -> dict[str, int]:
    counts = {state: 0 for state in DIRECT_EVIDENCE_ARCHIVE_STATES}
    for row in rows:
        bucket = resolve_archive_bucket(
            getattr(row, "archive_state", None),
            getattr(row, "capture_verification", None),
            getattr(row, "repair_needed", False),
        )
        counts[bucket] += 1
    return counts