from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func

from backend.chroma_shadow_contracts import (
    ChromaShadowAdoptionGateRequest,
    ChromaShadowBackfillBatchRequest,
    ChromaShadowBackfillDryRunRequest,
    ChromaShadowFallbackRunbookRequest,
    ChromaShadowHealthProbeRequest,
    ChromaShadowRebuildDrillRequest,
    ChromaShadowReembedAuditRequest,
    ChromaShadowReleaseHygieneRequest,
    ChromaShadowVisibilityGuardRequest,
)
from backend.database import SessionLocal
from backend.embedding_provenance import assess_memory_embedding_state as _assess_memory_embedding_state_c17
from backend.models import (
    ArcSummary,
    ChapterSummary,
    EpisodeSummary,
    Memory,
    SagaDigest,
)
from backend.retrieval_document_builder import _build_retrieval_document_q1a, _safe_json_loads_q1a
from backend.retrieval_contracts import _RETRIEVAL_DOCUMENT_SCHEMA_Q1A
from backend.retrieval_index import retrieval_index_registry
from backend.settings import settings as _settings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Layer 1: Config / Path Resolution                                   #
# ------------------------------------------------------------------ #

_BACKEND_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _resolve_chroma_shadow_persist_dir_c17() -> Path:
    raw = " ".join(str(_settings.CHROMA_SHADOW_PERSIST_DIRECTORY or ".chroma_shadow").split()).strip()
    target = Path(raw or ".chroma_shadow").expanduser()
    if not target.is_absolute():
        target = (_BACKEND_PACKAGE_ROOT / target).resolve()
    return target


def _resolve_chroma_shadow_collection_name_c17() -> str:
    raw = " ".join(str(_settings.CHROMA_SHADOW_COLLECTION_NAME or "archive_center_shadow").split()).strip()
    return raw or "archive_center_shadow"


def _resolve_chroma_shadow_contract_path_c17() -> Path:
    return _resolve_chroma_shadow_persist_dir_c17() / "bootstrap_contract.json"


# ------------------------------------------------------------------ #
#  Layer 2: Diagnostics / Identity / Dependency                        #
# ------------------------------------------------------------------ #

def _estimate_directory_size_bytes_c17(target: Path) -> int:
    if not target.exists() or not target.is_dir():
        return 0
    total = 0
    try:
        for item in target.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _resolve_chroma_shadow_dependency_c17() -> dict:
    try:
        import chromadb as _chromadb  # type: ignore

        return {
            "package": "chromadb",
            "available": True,
            "version": str(getattr(_chromadb, "__version__", "unknown") or "unknown"),
            "detail": None,
        }
    except Exception as exc:
        return {
            "package": "chromadb",
            "available": False,
            "version": None,
            "detail": exc.__class__.__name__,
        }


def _collect_chroma_shadow_sqlite_counts_c17() -> dict:
    db = SessionLocal()
    try:
        counts: dict[str, int] = {}
        warnings: list[str] = []
        targets = (
            ("memory", Memory),
            ("episode", EpisodeSummary),
            ("chapter", ChapterSummary),
            ("arc", ArcSummary),
            ("saga", SagaDigest),
        )
        for tier_name, model in targets:
            try:
                counts[tier_name] = int(db.query(func.count(model.id)).scalar() or 0)
            except Exception as exc:
                counts[tier_name] = 0
                warnings.append(f"{tier_name}_count_unavailable:{exc.__class__.__name__}")
        return {
            "counts": counts,
            "warnings": warnings,
        }
    finally:
        db.close()


def _build_chroma_shadow_embedder_identity_c17() -> dict:
    current_embedding_model = " ".join(str(_settings.PROJECT_EMBEDDING_MODEL or "").split()).strip()
    current_embedding_endpoint = " ".join(str(_settings.PROJECT_EMBEDDING_ENDPOINT or "").split()).strip()
    current_embedding_provider = " ".join(str(_settings.PROJECT_EMBEDDING_PROVIDER or "").split()).strip()
    return {
        "provider": current_embedding_provider or None,
        "model": current_embedding_model or None,
        "endpoint": current_embedding_endpoint or None,
    }


def _install_chroma_telemetry_noop_c17() -> None:
    try:
        from chromadb.telemetry.product import posthog as _chroma_posthog  # type: ignore

        posthog_client = getattr(_chroma_posthog, "posthog", None)
        if posthog_client is not None and not getattr(posthog_client, "_archive_center_noop_c17", False):
            def _noop_capture(*args, **kwargs):
                return None

            posthog_client.capture = _noop_capture
            setattr(posthog_client, "_archive_center_noop_c17", True)
    except Exception:
        pass
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


# ------------------------------------------------------------------ #
#  Layer 3: Collection Open / Normalization / Serialization            #
# ------------------------------------------------------------------ #

def _open_chroma_shadow_collection_c17() -> dict:
    persist_dir = _resolve_chroma_shadow_persist_dir_c17()
    collection_name = _resolve_chroma_shadow_collection_name_c17()
    persist_dir.mkdir(parents=True, exist_ok=True)

    import chromadb as _chromadb  # type: ignore

    _install_chroma_telemetry_noop_c17()

    try:
        from chromadb.config import Settings as _ChromaSettings  # type: ignore

        client = _chromadb.PersistentClient(
            path=str(persist_dir),
            settings=_ChromaSettings(anonymized_telemetry=False),
        )
    except Exception:
        client = _chromadb.PersistentClient(path=str(persist_dir))

    collection_preexisting = False
    try:
        existing = client.list_collections()
        existing_names = []
        for item in existing:
            if isinstance(item, str):
                existing_names.append(item)
            else:
                existing_names.append(str(getattr(item, "name", "") or ""))
        collection_preexisting = collection_name in existing_names
    except Exception:
        collection_preexisting = False

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "contract_version": "c17.bootstrap.v1",
            "schema_version": _RETRIEVAL_DOCUMENT_SCHEMA_Q1A["version"],
            "truth_authority": "sqlite",
            "session_partitioning": "session_partitioned",
        },
    )
    return {
        "client": client,
        "collection": collection,
        "collection_name": collection_name,
        "collection_preexisting": collection_preexisting,
        "persist_dir": persist_dir,
    }


def _create_chroma_shadow_collection_with_client_c17(client, *, collection_name: str):
    return client.get_or_create_collection(
        name=collection_name,
        metadata={
            "contract_version": "c17.bootstrap.v1",
            "schema_version": _RETRIEVAL_DOCUMENT_SCHEMA_Q1A["version"],
            "truth_authority": "sqlite",
            "session_partitioning": "session_partitioned",
        },
    )


def _normalize_chroma_shadow_sample_limit_c17(value: object) -> int:
    try:
        normalized = int(value)
    except Exception:
        normalized = 2
    return max(1, min(normalized, 5))


def _normalize_chroma_shadow_batch_size_c17(value: object) -> int:
    try:
        normalized = int(value)
    except Exception:
        normalized = 25
    return max(1, min(normalized, 100))


def _normalize_chroma_shadow_queue_limit_c17(value: object) -> int:
    try:
        normalized = int(value)
    except Exception:
        normalized = 25
    return max(1, min(normalized, 100))


def _serialize_chroma_shadow_created_at_c17(raw: object) -> str | None:
    if raw is None:
        return None
    if hasattr(raw, "isoformat"):
        try:
            return raw.isoformat()
        except Exception:
            return str(raw)
    return str(raw)


def _row_to_chroma_shadow_item_c17(tier: str, row) -> dict:
    created_at = _serialize_chroma_shadow_created_at_c17(getattr(row, "created_at", None))
    if tier == "memory":
        return {
            "id": row.id,
            "chat_session_id": row.chat_session_id,
            "turn_index": row.turn_index,
            "summary_json": row.summary_json,
            "importance": row.importance,
            "archive_wing": getattr(row, "archive_wing", None),
            "archive_room": getattr(row, "archive_room", None),
            "embedding": row.embedding,
            "embedding_model": row.embedding_model,
            "created_at": created_at,
            "source_type": "memory",
        }
    if tier == "episode":
        return {
            "id": row.id,
            "chat_session_id": row.chat_session_id,
            "from_turn": row.from_turn,
            "to_turn": row.to_turn,
            "summary_text": row.summary_text,
            "key_entities": row.key_entities,
            "key_events": row.key_events,
            "open_loops_json": row.open_loops_json,
            "relationship_changes_json": row.relationship_changes_json,
            "embedding_vector": row.embedding_vector,
            "embedding_model": row.embedding_model,
            "created_at": created_at,
        }
    if tier == "chapter":
        return {
            "id": row.id,
            "chat_session_id": row.chat_session_id,
            "from_turn": row.from_turn,
            "to_turn": row.to_turn,
            "chapter_index": row.chapter_index,
            "chapter_title": row.chapter_title,
            "summary_text": row.summary_text,
            "resume_text": row.resume_text,
            "open_loops_json": row.open_loops_json,
            "relationship_changes_json": row.relationship_changes_json,
            "world_changes_json": row.world_changes_json,
            "callback_candidates_json": row.callback_candidates_json,
            "embedding_vector": row.embedding_vector,
            "embedding_model": row.embedding_model,
            "created_at": created_at,
        }
    if tier == "arc":
        return {
            "id": row.id,
            "chat_session_id": row.chat_session_id,
            "from_turn": row.from_turn,
            "to_turn": row.to_turn,
            "arc_index": row.arc_index,
            "arc_name": row.arc_name,
            "arc_status": row.arc_status,
            "core_conflict": row.core_conflict,
            "arc_resume_text": row.arc_resume_text,
            "active_promises_json": row.active_promises_json,
            "unresolved_debts_json": row.unresolved_debts_json,
            "resolved_payoffs_json": row.resolved_payoffs_json,
            "callback_candidates_json": row.callback_candidates_json,
            "irreversible_turns_json": row.irreversible_turns_json,
            "callback_debts_json": row.callback_debts_json,
            "relationship_pivots_json": row.relationship_pivots_json,
            "future_payoff_candidates_json": row.future_payoff_candidates_json,
            "embedding_vector": row.embedding_vector,
            "embedding_model": row.embedding_model,
            "created_at": created_at,
        }
    if tier == "saga":
        return {
            "id": row.id,
            "chat_session_id": row.chat_session_id,
            "from_turn": row.from_turn,
            "to_turn": row.to_turn,
            "era_label": row.era_label,
            "saga_summary": row.saga_summary,
            "resume_pack_text": row.resume_pack_text,
            "persistent_facts_json": row.persistent_facts_json,
            "never_drop_candidates_json": row.never_drop_candidates_json,
            "embedding_vector": row.embedding_vector,
            "embedding_model": row.embedding_model,
            "created_at": created_at,
        }
    return {}


def _resolve_chroma_shadow_tier_model_c17(tier: str):
    mapping = {
        "memory": Memory,
        "episode": EpisodeSummary,
        "chapter": ChapterSummary,
        "arc": ArcSummary,
        "saga": SagaDigest,
    }
    return mapping.get(str(tier or "").strip().lower())


def _normalize_chroma_shadow_checkpoint_map_c17(raw: object) -> dict[str, int]:
    checkpoint: dict[str, int] = {}
    if not isinstance(raw, dict):
        return checkpoint
    for tier in ("memory", "episode", "chapter", "arc", "saga"):
        value = raw.get(tier)
        try:
            normalized = int(value)
        except Exception:
            continue
        if normalized > 0:
            checkpoint[tier] = normalized
    return checkpoint


def _normalize_chroma_shadow_retry_rows_c17(raw: object) -> dict[str, list[int]]:
    retry_rows: dict[str, list[int]] = {
        "memory": [],
        "episode": [],
        "chapter": [],
        "arc": [],
        "saga": [],
    }
    if not isinstance(raw, list):
        return retry_rows
    for item in raw:
        if not isinstance(item, dict):
            continue
        tier = str(item.get("tier") or "").strip().lower()
        if tier not in retry_rows:
            continue
        try:
            source_row_id = int(item.get("source_row_id"))
        except Exception:
            continue
        if source_row_id > 0 and source_row_id not in retry_rows[tier]:
            retry_rows[tier].append(source_row_id)
    return retry_rows


# ------------------------------------------------------------------ #
#  Layer 3+: Query / Count / Payload Builder                         #
# ------------------------------------------------------------------ #

def _query_chroma_shadow_backfill_batch_candidates_c17(
    *,
    sid: str,
    batch_size_per_tier: int,
    checkpoint: dict[str, int],
    retry_rows: dict[str, list[int]],
) -> dict:
    db = SessionLocal()
    rows_by_tier = {tier: [] for tier in ("memory", "episode", "chapter", "arc", "saga")}
    warnings: list[str] = []
    checkpoint_start = {tier: int(checkpoint.get(tier) or 0) for tier in rows_by_tier.keys()}
    checkpoint_next = dict(checkpoint_start)
    remaining_counts_by_tier = {tier: 0 for tier in rows_by_tier.keys()}
    selected_counts_by_tier = {tier: 0 for tier in rows_by_tier.keys()}
    try:
        for tier in rows_by_tier.keys():
            model = _resolve_chroma_shadow_tier_model_c17(tier)
            if model is None:
                continue
            start_id = int(checkpoint_start.get(tier) or 0)
            try:
                selected_rows = (
                    db.query(model)
                    .filter(model.chat_session_id == sid)
                    .filter(model.id > start_id)
                    .order_by(model.id.asc())
                    .limit(batch_size_per_tier)
                    .all()
                )
                selected_ids: list[int] = []
                merged_rows = []
                for row in selected_rows:
                    selected_ids.append(int(row.id))
                    merged_rows.append(row)
                for retry_row_id in retry_rows.get(tier) or []:
                    if retry_row_id in selected_ids:
                        continue
                    retry_row = (
                        db.query(model)
                        .filter(model.chat_session_id == sid)
                        .filter(model.id == retry_row_id)
                        .first()
                    )
                    if retry_row is None:
                        warnings.append(f"{tier}_retry_row_missing:{retry_row_id}")
                        continue
                    merged_rows.append(retry_row)
                    selected_ids.append(int(retry_row.id))
                merged_rows.sort(key=lambda row: int(row.id))
                rows_by_tier[tier] = [_row_to_chroma_shadow_item_c17(tier, row) for row in merged_rows]
                selected_counts_by_tier[tier] = len(rows_by_tier[tier])
                if selected_rows:
                    checkpoint_next[tier] = int(selected_rows[-1].id)
                else:
                    checkpoint_next[tier] = start_id
                remaining_counts_by_tier[tier] = int(
                    db.query(func.count(model.id))
                    .filter(model.chat_session_id == sid)
                    .filter(model.id > int(checkpoint_next.get(tier) or 0))
                    .scalar()
                    or 0
                )
            except Exception as exc:
                warnings.append(f"{tier}_query_failed:{exc.__class__.__name__}")
                remaining_counts_by_tier[tier] = 0
        return {
            "rows_by_tier": rows_by_tier,
            "checkpoint_next": checkpoint_next,
            "remaining_counts_by_tier": remaining_counts_by_tier,
            "selected_counts_by_tier": selected_counts_by_tier,
            "warnings": warnings,
        }
    finally:
        db.close()


# ------------------------------------------------------------------ #
#  Layer 4: Query / Count / Payload / Incident Builders              #
# ------------------------------------------------------------------ #
def _query_chroma_shadow_backfill_candidates_c17(*, sid: str, sample_limit_per_tier: int) -> dict:
    db = SessionLocal()
    rows_by_tier = {
        "memory": [],
        "episode": [],
        "chapter": [],
        "arc": [],
        "saga": [],
    }
    warnings: list[str] = []
    try:
        queries = (
            ("memory", lambda: db.query(Memory).filter(Memory.chat_session_id == sid).order_by(Memory.turn_index.desc(), Memory.id.desc()).limit(sample_limit_per_tier).all()),
            ("episode", lambda: db.query(EpisodeSummary).filter(EpisodeSummary.chat_session_id == sid).order_by(EpisodeSummary.to_turn.desc(), EpisodeSummary.id.desc()).limit(sample_limit_per_tier).all()),
            ("chapter", lambda: db.query(ChapterSummary).filter(ChapterSummary.chat_session_id == sid).order_by(ChapterSummary.to_turn.desc(), ChapterSummary.id.desc()).limit(sample_limit_per_tier).all()),
            ("arc", lambda: db.query(ArcSummary).filter(ArcSummary.chat_session_id == sid).order_by(ArcSummary.to_turn.desc(), ArcSummary.id.desc()).limit(sample_limit_per_tier).all()),
            ("saga", lambda: db.query(SagaDigest).filter(SagaDigest.chat_session_id == sid).order_by(SagaDigest.to_turn.desc(), SagaDigest.id.desc()).limit(sample_limit_per_tier).all()),
        )
        for tier, runner in queries:
            try:
                rows_by_tier[tier] = [_row_to_chroma_shadow_item_c17(tier, row) for row in runner()]
            except Exception as exc:
                rows_by_tier[tier] = []
                warnings.append(f"{tier}_query_unavailable:{exc.__class__.__name__}")
        return {
            "rows_by_tier": rows_by_tier,
            "warnings": warnings,
        }
    finally:
        db.close()




def _extract_chroma_shadow_embedding_c17(item: dict) -> list[float] | None:
    raw = item.get("embedding")
    if raw in (None, ""):
        raw = item.get("embedding_vector")
    parsed = _safe_json_loads_q1a(raw)
    if not isinstance(parsed, list) or not parsed:
        return None
    try:
        return [float(value) for value in parsed]
    except Exception:
        return None




def _build_chroma_shadow_reembed_invalidation_rules_c17() -> dict:
    return {
        "project_model_unset": {
            "recommended_action": "set_project_embedding_model_before_reembed",
            "invalidation_rule": "hold_shadow_reembed_decision_until_project_model_is_set",
        },
        "missing_embedding_and_model": {
            "recommended_action": "skip_shadow_upsert_until_reembed",
            "invalidation_rule": "no_valid_vector_present_do_not_refresh_shadow_document",
        },
        "missing_embedding_vector": {
            "recommended_action": "skip_shadow_upsert_until_reembed",
            "invalidation_rule": "stored_model_without_vector_cannot_refresh_shadow_document",
        },
        "missing_embedding_model": {
            "recommended_action": "invalidate_existing_chroma_document_then_reembed_source",
            "invalidation_rule": "vector_exists_but_model_provenance_missing_treat_shadow_copy_as_stale",
        },
        "model_mismatch": {
            "recommended_action": "invalidate_existing_chroma_document_then_reembed_source",
            "invalidation_rule": "stored_vector_model_mismatch_treat_shadow_copy_as_stale",
        },
        "current_model_match": {
            "recommended_action": "keep_existing_shadow_document",
            "invalidation_rule": "no_invalidation_needed",
        },
    }




def _collect_chroma_shadow_eligible_counts_c17(*, sid: str, current_model: str) -> dict:
    db = SessionLocal()
    sqlite_total_counts = {tier: 0 for tier in ("memory", "episode", "chapter", "arc", "saga")}
    eligible_counts = {tier: 0 for tier in ("memory", "episode", "chapter", "arc", "saga")}
    warnings: list[str] = []
    sample_query_embedding: list[float] | None = None
    sample_query_source: dict | None = None
    try:
        for tier in sqlite_total_counts.keys():
            model = _resolve_chroma_shadow_tier_model_c17(tier)
            if model is None:
                continue
            try:
                rows = (
                    db.query(model)
                    .filter(model.chat_session_id == sid)
                    .order_by(model.id.asc())
                    .all()
                )
            except Exception as exc:
                warnings.append(f"{tier}_health_query_unavailable:{exc.__class__.__name__}")
                continue
            sqlite_total_counts[tier] = len(rows)
            for row in rows:
                item = _row_to_chroma_shadow_item_c17(tier, row)
                raw_embedding = item.get("embedding")
                if raw_embedding in (None, ""):
                    raw_embedding = item.get("embedding_vector")
                parsed_embedding = _safe_json_loads_q1a(raw_embedding)
                embedding_value = parsed_embedding if isinstance(parsed_embedding, list) else raw_embedding
                audit_state = _assess_memory_embedding_state_c17(
                    embedding_value=embedding_value,
                    embedding_model=item.get("embedding_model"),
                    current_model=current_model,
                )
                if audit_state.get("needs_reembed") is False:
                    eligible_counts[tier] = int(eligible_counts.get(tier) or 0) + 1
                    if sample_query_embedding is None and isinstance(parsed_embedding, list) and parsed_embedding:
                        sample_query_embedding = [float(value) for value in parsed_embedding]
                        sample_query_source = {
                            "tier": tier,
                            "source_row_id": item.get("id"),
                            "status": audit_state.get("status"),
                        }
        return {
            "sqlite_total_counts": sqlite_total_counts,
            "eligible_counts": eligible_counts,
            "warnings": warnings,
            "sample_query_embedding": sample_query_embedding,
            "sample_query_source": sample_query_source,
        }
    finally:
        db.close()




def _count_chroma_shadow_session_documents_c17(collection, *, sid: str) -> dict:
    session_counts = {tier: 0 for tier in ("memory", "episode", "chapter", "arc", "saga")}
    warnings: list[str] = []
    total_count = None
    try:
        total_count = int(collection.count())
    except Exception as exc:
        warnings.append(f"collection_count_unavailable:{exc.__class__.__name__}")
    try:
        payload = collection.get(include=["metadatas"])
        metadatas = (payload or {}).get("metadatas") or []
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("chat_session_id") or "") != sid:
                continue
            tier = str(metadata.get("tier") or "").strip().lower()
            if tier in session_counts:
                session_counts[tier] = int(session_counts.get(tier) or 0) + 1
    except Exception as exc:
        warnings.append(f"collection_metadata_unavailable:{exc.__class__.__name__}")
    return {
        "session_counts": session_counts,
        "warnings": warnings,
        "collection_document_count": total_count,
    }




def _build_chroma_shadow_fallback_entry_c17(*, preflight_issues: list[str], current_model_ready: bool, divergence_by_tier: dict, sample_query_sanity: dict) -> dict:
    if any(issue in {"chromadb_dependency_unavailable", "persist_directory_not_writable"} for issue in preflight_issues):
        return {
            "decision": "blocked_fail_open_sqlite",
            "sqlite_fail_open": True,
            "automatic_shadow_write_freeze": True,
            "read_only_shadow_mode": True,
            "reason": "shadow_infrastructure_blocked",
        }
    if not current_model_ready:
        return {
            "decision": "hold_shadow_and_fail_open_sqlite",
            "sqlite_fail_open": True,
            "automatic_shadow_write_freeze": False,
            "read_only_shadow_mode": True,
            "reason": "project_model_unset",
        }
    if divergence_by_tier:
        return {
            "decision": "count_divergence_fail_open_sqlite",
            "sqlite_fail_open": True,
            "automatic_shadow_write_freeze": True,
            "read_only_shadow_mode": True,
            "reason": "eligible_vs_shadow_count_divergence",
        }
    if str(sample_query_sanity.get("status") or "") == "fail":
        return {
            "decision": "query_sanity_fail_open_sqlite",
            "sqlite_fail_open": True,
            "automatic_shadow_write_freeze": True,
            "read_only_shadow_mode": True,
            "reason": "sample_query_sanity_failed",
        }
    if str(sample_query_sanity.get("status") or "") == "skip":
        return {
            "decision": "insufficient_shadow_signal_yet",
            "sqlite_fail_open": True,
            "automatic_shadow_write_freeze": False,
            "read_only_shadow_mode": True,
            "reason": "no_eligible_documents_or_shadow_rows",
        }
    return {
        "decision": "shadow_health_green",
        "sqlite_fail_open": False,
        "automatic_shadow_write_freeze": False,
        "read_only_shadow_mode": False,
        "reason": "eligible_counts_match_and_sample_query_passed",
    }




def _build_chroma_shadow_fallback_operator_actions_c17(*, reason: str, health_probe: dict) -> list[dict]:
    current_model_ready = bool(health_probe.get("current_model_ready"))
    divergence_by_tier = health_probe.get("divergence_by_tier") or {}
    sample_query_sanity = health_probe.get("sample_query_sanity") or {}
    preflight_issues = list(health_probe.get("preflight_issues") or [])
    actions: list[dict] = []

    if reason == "shadow_infrastructure_blocked":
        actions.extend([
            {
                "action": "inspect_preflight_blockers",
                "endpoint": "/chroma-shadow/preflight",
                "availability": "landed",
                "required": True,
                "detail": ",".join(preflight_issues) or "shadow_infrastructure_blocked",
            },
            {
                "action": "rerun_bootstrap_after_fix",
                "endpoint": "/chroma-shadow/bootstrap",
                "availability": "landed",
                "required": True,
                "detail": "dependency_or_persist_directory_must_be_restored_before_shadow_recovery",
            },
        ])
    elif reason == "project_model_unset":
        actions.extend([
            {
                "action": "set_project_embedding_model",
                "endpoint": "/chroma-shadow/preflight",
                "availability": "landed",
                "required": True,
                "detail": "embedding_model_missing_must_clear_before_reembed_or_shadow_cutover",
            },
            {
                "action": "rerun_reembed_audit",
                "endpoint": "/chroma-shadow/reembed-audit",
                "availability": "landed",
                "required": True,
                "detail": "recompute_needs_reembed_queue_once_project_model_is_set",
            },
        ])
    elif reason == "eligible_vs_shadow_count_divergence":
        actions.extend([
            {
                "action": "freeze_shadow_writes_and_inspect_divergence",
                "endpoint": "/chroma-shadow/health-probe",
                "availability": "landed",
                "required": True,
                "detail": json.dumps(divergence_by_tier, ensure_ascii=True, sort_keys=True),
            },
            {
                "action": "run_targeted_backfill_batch",
                "endpoint": "/chroma-shadow/backfill-batch",
                "availability": "landed",
                "required": True,
                "detail": "resume_from_checkpoint_or_retry_missing_rows_then_rerun_health_probe",
            },
            {
                "action": "run_reembed_audit_before_retry_if_needed",
                "endpoint": "/chroma-shadow/reembed-audit",
                "availability": "landed",
                "required": False,
                "detail": "use_when_divergence_tracks_stale_or_invalid_vectors",
            },
        ])
    elif reason == "sample_query_sanity_failed":
        actions.extend([
            {
                "action": "capture_query_failure_context",
                "endpoint": "/chroma-shadow/health-probe",
                "availability": "landed",
                "required": True,
                "detail": str(sample_query_sanity.get("detail") or "query_sanity_failed"),
            },
            {
                "action": "run_backfill_dry_run_for_comparison",
                "endpoint": "/chroma-shadow/backfill-dry-run",
                "availability": "landed",
                "required": True,
                "detail": "verify_q1a_document_build_and_embedding_payload_before_rebuild",
            },
            {
                "action": "prepare_rebuild_or_rollback_drill",
                "endpoint": "/chroma-shadow/rebuild-drill",
                "availability": "pending",
                "required": False,
                "detail": "future_slice_c8_handles_collection_reset_and_snapshot_recovery",
            },
        ])
    elif reason == "no_eligible_documents_or_shadow_rows":
        actions.extend([
            {
                "action": "keep_shadow_read_only_until_signal_exists",
                "endpoint": "/chroma-shadow/health-probe",
                "availability": "landed",
                "required": True,
                "detail": "no_eligible_documents_or_shadow_rows_yet",
            },
            {
                "action": "bootstrap_or_backfill_shadow",
                "endpoint": "/chroma-shadow/backfill-batch",
                "availability": "landed",
                "required": False,
                "detail": "once_source_rows_exist_run_backfill_and_repeat_health_probe",
            },
        ])
    else:
        actions.extend([
            {
                "action": "rerun_health_probe_before_cutover",
                "endpoint": "/chroma-shadow/health-probe",
                "availability": "landed",
                "required": True,
                "detail": "keep_sqlite_truth_until_adoption_gate_is_green",
            },
            {
                "action": "evaluate_adoption_gate",
                "endpoint": "/chroma-shadow/adoption-gate",
                "availability": "pending",
                "required": False,
                "detail": "future_slice_c9_decides_limited_cutover_approval",
            },
        ])

    if not current_model_ready and reason != "project_model_unset":
        actions.append({
            "action": "note_project_model_is_not_ready",
            "endpoint": "/chroma-shadow/preflight",
            "availability": "landed",
            "required": True,
            "detail": "embedding_model_missing_blocks_shadow_green_even_if_other_signals_clear",
        })
    return actions




def _build_chroma_shadow_fallback_mode_contract_c17(*, fallback_entry: dict) -> dict:
    decision = str(fallback_entry.get("decision") or "")
    reason = str(fallback_entry.get("reason") or "")
    sqlite_fail_open = bool(fallback_entry.get("sqlite_fail_open"))
    write_freeze = bool(fallback_entry.get("automatic_shadow_write_freeze"))
    read_only_shadow_mode = bool(fallback_entry.get("read_only_shadow_mode"))
    entry_conditions = [
        f"decision={decision}" if decision else "decision=unknown",
        f"reason={reason}" if reason else "reason=unknown",
    ]
    exit_conditions: list[str] = []
    if reason == "shadow_infrastructure_blocked":
        exit_conditions = [
            "chromadb_dependency_and_persist_directory_are_restored",
            "bootstrap_returns_ready_without_blocking_issues",
            "health_probe_no_longer_reports_shadow_infrastructure_blocked",
        ]
    elif reason == "project_model_unset":
        exit_conditions = [
            "embedding_model_missing_is_cleared_in_preflight",
            "reembed_audit_recomputes_queue_with_current_project_model",
            "health_probe_current_model_ready_becomes_true",
        ]
    elif reason == "eligible_vs_shadow_count_divergence":
        exit_conditions = [
            "health_probe_divergence_by_tier_returns_empty",
            "targeted_backfill_or_reembed_remediation_has_been_applied",
            "sample_query_sanity_is_not_fail",
        ]
    elif reason == "sample_query_sanity_failed":
        exit_conditions = [
            "sample_query_sanity_returns_pass",
            "document_build_and_embedding_payload_are_verified",
            "rebuild_or_rollback_drill_is_not_required_or_has_passed",
        ]
    elif reason == "no_eligible_documents_or_shadow_rows":
        exit_conditions = [
            "eligible_documents_exist_or_shadow_rows_exist",
            "health_probe_moves_out_of_insufficient_shadow_signal_yet",
            "shadow_remains_read_only_until_adoption_gate_exists",
        ]
    else:
        exit_conditions = [
            "health_probe_remains_green_on_rerun",
            "adoption_gate_and_release_hygiene_are_green",
            "sqlite_remains_truth_until_limited_cutover_is_explicitly_approved",
        ]

    return {
        "sqlite_fail_open": sqlite_fail_open,
        "shadow_write_policy": "freeze" if write_freeze else "allow",
        "shadow_read_policy": "read_only_shadow" if read_only_shadow_mode else "cutover_candidate",
        "entry_conditions": entry_conditions,
        "exit_conditions": exit_conditions,
    }




def _build_chroma_shadow_fallback_runbook_c17(*, health_probe: dict) -> dict:
    fallback_entry = health_probe.get("fallback_entry") or {}
    reason = str(fallback_entry.get("reason") or "unknown")
    operator_actions = _build_chroma_shadow_fallback_operator_actions_c17(
        reason=reason,
        health_probe=health_probe,
    )
    remediation_surface = {
        "landed_endpoints": [
            "/chroma-shadow/preflight",
            "/chroma-shadow/bootstrap",
            "/chroma-shadow/backfill-dry-run",
            "/chroma-shadow/backfill-batch",
            "/chroma-shadow/reembed-audit",
            "/chroma-shadow/health-probe",
        ],
        "pending_endpoints": [
            "/chroma-shadow/rebuild-drill",
            "/chroma-shadow/adoption-gate",
            "/chroma-shadow/release-hygiene",
            "/chroma-shadow/visibility-guard",
        ],
        "operator_actions": operator_actions,
    }
    return {
        "decision_matrix": {
            "decision": fallback_entry.get("decision"),
            "reason": fallback_entry.get("reason"),
            "mode_contract": _build_chroma_shadow_fallback_mode_contract_c17(fallback_entry=fallback_entry),
        },
        "operator_gated_admin_remediation_surface": remediation_surface,
        "health_probe_snapshot": {
            "eligible_document_counts": health_probe.get("eligible_document_counts") or {},
            "chroma_session_counts": health_probe.get("chroma_session_counts") or {},
            "divergence_by_tier": health_probe.get("divergence_by_tier") or {},
            "sample_query_sanity": health_probe.get("sample_query_sanity") or {},
        },
    }




def _build_chroma_shadow_upsert_payload_from_rows_c17(*, sid: str, rows_by_tier: dict, id_prefix: str) -> dict:
    upsert_ids: list[str] = []
    upsert_documents: list[str] = []
    upsert_embeddings: list[list[float]] = []
    upsert_metadatas: list[dict] = []
    failure_log: list[dict] = []
    selected_document_count = 0
    for tier in ("memory", "episode", "chapter", "arc", "saga"):
        items = rows_by_tier.get(tier) or []
        for item in items:
            source_row_id = item.get("id") if isinstance(item.get("id"), int) else None
            doc = _build_retrieval_document_q1a(tier, item, sid)
            if not doc:
                failure_log.append({
                    "tier": tier,
                    "source_row_id": source_row_id,
                    "reason": "document_build_skipped",
                })
                continue
            selected_document_count += 1
            embedding = _extract_chroma_shadow_embedding_c17(item)
            if not embedding:
                failure_log.append({
                    "tier": tier,
                    "source_row_id": source_row_id,
                    "reason": "embedding_missing",
                })
                continue
            doc_id = f"{id_prefix}:{doc.get('document_id')}:{source_row_id if source_row_id is not None else doc.get('document_id')}"
            upsert_ids.append(doc_id)
            upsert_documents.append(str(doc.get("text") or ""))
            upsert_embeddings.append(embedding)
            upsert_metadatas.append({
                "tier": str(doc.get("tier") or tier),
                "chat_session_id": sid,
                "source_table": str(doc.get("source_table") or ""),
                "source_row_id": str(doc.get("source_row_id") or source_row_id or ""),
                "schema_version": str(doc.get("schema_version") or _RETRIEVAL_DOCUMENT_SCHEMA_Q1A["version"]),
            })
    return {
        "ids": upsert_ids,
        "documents": upsert_documents,
        "embeddings": upsert_embeddings,
        "metadatas": upsert_metadatas,
        "failure_log": failure_log,
        "selected_document_count": selected_document_count,
    }




def _snapshot_chroma_shadow_collection_c17(collection) -> dict:
    warnings: list[str] = []
    count = None
    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []
    try:
        count = int(collection.count())
    except Exception as exc:
        warnings.append(f"snapshot_count_unavailable:{exc.__class__.__name__}")
    try:
        payload = collection.get(include=["documents", "embeddings", "metadatas"])
        ids = list((payload or {}).get("ids") or [])
        documents = list((payload or {}).get("documents") or [])
        embeddings = list((payload or {}).get("embeddings") or [])
        metadatas = list((payload or {}).get("metadatas") or [])
    except ValueError:
        if count not in (None, 0):
            warnings.append("snapshot_payload_unavailable:ValueError")
    except Exception as exc:
        warnings.append(f"snapshot_payload_unavailable:{exc.__class__.__name__}")
    rollback_recoverable = len(ids) == len(documents) == len(embeddings) == len(metadatas)
    if ids == [] and count in (None, 0):
        rollback_recoverable = True
    return {
        "count": count,
        "ids": ids,
        "documents": documents,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "warnings": warnings,
        "rollback_recoverable": rollback_recoverable,
    }




def _normalize_chroma_shadow_adoption_evidence_c17(raw: object) -> dict:
    evidence = raw if isinstance(raw, dict) else {}
    return {
        "regression_corpus_green": bool(evidence.get("regression_corpus_green")),
        "temporal_replay_green": bool(evidence.get("temporal_replay_green")),
        "source_tag_replay_green": bool(evidence.get("source_tag_replay_green")),
        "rebuild_rollback_drill_green": bool(evidence.get("rebuild_rollback_drill_green")),
        "fallback_runbook_reviewed": bool(evidence.get("fallback_runbook_reviewed")),
    }




def _normalize_chroma_shadow_release_evidence_c17(raw: object) -> dict:
    evidence = raw if isinstance(raw, dict) else {}
    return {
        "evaluation_split_smoke_green": bool(evidence.get("evaluation_split_smoke_green")),
        "ops_dry_run_checklist_green": bool(evidence.get("ops_dry_run_checklist_green")),
        "inspection_lane_review_green": bool(evidence.get("inspection_lane_review_green")),
        "bundle_regenerate_synced": bool(evidence.get("bundle_regenerate_synced")),
        "post_migration_smoke_green": bool(evidence.get("post_migration_smoke_green")),
        "release_note_synced": bool(evidence.get("release_note_synced")),
        "known_risk_ledger_synced": bool(evidence.get("known_risk_ledger_synced")),
    }




def _normalize_chroma_shadow_visibility_sample_limit_c17(value: object) -> int:
    try:
        normalized = int(value)
    except Exception:
        normalized = 20
    return max(1, min(normalized, 100))


def _safe_audit_details_c17(raw: object) -> dict:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}




def _query_chroma_shadow_visibility_audit_rows_c17(*, sid: str, sample_limit: int) -> dict:
    db = SessionLocal()
    rows_by_type = {
        "complete_turn_visibility": [],
        "critic_pipeline_trace": [],
    }
    warnings: list[str] = []
    try:
        try:
            rows = (
                db.query(AuditLog)
                .filter(
                    AuditLog.chat_session_id == sid,
                    AuditLog.event_type.in_(["complete_turn_visibility", "critic_pipeline_trace"]),
                )
                .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
                .limit(max(sample_limit * 4, 20))
                .all()
            )
        except Exception as exc:
            warnings.append(f"visibility_audit_query_unavailable:{exc.__class__.__name__}")
            rows = []

        for row in rows:
            event_type = str(getattr(row, "event_type", "") or "")
            if event_type not in rows_by_type:
                continue
            if len(rows_by_type[event_type]) >= sample_limit:
                continue
            rows_by_type[event_type].append(row)

        return {
            "rows_by_type": rows_by_type,
            "warnings": warnings,
        }
    finally:
        db.close()




def _build_chroma_shadow_visibility_incidents_c17(*, rows_by_type: dict) -> dict:
    incidents: list[dict] = []
    signal_coverage = {
        "complete_turn_visibility": False,
        "critic_pipeline_trace": False,
    }

    for event_type, rows in (rows_by_type or {}).items():
        signal_coverage[event_type] = bool(rows)
        for row in rows:
            details = _safe_audit_details_c17(getattr(row, "details_json", None))
            status = str(details.get("status") or "").strip().lower()
            warnings = details.get("warnings") if isinstance(details.get("warnings"), list) else []
            incident_count = int(details.get("incident_count") or details.get("visible_failure_count") or 0)
            if event_type == "critic_pipeline_trace":
                is_visible_failure = status in {"failed", "degraded"}
                detail = str(details.get("status") or getattr(row, "summary", None) or "critic_pipeline_trace")
            else:
                is_visible_failure = status in {"failed", "degraded", "blocked"} or incident_count > 0
                detail = str(details.get("status") or getattr(row, "summary", None) or "complete_turn_visibility")

            if not is_visible_failure:
                continue
            incidents.append({
                "event_type": event_type,
                "audit_id": getattr(row, "id", None),
                "created_at": getattr(row, "created_at", None).isoformat() if getattr(row, "created_at", None) is not None and hasattr(getattr(row, "created_at", None), "isoformat") else str(getattr(row, "created_at", None) or ""),
                "status": status or "incident",
                "summary": getattr(row, "summary", None),
                "detail": detail,
                "warning_count": len(warnings),
                "incident_count": incident_count,
            })

    incidents.sort(key=lambda item: (str(item.get("created_at") or ""), int(item.get("audit_id") or 0)), reverse=True)
    coverage_score = sum(1 for present in signal_coverage.values() if present)
    if incidents:
        guard_status = "blocked"
    elif coverage_score == len(signal_coverage):
        guard_status = "clear"
    else:
        guard_status = "limited_signal"

    return {
        "guard_status": guard_status,
        "signal_coverage": signal_coverage,
        "signal_coverage_score": coverage_score,
        "visible_failure_count": len(incidents),
        "recent_incidents": incidents[:8],
    }


# ------------------------------------------------------------------ #
#  Layer 5: Public Workflow Runners                                    #
# ------------------------------------------------------------------ #
def _run_chroma_shadow_backfill_dry_run_c17(req: ChromaShadowBackfillDryRunRequest) -> dict:
    sid = " ".join(str(req.chat_session_id or "").split()).strip()
    if not sid:
        raise ValueError("chat_session_id is required")

    sample_limit_per_tier = _normalize_chroma_shadow_sample_limit_c17(req.sample_limit_per_tier)
    bootstrap = _build_chroma_shadow_bootstrap_c17()
    if bootstrap.get("status") != "ok":
        return {
            "status": "blocked",
            "step": "17-C3",
            "chat_session_id": sid,
            "sample_limit_per_tier": sample_limit_per_tier,
            "bootstrap": bootstrap,
        }

    collection_handle = _open_chroma_shadow_collection_c17()
    collection = collection_handle["collection"]
    candidate_bundle = _query_chroma_shadow_backfill_candidates_c17(
        sid=sid,
        sample_limit_per_tier=sample_limit_per_tier,
    )
    rows_by_tier = candidate_bundle.get("rows_by_tier") or {}

    candidate_documents: list[dict] = []
    upsert_ids: list[str] = []
    upsert_documents: list[str] = []
    upsert_embeddings: list[list[float]] = []
    upsert_metadatas: list[dict] = []
    sample_preview: list[dict] = []
    warnings = list(candidate_bundle.get("warnings") or [])

    for tier in ("memory", "episode", "chapter", "arc", "saga"):
        items = rows_by_tier.get(tier) or []
        for item in items:
            doc = _build_retrieval_document_q1a(tier, item, sid)
            if not doc:
                warnings.append(f"{tier}_document_build_skipped:{item.get('id')}")
                continue
            candidate_documents.append(doc)
            embedding = _extract_chroma_shadow_embedding_c17(item)
            if not embedding:
                warnings.append(f"{tier}_embedding_missing:{item.get('id')}")
                continue
            source_row_id = doc.get("source_row_id")
            source_key = source_row_id if source_row_id is not None else doc.get("document_id")
            upsert_ids.append(f"dryrun:{doc.get('document_id')}:{source_key}")
            upsert_documents.append(str(doc.get("text") or ""))
            upsert_embeddings.append(embedding)
            upsert_metadatas.append({
                "tier": str(doc.get("tier") or tier),
                "chat_session_id": sid,
                "source_table": str(doc.get("source_table") or ""),
                "source_row_id": str(source_row_id or ""),
                "schema_version": str(doc.get("schema_version") or _RETRIEVAL_DOCUMENT_SCHEMA_Q1A["version"]),
            })
            sample_preview.append({
                "document_id": doc.get("document_id"),
                "tier": doc.get("tier"),
                "title": doc.get("title"),
                "source_row_id": source_row_id,
                "text_preview": _clean_retrieval_text_q1a(doc.get("text"), max_len=120),
            })

    count_before = None
    count_after = None
    try:
        count_before = int(collection.count())
    except Exception:
        count_before = None

    if upsert_ids:
        collection.upsert(
            ids=upsert_ids,
            documents=upsert_documents,
            embeddings=upsert_embeddings,
            metadatas=upsert_metadatas,
        )

    try:
        count_after = int(collection.count())
    except Exception:
        count_after = None

    return {
        "status": "ok",
        "step": "17-C3",
        "chat_session_id": sid,
        "sample_limit_per_tier": sample_limit_per_tier,
        "candidate_document_count": len(candidate_documents),
        "sample_upsert_count": len(upsert_ids),
        "count_before": count_before,
        "count_after": count_after,
        "collection_name": collection_handle.get("collection_name"),
        "bootstrap_contract_path": str(_resolve_chroma_shadow_contract_path_c17()),
        "sample_preview": sample_preview[:8],
        "warnings": warnings,
        "preflight_issues": list((bootstrap.get("preflight_issues") or bootstrap.get("health_probe_baseline", {}).get("preflight_issues") or [])),
        "retrieval_read_path_changed": False,
    }




def _run_chroma_shadow_backfill_batch_c17(req: ChromaShadowBackfillBatchRequest) -> dict:
    sid = " ".join(str(req.chat_session_id or "").split()).strip()
    if not sid:
        raise ValueError("chat_session_id is required")

    batch_size_per_tier = _normalize_chroma_shadow_batch_size_c17(req.batch_size_per_tier)
    checkpoint = _normalize_chroma_shadow_checkpoint_map_c17(req.checkpoint)
    retry_rows = _normalize_chroma_shadow_retry_rows_c17(req.retry_rows)

    bootstrap = _build_chroma_shadow_bootstrap_c17()
    if bootstrap.get("status") != "ok":
        return {
            "status": "blocked",
            "step": "17-C4",
            "chat_session_id": sid,
            "batch_size_per_tier": batch_size_per_tier,
            "bootstrap": bootstrap,
        }

    collection_handle = _open_chroma_shadow_collection_c17()
    collection = collection_handle["collection"]
    candidate_bundle = _query_chroma_shadow_backfill_batch_candidates_c17(
        sid=sid,
        batch_size_per_tier=batch_size_per_tier,
        checkpoint=checkpoint,
        retry_rows=retry_rows,
    )
    rows_by_tier = candidate_bundle.get("rows_by_tier") or {}
    warnings = list(candidate_bundle.get("warnings") or [])
    failure_log: list[dict] = []
    ingested_count = 0
    selected_document_count = 0

    count_before = None
    count_after = None
    try:
        count_before = int(collection.count())
    except Exception:
        count_before = None

    for tier in ("memory", "episode", "chapter", "arc", "saga"):
        items = rows_by_tier.get(tier) or []
        for item in items:
            source_row_id = item.get("id") if isinstance(item.get("id"), int) else None
            doc = _build_retrieval_document_q1a(tier, item, sid)
            if not doc:
                failure_log.append({
                    "tier": tier,
                    "source_row_id": source_row_id,
                    "reason": "document_build_skipped",
                })
                continue
            selected_document_count += 1
            embedding = _extract_chroma_shadow_embedding_c17(item)
            if not embedding:
                failure_log.append({
                    "tier": tier,
                    "source_row_id": source_row_id,
                    "reason": "embedding_missing",
                })
                continue
            doc_id = f"batch:{doc.get('document_id')}:{source_row_id if source_row_id is not None else doc.get('document_id')}"
            metadata = {
                "tier": str(doc.get("tier") or tier),
                "chat_session_id": sid,
                "source_table": str(doc.get("source_table") or ""),
                "source_row_id": str(doc.get("source_row_id") or source_row_id or ""),
                "schema_version": str(doc.get("schema_version") or _RETRIEVAL_DOCUMENT_SCHEMA_Q1A["version"]),
            }
            try:
                collection.upsert(
                    ids=[doc_id],
                    documents=[str(doc.get("text") or "")],
                    embeddings=[embedding],
                    metadatas=[metadata],
                )
                ingested_count += 1
            except Exception as exc:
                failure_log.append({
                    "tier": tier,
                    "source_row_id": source_row_id,
                    "reason": f"upsert_failed:{exc.__class__.__name__}",
                })

    try:
        count_after = int(collection.count())
    except Exception:
        count_after = None

    return {
        "status": "ok",
        "step": "17-C4",
        "chat_session_id": sid,
        "batch_size_per_tier": batch_size_per_tier,
        "checkpoint_start": candidate_bundle.get("checkpoint_start") or {},
        "checkpoint_next": candidate_bundle.get("checkpoint_next") or {},
        "remaining_counts_by_tier": candidate_bundle.get("remaining_counts_by_tier") or {},
        "selected_counts_by_tier": candidate_bundle.get("selected_counts_by_tier") or {},
        "selected_document_count": selected_document_count,
        "ingested_count": ingested_count,
        "has_more": bool(candidate_bundle.get("has_more")),
        "failure_log": failure_log,
        "warnings": warnings,
        "count_before": count_before,
        "count_after": count_after,
        "collection_name": collection_handle.get("collection_name"),
        "retry_rows_applied": retry_rows,
        "preflight_issues": list((bootstrap.get("preflight_issues") or bootstrap.get("health_probe_baseline", {}).get("preflight_issues") or [])),
        "retrieval_read_path_changed": False,
    }




def _run_chroma_shadow_reembed_audit_c17(req: ChromaShadowReembedAuditRequest) -> dict:
    sid = " ".join(str(req.chat_session_id or "").split()).strip()
    if not sid:
        raise ValueError("chat_session_id is required")

    queue_limit_per_tier = _normalize_chroma_shadow_queue_limit_c17(req.queue_limit_per_tier)
    preflight = _build_chroma_shadow_preflight_c17()
    current_model = str((_build_chroma_shadow_embedder_identity_c17().get("model") or "")).strip()
    invalidation_rules = _build_chroma_shadow_reembed_invalidation_rules_c17()

    db = SessionLocal()
    status_counts_by_tier = {tier: {} for tier in ("memory", "episode", "chapter", "arc", "saga")}
    candidate_counts_by_tier = {tier: 0 for tier in ("memory", "episode", "chapter", "arc", "saga")}
    queue_by_tier = {tier: [] for tier in ("memory", "episode", "chapter", "arc", "saga")}
    warnings: list[str] = []
    try:
        for tier in queue_by_tier.keys():
            model = _resolve_chroma_shadow_tier_model_c17(tier)
            if model is None:
                continue
            try:
                rows = (
                    db.query(model)
                    .filter(model.chat_session_id == sid)
                    .order_by(model.id.asc())
                    .all()
                )
            except Exception as exc:
                warnings.append(f"{tier}_audit_query_unavailable:{exc.__class__.__name__}")
                continue

            for row in rows:
                item = _row_to_chroma_shadow_item_c17(tier, row)
                raw_embedding = item.get("embedding")
                if raw_embedding in (None, ""):
                    raw_embedding = item.get("embedding_vector")
                parsed_embedding = _safe_json_loads_q1a(raw_embedding)
                embedding_value = parsed_embedding if isinstance(parsed_embedding, list) else raw_embedding
                audit_state = _assess_memory_embedding_state_c17(
                    embedding_value=embedding_value,
                    embedding_model=item.get("embedding_model"),
                    current_model=current_model,
                )
                status = str(audit_state.get("status") or "unknown")
                tier_status_counts = status_counts_by_tier[tier]
                tier_status_counts[status] = int(tier_status_counts.get(status) or 0) + 1
                if audit_state.get("needs_reembed") is not True:
                    continue
                candidate_counts_by_tier[tier] = int(candidate_counts_by_tier.get(tier) or 0) + 1
                if len(queue_by_tier[tier]) >= queue_limit_per_tier:
                    continue
                doc = _build_retrieval_document_q1a(tier, item, sid)
                rule = invalidation_rules.get(status) or {}
                queue_by_tier[tier].append({
                    "tier": tier,
                    "source_row_id": item.get("id"),
                    "document_id": (doc or {}).get("document_id"),
                    "title": (doc or {}).get("title"),
                    "status": status,
                    "stored_embedding_model": audit_state.get("stored_model"),
                    "current_project_embedding_model": audit_state.get("current_model"),
                    "has_embedding_vector": bool(audit_state.get("has_embedding_vector")),
                    "recommended_action": rule.get("recommended_action"),
                    "invalidation_rule": rule.get("invalidation_rule"),
                })
    finally:
        db.close()

    total_candidates = sum(int(value or 0) for value in candidate_counts_by_tier.values())
    status_totals: dict[str, int] = {}
    for tier_counts in status_counts_by_tier.values():
        for status, count in tier_counts.items():
            status_totals[status] = int(status_totals.get(status) or 0) + int(count or 0)

    blockers: list[str] = []
    if not current_model:
        blockers.append("project_model_unset")

    return {
        "status": "ok",
        "step": "17-C5",
        "chat_session_id": sid,
        "queue_limit_per_tier": queue_limit_per_tier,
        "current_project_embedding_model": current_model or None,
        "current_model_ready": bool(current_model),
        "blockers": blockers,
        "candidate_counts_by_tier": candidate_counts_by_tier,
        "needs_reembed_total": total_candidates,
        "status_counts_by_tier": status_counts_by_tier,
        "status_totals": status_totals,
        "queue_by_tier": queue_by_tier,
        "invalidation_rules": invalidation_rules,
        "warnings": warnings,
        "preflight_issues": list(preflight.get("issues") or []),
        "retrieval_read_path_changed": False,
    }




def _run_chroma_shadow_health_probe_c17(req: ChromaShadowHealthProbeRequest) -> dict:
    sid = " ".join(str(req.chat_session_id or "").split()).strip()
    if not sid:
        raise ValueError("chat_session_id is required")

    preflight = _build_chroma_shadow_preflight_c17()
    current_model = str((_build_chroma_shadow_embedder_identity_c17().get("model") or "")).strip()
    collection_handle = _open_chroma_shadow_collection_c17()
    collection = collection_handle.get("collection")

    eligible_bundle = _collect_chroma_shadow_eligible_counts_c17(sid=sid, current_model=current_model)
    shadow_bundle = _count_chroma_shadow_session_documents_c17(collection, sid=sid)

    eligible_counts = eligible_bundle.get("eligible_counts") or {}
    shadow_counts = shadow_bundle.get("session_counts") or {}
    divergence_by_tier = {}
    for tier in ("memory", "episode", "chapter", "arc", "saga"):
        eligible = int(eligible_counts.get(tier) or 0)
        shadow = int(shadow_counts.get(tier) or 0)
        if eligible != shadow:
            divergence_by_tier[tier] = {
                "eligible_count": eligible,
                "shadow_count": shadow,
                "delta": eligible - shadow,
            }

    sample_query_sanity = {
        "status": "skip",
        "detail": "no_sample_query_executed",
        "returned_id": None,
        "sample_source": eligible_bundle.get("sample_query_source"),
    }
    sample_embedding = eligible_bundle.get("sample_query_embedding")
    if sample_embedding and sum(int(value or 0) for value in shadow_counts.values()) > 0:
        try:
            result = collection.query(query_embeddings=[sample_embedding], n_results=1)
            ids = (result or {}).get("ids") or []
            first_row = ids[0] if isinstance(ids, list) and ids else []
            first_id = first_row[0] if isinstance(first_row, list) and first_row else None
            sample_query_sanity = {
                "status": "pass" if first_id else "fail",
                "detail": "query_returned_candidate" if first_id else "query_returned_no_candidate",
                "returned_id": first_id,
                "sample_source": eligible_bundle.get("sample_query_source"),
            }
        except Exception as exc:
            sample_query_sanity = {
                "status": "fail",
                "detail": f"query_failed:{exc.__class__.__name__}",
                "returned_id": None,
                "sample_source": eligible_bundle.get("sample_query_source"),
            }

    fallback_entry = _build_chroma_shadow_fallback_entry_c17(
        preflight_issues=list(preflight.get("issues") or []),
        current_model_ready=bool(current_model),
        divergence_by_tier=divergence_by_tier,
        sample_query_sanity=sample_query_sanity,
    )

    return {
        "status": "ok",
        "step": "17-C6",
        "chat_session_id": sid,
        "current_project_embedding_model": current_model or None,
        "current_model_ready": bool(current_model),
        "sqlite_total_counts": eligible_bundle.get("sqlite_total_counts") or {},
        "eligible_document_counts": eligible_counts,
        "chroma_session_counts": shadow_counts,
        "collection_document_count": shadow_bundle.get("collection_document_count"),
        "divergence_by_tier": divergence_by_tier,
        "sample_query_sanity": sample_query_sanity,
        "fallback_entry": fallback_entry,
        "warnings": list(eligible_bundle.get("warnings") or []) + list(shadow_bundle.get("warnings") or []),
        "preflight_issues": list(preflight.get("issues") or []),
        "retrieval_read_path_changed": False,
    }




def _run_chroma_shadow_fallback_runbook_c17(req: ChromaShadowFallbackRunbookRequest) -> dict:
    sid = " ".join(str(req.chat_session_id or "").split()).strip()
    if not sid:
        raise ValueError("chat_session_id is required")

    health_probe = _run_chroma_shadow_health_probe_c17(ChromaShadowHealthProbeRequest(chat_session_id=sid))
    runbook = _build_chroma_shadow_fallback_runbook_c17(health_probe=health_probe)
    return {
        "status": "ok",
        "step": "17-C7",
        "chat_session_id": sid,
        "decision_matrix": runbook.get("decision_matrix") or {},
        "operator_gated_admin_remediation_surface": runbook.get("operator_gated_admin_remediation_surface") or {},
        "health_probe_snapshot": runbook.get("health_probe_snapshot") or {},
        "fallback_entry": health_probe.get("fallback_entry") or {},
        "warnings": list(health_probe.get("warnings") or []),
        "preflight_issues": list(health_probe.get("preflight_issues") or []),
        "retrieval_read_path_changed": False,
    }




def _run_chroma_shadow_rebuild_drill_c17(req: ChromaShadowRebuildDrillRequest) -> dict:
    sid = " ".join(str(req.chat_session_id or "").split()).strip()
    if not sid:
        raise ValueError("chat_session_id is required")

    batch_size_per_tier = _normalize_chroma_shadow_batch_size_c17(req.batch_size_per_tier)
    checkpoint = _normalize_chroma_shadow_checkpoint_map_c17(req.checkpoint)
    retry_rows = _normalize_chroma_shadow_retry_rows_c17(req.retry_rows)
    execute_rebuild_then_rollback = bool(req.execute_rebuild_then_rollback)

    bootstrap = _build_chroma_shadow_bootstrap_c17()
    if bootstrap.get("status") != "ok":
        return {
            "status": "blocked",
            "step": "17-C8",
            "chat_session_id": sid,
            "batch_size_per_tier": batch_size_per_tier,
            "execute_rebuild_then_rollback": execute_rebuild_then_rollback,
            "bootstrap": bootstrap,
        }

    collection_handle = _open_chroma_shadow_collection_c17()
    client = collection_handle.get("client")
    collection = collection_handle.get("collection")
    collection_name = str(collection_handle.get("collection_name") or "")
    candidate_bundle = _query_chroma_shadow_backfill_batch_candidates_c17(
        sid=sid,
        batch_size_per_tier=batch_size_per_tier,
        checkpoint=checkpoint,
        retry_rows=retry_rows,
    )
    rows_by_tier = candidate_bundle.get("rows_by_tier") or {}
    planned_payload = _build_chroma_shadow_upsert_payload_from_rows_c17(
        sid=sid,
        rows_by_tier=rows_by_tier,
        id_prefix="rebuild",
    )
    snapshot = _snapshot_chroma_shadow_collection_c17(collection)
    warnings = list(candidate_bundle.get("warnings") or []) + list(snapshot.get("warnings") or [])
    execution_log: list[str] = []
    rebuild_count = None
    rollback_count = None
    rollback_status = "not_requested"

    if execute_rebuild_then_rollback:
        if not snapshot.get("rollback_recoverable"):
            rollback_status = "snapshot_unrecoverable"
            warnings.append("rollback_snapshot_not_recoverable")
        else:
            try:
                client.delete_collection(name=collection_name)
                execution_log.append("deleted_shadow_collection")
            except Exception as exc:
                raise RuntimeError(f"delete_collection_failed:{exc.__class__.__name__}") from exc

            rebuilt_collection = _create_chroma_shadow_collection_with_client_c17(client, collection_name=collection_name)
            execution_log.append("recreated_shadow_collection")
            if planned_payload.get("ids"):
                rebuilt_collection.upsert(
                    ids=planned_payload.get("ids") or [],
                    documents=planned_payload.get("documents") or [],
                    embeddings=planned_payload.get("embeddings") or [],
                    metadatas=planned_payload.get("metadatas") or [],
                )
                execution_log.append("upserted_rebuild_payload")
            try:
                rebuild_count = int(rebuilt_collection.count())
            except Exception as exc:
                warnings.append(f"rebuild_count_unavailable:{exc.__class__.__name__}")

            try:
                client.delete_collection(name=collection_name)
                execution_log.append("deleted_rebuild_collection_for_rollback")
            except Exception as exc:
                raise RuntimeError(f"rollback_delete_failed:{exc.__class__.__name__}") from exc

            rollback_collection = _create_chroma_shadow_collection_with_client_c17(client, collection_name=collection_name)
            if snapshot.get("ids"):
                rollback_collection.upsert(
                    ids=snapshot.get("ids") or [],
                    documents=snapshot.get("documents") or [],
                    embeddings=snapshot.get("embeddings") or [],
                    metadatas=snapshot.get("metadatas") or [],
                )
                execution_log.append("restored_snapshot_payload")
            try:
                rollback_count = int(rollback_collection.count())
            except Exception as exc:
                warnings.append(f"rollback_count_unavailable:{exc.__class__.__name__}")
            rollback_status = "completed"

    return {
        "status": "ok",
        "step": "17-C8",
        "chat_session_id": sid,
        "batch_size_per_tier": batch_size_per_tier,
        "execute_rebuild_then_rollback": execute_rebuild_then_rollback,
        "checkpoint_start": candidate_bundle.get("checkpoint_start") or {},
        "checkpoint_next": candidate_bundle.get("checkpoint_next") or {},
        "remaining_counts_by_tier": candidate_bundle.get("remaining_counts_by_tier") or {},
        "selected_counts_by_tier": candidate_bundle.get("selected_counts_by_tier") or {},
        "selected_document_count": planned_payload.get("selected_document_count") or 0,
        "planned_upsert_count": len(planned_payload.get("ids") or []),
        "failure_log": planned_payload.get("failure_log") or [],
        "snapshot_count_before_rebuild": snapshot.get("count"),
        "rollback_recoverable": bool(snapshot.get("rollback_recoverable")),
        "rebuild_count_after_execute": rebuild_count,
        "rollback_count_after_restore": rollback_count,
        "rollback_status": rollback_status,
        "execution_log": execution_log,
        "collection_name": collection_name,
        "warnings": warnings,
        "preflight_issues": list((bootstrap.get("preflight_issues") or bootstrap.get("health_probe_baseline", {}).get("preflight_issues") or [])),
        "retrieval_read_path_changed": False,
    }




def _run_chroma_shadow_adoption_gate_c17(req: ChromaShadowAdoptionGateRequest) -> dict:
    sid = " ".join(str(req.chat_session_id or "").split()).strip()
    if not sid:
        raise ValueError("chat_session_id is required")

    operator_evidence = _normalize_chroma_shadow_adoption_evidence_c17(req.operator_evidence)
    health_probe = _run_chroma_shadow_health_probe_c17(ChromaShadowHealthProbeRequest(chat_session_id=sid))
    fallback_runbook = _run_chroma_shadow_fallback_runbook_c17(ChromaShadowFallbackRunbookRequest(chat_session_id=sid))
    rebuild_drill = _run_chroma_shadow_rebuild_drill_c17(
        ChromaShadowRebuildDrillRequest(
            chat_session_id=sid,
            execute_rebuild_then_rollback=False,
        )
    )

    preflight_issues = list(health_probe.get("preflight_issues") or [])
    fallback_entry = fallback_runbook.get("fallback_entry") or {}
    fallback_decision = str((fallback_runbook.get("decision_matrix") or {}).get("decision") or fallback_entry.get("decision") or "")
    blocking_reasons: list[str] = []
    hold_reasons: list[str] = []

    if any(issue in {"chromadb_dependency_unavailable", "persist_directory_not_writable"} for issue in preflight_issues):
        blocking_reasons.append("shadow_infrastructure_blocked")
    if not bool(health_probe.get("current_model_ready")):
        blocking_reasons.append("project_model_unset")

    if fallback_decision != "shadow_health_green":
        hold_reasons.append(f"shadow_compare_not_green:{fallback_decision or 'unknown'}")
    if str(rebuild_drill.get("rollback_status") or "") not in {"not_requested", "completed"}:
        hold_reasons.append("rebuild_drill_not_recoverable")
    if not bool(rebuild_drill.get("rollback_recoverable")):
        hold_reasons.append("rollback_snapshot_not_recoverable")

    missing_evidence = [
        key for key, passed in operator_evidence.items()
        if passed is not True
    ]
    if missing_evidence:
        hold_reasons.append("missing_operator_evidence")

    if blocking_reasons:
        gate_status = "blocked"
    elif hold_reasons:
        gate_status = "hold"
    else:
        gate_status = "approved"

    operator_checklist = [
        {
            "item": "health_probe_shadow_compare",
            "passed": fallback_decision == "shadow_health_green",
            "detail": fallback_decision or "unknown",
        },
        {
            "item": "regression_corpus",
            "passed": operator_evidence.get("regression_corpus_green") is True,
            "detail": "operator_supplied",
        },
        {
            "item": "temporal_replay",
            "passed": operator_evidence.get("temporal_replay_green") is True,
            "detail": "operator_supplied",
        },
        {
            "item": "source_tag_replay",
            "passed": operator_evidence.get("source_tag_replay_green") is True,
            "detail": "operator_supplied",
        },
        {
            "item": "rebuild_rollback_drill",
            "passed": operator_evidence.get("rebuild_rollback_drill_green") is True and bool(rebuild_drill.get("rollback_recoverable")),
            "detail": str(rebuild_drill.get("rollback_status") or "unknown"),
        },
        {
            "item": "fallback_runbook_review",
            "passed": operator_evidence.get("fallback_runbook_reviewed") is True,
            "detail": str((fallback_runbook.get("decision_matrix") or {}).get("reason") or "unknown"),
        },
    ]

    return {
        "status": "ok",
        "step": "17-C9",
        "chat_session_id": sid,
        "gate_status": gate_status,
        "limited_cutover_approved": gate_status == "approved",
        "truth_source": "health_probe_plus_fallback_runbook",
        "blocking_reasons": blocking_reasons,
        "hold_reasons": hold_reasons,
        "missing_operator_evidence": missing_evidence,
        "operator_checklist": operator_checklist,
        "operator_evidence": operator_evidence,
        "health_probe_snapshot": {
            "fallback_entry": health_probe.get("fallback_entry") or {},
            "divergence_by_tier": health_probe.get("divergence_by_tier") or {},
            "sample_query_sanity": health_probe.get("sample_query_sanity") or {},
        },
        "fallback_runbook_snapshot": {
            "decision_matrix": fallback_runbook.get("decision_matrix") or {},
            "operator_gated_admin_remediation_surface": fallback_runbook.get("operator_gated_admin_remediation_surface") or {},
        },
        "rebuild_drill_snapshot": {
            "rollback_recoverable": bool(rebuild_drill.get("rollback_recoverable")),
            "rollback_status": rebuild_drill.get("rollback_status"),
            "planned_upsert_count": rebuild_drill.get("planned_upsert_count"),
        },
        "warnings": list(health_probe.get("warnings") or []) + list(rebuild_drill.get("warnings") or []),
        "preflight_issues": preflight_issues,
        "retrieval_read_path_changed": False,
    }




def _run_chroma_shadow_release_hygiene_c17(req: ChromaShadowReleaseHygieneRequest) -> dict:
    sid = " ".join(str(req.chat_session_id or "").split()).strip()
    if not sid:
        raise ValueError("chat_session_id is required")

    adoption_gate = _run_chroma_shadow_adoption_gate_c17(
        ChromaShadowAdoptionGateRequest(
            chat_session_id=sid,
            operator_evidence=req.adoption_operator_evidence,
        )
    )
    release_evidence = _normalize_chroma_shadow_release_evidence_c17(req.release_evidence)
    missing_release_evidence = [
        key for key, passed in release_evidence.items()
        if passed is not True
    ]

    gate_status = str(adoption_gate.get("gate_status") or "unknown")
    blocking_reasons: list[str] = []
    pending_reasons: list[str] = []
    if gate_status == "blocked":
        blocking_reasons.extend(list(adoption_gate.get("blocking_reasons") or []) or ["adoption_gate_blocked"])
    elif gate_status != "approved":
        pending_reasons.append(f"adoption_gate_not_ready:{gate_status}")

    if missing_release_evidence:
        pending_reasons.append("missing_release_evidence")

    if blocking_reasons:
        hygiene_status = "blocked"
    elif pending_reasons:
        hygiene_status = "pending"
    else:
        hygiene_status = "ready"

    operator_checklist = [
        {
            "item": "adoption_gate",
            "passed": gate_status == "approved",
            "detail": gate_status,
        },
        {
            "item": "evaluation_split_smoke",
            "passed": release_evidence.get("evaluation_split_smoke_green") is True,
            "detail": "operator_supplied",
        },
        {
            "item": "ops_dry_run_checklist",
            "passed": release_evidence.get("ops_dry_run_checklist_green") is True,
            "detail": "operator_supplied",
        },
        {
            "item": "inspection_lane_review",
            "passed": release_evidence.get("inspection_lane_review_green") is True,
            "detail": "operator_supplied",
        },
        {
            "item": "bundle_regenerate_sync",
            "passed": release_evidence.get("bundle_regenerate_synced") is True,
            "detail": "operator_supplied",
        },
        {
            "item": "post_migration_smoke",
            "passed": release_evidence.get("post_migration_smoke_green") is True,
            "detail": "operator_supplied",
        },
        {
            "item": "release_note_sync",
            "passed": release_evidence.get("release_note_synced") is True,
            "detail": "operator_supplied",
        },
        {
            "item": "known_risk_ledger_sync",
            "passed": release_evidence.get("known_risk_ledger_synced") is True,
            "detail": "operator_supplied",
        },
    ]

    return {
        "status": "ok",
        "step": "17-C10",
        "chat_session_id": sid,
        "release_hygiene_status": hygiene_status,
        "release_ready": hygiene_status == "ready",
        "blocking_reasons": blocking_reasons,
        "pending_reasons": pending_reasons,
        "missing_release_evidence": missing_release_evidence,
        "operator_checklist": operator_checklist,
        "adoption_gate_snapshot": {
            "gate_status": adoption_gate.get("gate_status"),
            "limited_cutover_approved": adoption_gate.get("limited_cutover_approved"),
            "blocking_reasons": adoption_gate.get("blocking_reasons") or [],
            "hold_reasons": adoption_gate.get("hold_reasons") or [],
            "missing_operator_evidence": adoption_gate.get("missing_operator_evidence") or [],
        },
        "release_evidence": release_evidence,
        "warnings": list(adoption_gate.get("warnings") or []),
        "preflight_issues": list(adoption_gate.get("preflight_issues") or []),
        "retrieval_read_path_changed": False,
    }




def _run_chroma_shadow_visibility_guard_c17(req: ChromaShadowVisibilityGuardRequest) -> dict:
    sid = " ".join(str(req.chat_session_id or "").split()).strip()
    if not sid:
        raise ValueError("chat_session_id is required")

    sample_limit = _normalize_chroma_shadow_visibility_sample_limit_c17(req.sample_limit)
    visibility_bundle = _query_chroma_shadow_visibility_audit_rows_c17(
        sid=sid,
        sample_limit=sample_limit,
    )
    incident_bundle = _build_chroma_shadow_visibility_incidents_c17(
        rows_by_type=visibility_bundle.get("rows_by_type") or {},
    )
    guard_status = str(incident_bundle.get("guard_status") or "limited_signal")
    runbook_actions = [
        {
            "action": "inspect_visibility_dashboard",
            "endpoint": "/chroma-shadow/visibility-guard",
            "availability": "landed",
            "required": True,
            "detail": guard_status,
        },
        {
            "action": "review_fallback_and_release_runbooks",
            "endpoints": ["/chroma-shadow/fallback-runbook", "/chroma-shadow/release-hygiene"],
            "availability": "landed",
            "required": guard_status == "blocked",
            "detail": "visible_failure_incidents_must_be_read_in_runbook_context",
        },
    ]

    return {
        "status": "ok",
        "step": "17-C11",
        "chat_session_id": sid,
        "guard_status": guard_status,
        "blocked": guard_status == "blocked",
        "signal_coverage": incident_bundle.get("signal_coverage") or {},
        "signal_coverage_score": incident_bundle.get("signal_coverage_score"),
        "visible_failure_count": incident_bundle.get("visible_failure_count") or 0,
        "recent_incidents": incident_bundle.get("recent_incidents") or [],
        "runbook_actions": runbook_actions,
        "audit_event_types": ["complete_turn_visibility", "critic_pipeline_trace"],
        "warnings": list(visibility_bundle.get("warnings") or []),
        "retrieval_read_path_changed": False,
    }


# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Layer 6: Public Workflow Runners (Preflight / Bootstrap)          #
# ------------------------------------------------------------------ #
def _build_chroma_shadow_preflight_c17() -> dict:
    runtime_config = retrieval_index_registry.get_runtime_config()
    persist_dir = _resolve_chroma_shadow_persist_dir_c17()
    dependency = _resolve_chroma_shadow_dependency_c17()
    disk_usage_target = persist_dir if persist_dir.exists() else persist_dir.parent
    disk_free_mb = None
    disk_total_mb = None
    try:
        usage = shutil.disk_usage(disk_usage_target)
        disk_free_mb = round(usage.free / (1024 * 1024), 2)
        disk_total_mb = round(usage.total / (1024 * 1024), 2)
    except OSError:
        pass

    embedder_identity = _build_chroma_shadow_embedder_identity_c17()

    writable_probe = persist_dir if persist_dir.exists() else persist_dir.parent
    issues: list[str] = []
    if not dependency.get("available"):
        issues.append("chromadb_dependency_unavailable")
    if not embedder_identity.get("model"):
        issues.append("embedding_model_missing")
    if not os.access(writable_probe, os.W_OK):
        issues.append("persist_directory_not_writable")

    return {
        "status": "ok",
        "step": "17-C1",
        "ready": len(issues) == 0,
        "issues": issues,
        "enabled": bool(_settings.CHROMA_SHADOW_ENABLED),
        "collection_name": _resolve_chroma_shadow_collection_name_c17(),
        "embedder_identity": embedder_identity,
        "retrieval_document_schema": {
            "version": _RETRIEVAL_DOCUMENT_SCHEMA_Q1A["version"],
            "tiers": list(_RETRIEVAL_DOCUMENT_SCHEMA_Q1A["tiers"]),
            "index_version": runtime_config.get("index_version"),
        },
        "session_partitioning": {
            "mode": "session_partitioned",
            "session_partitioned": True,
            "shadow_runtime_mode": runtime_config.get("mode"),
            "shadow_write_enabled": bool(runtime_config.get("shadow_write_enabled")),
            "active_session_count": int(runtime_config.get("session_count") or 0),
        },
        "persist_directory": {
            "path": str(persist_dir),
            "exists": persist_dir.exists(),
            "writable": bool(os.access(writable_probe, os.W_OK)),
        },
        "disk_budget": {
            "budget_mb": int(_settings.CHROMA_SHADOW_DISK_BUDGET_MB),
            "free_mb": disk_free_mb,
            "total_mb": disk_total_mb,
            "target_size_mb": round(_estimate_directory_size_bytes_c17(persist_dir) / (1024 * 1024), 2),
        },
        "dependency": dependency,
    }




def _build_chroma_shadow_bootstrap_c17() -> dict:
    preflight = _build_chroma_shadow_preflight_c17()
    persist_dir = _resolve_chroma_shadow_persist_dir_c17()
    contract_path = _resolve_chroma_shadow_contract_path_c17()
    collection_name = _resolve_chroma_shadow_collection_name_c17()
    sqlite_baseline = _collect_chroma_shadow_sqlite_counts_c17()
    sqlite_counts = dict(sqlite_baseline.get("counts") or {})
    runtime_config = retrieval_index_registry.get_runtime_config()
    baseline_captured_at = datetime.now(timezone.utc).isoformat()

    blocking_issues = [
        issue for issue in list(preflight.get("issues") or [])
        if issue in {"chromadb_dependency_unavailable", "persist_directory_not_writable"}
    ]
    if blocking_issues:
        return {
            "status": "blocked",
            "step": "17-C2",
            "ready": False,
            "issues": blocking_issues,
            "preflight": preflight,
        }

    persist_dir_created_now = not persist_dir.exists()
    persist_dir.mkdir(parents=True, exist_ok=True)

    contract_payload = {
        "contract_version": "c17.bootstrap.v1",
        "step": "17-C2",
        "captured_at": baseline_captured_at,
        "collection_name": collection_name,
        "truth_authority": "sqlite",
        "retrieval_read_path_changed": False,
        "shadow_write_mode": runtime_config.get("mode"),
        "embedder_identity": preflight.get("embedder_identity") or {},
        "retrieval_document_schema": preflight.get("retrieval_document_schema") or {},
        "session_partitioning": preflight.get("session_partitioning") or {},
        "health_probe_baseline": {
            "sqlite_counts": sqlite_counts,
            "sqlite_count_warnings": list(sqlite_baseline.get("warnings") or []),
            "preflight_ready": bool(preflight.get("ready")),
            "preflight_issues": list(preflight.get("issues") or []),
            "active_shadow_sessions": int(runtime_config.get("session_count") or 0),
        },
    }

    collection_handle = _open_chroma_shadow_collection_c17()
    collection = collection_handle["collection"]
    collection_preexisting = bool(collection_handle.get("collection_preexisting"))
    collection_count = None
    try:
        collection_count = int(collection.count())
    except Exception:
        collection_count = None

    contract_preexisting = contract_path.exists()
    contract_path.write_text(
        json.dumps(contract_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "step": "17-C2",
        "ready": True,
        "backfill_ready": bool(preflight.get("ready")),
        "issues": [],
        "preflight_issues": list(preflight.get("issues") or []),
        "persist_directory": {
            "path": str(persist_dir),
            "created_now": persist_dir_created_now,
            "exists": True,
        },
        "collection": {
            "name": collection_name,
            "created_now": not collection_preexisting,
            "document_count": collection_count,
        },
        "metadata_contract": {
            "path": str(contract_path),
            "written": True,
            "created_now": not contract_preexisting,
            "contract_version": contract_payload["contract_version"],
        },
        "health_probe_baseline": contract_payload["health_probe_baseline"],
        "retrieval_read_path_changed": False,
    }




