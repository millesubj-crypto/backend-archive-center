from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone, timedelta
import threading


_KST = timezone(timedelta(hours=9))


def _now_kst_string() -> str:
    return datetime.now(timezone.utc).astimezone(_KST).strftime("%Y-%m-%d %H:%M:%S")


class SessionPartitionedRetrievalIndex:
    INDEX_VERSION = "q1e.v1"
    ALLOWED_TIERS = ("memory", "episode", "chapter", "arc", "saga")
    ALLOWED_RUNTIME_MODES = ("shadow", "off")

    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: dict[str, dict] = {}
        self._runtime_mode = "shadow"
        self._runtime_updated_at = _now_kst_string()
        self._runtime_reason = "default"

    def get_runtime_config(self) -> dict:
        with self._lock:
            return {
                "mode": self._runtime_mode,
                "shadow_write_enabled": self._runtime_mode == "shadow",
                "updated_at": self._runtime_updated_at,
                "reason": self._runtime_reason,
                "session_count": len(self._sessions),
                "index_version": self.INDEX_VERSION,
            }

    def set_runtime_mode(self, mode: str, *, reason: str = "api") -> dict:
        normalized_mode = self._normalize_runtime_mode(mode)
        now_text = _now_kst_string()
        with self._lock:
            previous_mode = self._runtime_mode
            cleared_sessions = len(self._sessions) if normalized_mode == "off" else 0
            self._runtime_mode = normalized_mode
            self._runtime_updated_at = now_text
            self._runtime_reason = reason
            if normalized_mode == "off":
                self._sessions.clear()
            current_session_count = len(self._sessions)
        return {
            "mode": normalized_mode,
            "previous_mode": previous_mode,
            "shadow_write_enabled": normalized_mode == "shadow",
            "updated_at": now_text,
            "reason": reason,
            "cleared_sessions": cleared_sessions,
            "session_count": current_session_count,
            "index_version": self.INDEX_VERSION,
        }

    def replace_session_documents(
        self,
        chat_session_id: str,
        documents: list[dict],
        *,
        reason: str = "prepare_turn_refresh",
    ) -> dict:
        if not self.is_shadow_mode_enabled():
            return self.snapshot(chat_session_id)

        sid = " ".join(str(chat_session_id or "").split()).strip()
        if not sid:
            return self._empty_snapshot("")

        normalized_docs = [self._normalize_document(doc, sid) for doc in documents if isinstance(doc, dict)]
        partitions = {tier: [] for tier in self.ALLOWED_TIERS}
        source_types: dict[str, int] = {}

        for doc in normalized_docs:
            tier = doc.get("tier")
            if tier not in partitions:
                continue
            partitions[tier].append(doc)
            source_type = str(doc.get("source_type") or tier)
            source_types[source_type] = source_types.get(source_type, 0) + 1

        for tier_docs in partitions.values():
            tier_docs.sort(
                key=lambda item: (
                    float(item.get("similarity") or 0.0),
                    int(item.get("to_turn") or item.get("turn_index") or -1),
                    str(item.get("document_id") or ""),
                ),
                reverse=True,
            )

        now_text = _now_kst_string()
        with self._lock:
            previous = deepcopy(self._sessions.get(sid)) if self._sessions.get(sid) else None

        if previous is None or previous.get("status") in {"empty", "discarded"}:
            last_event = "build"
        elif previous.get("status") == "dirty":
            last_event = "dirty_flush"
        else:
            last_event = "rebuild"

        stored = {
            "chat_session_id": sid,
            "documents": normalized_docs,
            "partitions": partitions,
            "source_types": source_types,
            "updated_at": now_text,
            "status": "ready",
            "last_event": last_event,
            "last_event_reason": reason,
            "dirty_reason": None,
            "dirty_turn": None,
            "last_dirty_at": previous.get("last_dirty_at") if previous else None,
            "last_discarded_at": previous.get("last_discarded_at") if previous else None,
            "discard_turn": previous.get("discard_turn") if previous else None,
        }
        with self._lock:
            self._sessions[sid] = stored
        return self.snapshot(sid)

    def mark_session_dirty(
        self,
        chat_session_id: str,
        *,
        reason: str = "source_mutation",
        turn_index: int | None = None,
    ) -> dict:
        if not self.is_shadow_mode_enabled():
            return self.snapshot(chat_session_id)

        sid = " ".join(str(chat_session_id or "").split()).strip()
        if not sid:
            return self._empty_snapshot("")

        now_text = _now_kst_string()
        with self._lock:
            previous = deepcopy(self._sessions.get(sid)) if self._sessions.get(sid) else None
            if previous is None:
                previous = {
                    "chat_session_id": sid,
                    "documents": [],
                    "partitions": {tier: [] for tier in self.ALLOWED_TIERS},
                    "source_types": {},
                    "updated_at": None,
                    "status": "empty",
                    "last_event": None,
                    "last_event_reason": None,
                    "dirty_reason": None,
                    "dirty_turn": None,
                    "last_dirty_at": None,
                    "last_discarded_at": None,
                    "discard_turn": None,
                }
            previous["status"] = "dirty"
            previous["last_event"] = "mark_dirty"
            previous["last_event_reason"] = reason
            previous["dirty_reason"] = reason
            previous["dirty_turn"] = turn_index if isinstance(turn_index, int) else None
            previous["last_dirty_at"] = now_text
            self._sessions[sid] = previous
        return self.snapshot(sid)

    def discard_session(
        self,
        chat_session_id: str,
        *,
        reason: str = "rollback",
        turn_index: int | None = None,
    ) -> dict:
        if not self.is_shadow_mode_enabled():
            return self.snapshot(chat_session_id)

        sid = " ".join(str(chat_session_id or "").split()).strip()
        if not sid:
            return self._empty_snapshot("")

        now_text = _now_kst_string()
        with self._lock:
            previous = deepcopy(self._sessions.get(sid)) if self._sessions.get(sid) else None
            stored = {
                "chat_session_id": sid,
                "documents": [],
                "partitions": {tier: [] for tier in self.ALLOWED_TIERS},
                "source_types": {},
                "updated_at": now_text,
                "status": "discarded",
                "last_event": "rollback_discard",
                "last_event_reason": reason,
                "dirty_reason": None,
                "dirty_turn": None,
                "last_dirty_at": previous.get("last_dirty_at") if previous else None,
                "last_discarded_at": now_text,
                "discard_turn": turn_index if isinstance(turn_index, int) else None,
            }
            self._sessions[sid] = stored
        return self.snapshot(sid)

    def snapshot(self, chat_session_id: str) -> dict:
        sid = " ".join(str(chat_session_id or "").split()).strip()
        if not sid:
            return self._empty_snapshot("")

        if not self.is_shadow_mode_enabled():
            return self._empty_snapshot(sid, status="off")

        with self._lock:
            stored = self._sessions.get(sid)
            if stored is None:
                return self._empty_snapshot(sid)
            runtime_mode = self._runtime_mode
            runtime_updated_at = self._runtime_updated_at
            runtime_reason = self._runtime_reason
            tier_counts = {
                tier: len(stored["partitions"].get(tier, []))
                for tier in self.ALLOWED_TIERS
            }
            return {
                "status": stored.get("status") or ("ready" if stored["documents"] else "empty"),
                "index_version": self.INDEX_VERSION,
                "chat_session_id": sid,
                "session_partitioned": True,
                "runtime_mode": runtime_mode,
                "shadow_write_enabled": runtime_mode == "shadow",
                "runtime_updated_at": runtime_updated_at,
                "runtime_reason": runtime_reason,
                "partition_count": sum(1 for count in tier_counts.values() if count > 0),
                "document_count": len(stored["documents"]),
                "tier_counts": tier_counts,
                "source_type_counts": dict(stored["source_types"]),
                "updated_at": stored["updated_at"],
                "dirty": stored.get("status") == "dirty",
                "last_event": stored.get("last_event"),
                "last_event_reason": stored.get("last_event_reason"),
                "dirty_reason": stored.get("dirty_reason"),
                "dirty_turn": stored.get("dirty_turn"),
                "last_dirty_at": stored.get("last_dirty_at"),
                "last_discarded_at": stored.get("last_discarded_at"),
                "discard_turn": stored.get("discard_turn"),
            }

    def query_session_documents(
        self,
        chat_session_id: str,
        *,
        tier: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        if not self.is_shadow_mode_enabled():
            return []

        sid = " ".join(str(chat_session_id or "").split()).strip()
        if not sid:
            return []

        normalized_tier = " ".join(str(tier or "").split()).strip().lower() or None
        with self._lock:
            stored = self._sessions.get(sid)
            if stored is None:
                return []
            if normalized_tier:
                rows = stored["partitions"].get(normalized_tier, [])
            else:
                rows = stored["documents"]
            copied = deepcopy(rows)

        if isinstance(limit, int) and limit >= 0:
            return copied[:limit]
        return copied

    def get_document(self, chat_session_id: str, document_id: str) -> dict | None:
        if not self.is_shadow_mode_enabled():
            return None

        sid = " ".join(str(chat_session_id or "").split()).strip()
        doc_id = " ".join(str(document_id or "").split()).strip()
        if not sid or not doc_id:
            return None

        with self._lock:
            stored = self._sessions.get(sid)
            if stored is None:
                return None
            for row in stored.get("documents", []):
                if str(row.get("document_id") or "") == doc_id:
                    return deepcopy(row)
        return None

    def clear_session(self, chat_session_id: str) -> None:
        sid = " ".join(str(chat_session_id or "").split()).strip()
        if not sid:
            return
        with self._lock:
            self._sessions.pop(sid, None)

    def clear_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    def is_shadow_mode_enabled(self) -> bool:
        with self._lock:
            return self._runtime_mode == "shadow"

    def _normalize_runtime_mode(self, mode: str) -> str:
        normalized = " ".join(str(mode or "").split()).strip().lower() or "shadow"
        if normalized not in self.ALLOWED_RUNTIME_MODES:
            raise ValueError(f"unsupported runtime mode: {mode}")
        return normalized

    def _normalize_document(self, document: dict, sid: str) -> dict:
        normalized = deepcopy(document)
        normalized["chat_session_id"] = str(normalized.get("chat_session_id") or sid)
        normalized["tier"] = str(normalized.get("tier") or "").strip().lower()
        if normalized.get("tier") not in self.ALLOWED_TIERS:
            normalized["tier"] = "memory"
        return normalized

    def _empty_snapshot(self, chat_session_id: str, *, status: str = "empty") -> dict:
        runtime = self.get_runtime_config()
        return {
            "status": status,
            "index_version": self.INDEX_VERSION,
            "chat_session_id": chat_session_id,
            "session_partitioned": True,
            "runtime_mode": runtime["mode"],
            "shadow_write_enabled": runtime["shadow_write_enabled"],
            "runtime_updated_at": runtime["updated_at"],
            "runtime_reason": runtime["reason"],
            "partition_count": 0,
            "document_count": 0,
            "tier_counts": {tier: 0 for tier in self.ALLOWED_TIERS},
            "source_type_counts": {},
            "updated_at": None,
            "dirty": False,
            "last_event": None,
            "last_event_reason": None,
            "dirty_reason": None,
            "dirty_turn": None,
            "last_dirty_at": None,
            "last_discarded_at": None,
            "discard_turn": None,
        }


retrieval_index_registry = SessionPartitionedRetrievalIndex()