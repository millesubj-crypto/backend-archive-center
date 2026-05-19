from __future__ import annotations

import json

from sqlalchemy.orm import Session

from backend.models import AuditLog, CharacterState, KGTriple, Memory


CONSISTENCY_POLICY_VERSION = "gap2b.v1"


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _normalize_name(value: object) -> str:
    return "".join(str(value or "").strip().casefold().split())


def _load_summary_text(summary_json: str | None) -> str:
    if not summary_json:
        return ""
    try:
        payload = json.loads(summary_json)
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    return _normalize_text(payload.get("turn_summary"))


class ConsistencyChecker:
    def resolve_contradiction(
        self,
        *,
        decision: str,
        reason: str,
        surface: str,
        conflicting_record_id: int | None = None,
        conflicting_record_ids: list[int] | None = None,
        details: dict | None = None,
    ) -> dict:
        return {
            "policy_version": CONSISTENCY_POLICY_VERSION,
            "surface": surface,
            "decision": decision,
            "reason": reason,
            "conflicting_record_id": conflicting_record_id,
            "conflicting_record_ids": conflicting_record_ids or [],
            "details": details or {},
        }

    def check_before_save(
        self,
        db: Session,
        *,
        chat_session_id: str,
        new_memory: dict | None = None,
        new_kg_triple: dict | None = None,
        new_character_state: dict | None = None,
    ) -> dict:
        if isinstance(new_memory, dict):
            return self._check_memory(db, chat_session_id=chat_session_id, new_memory=new_memory)
        if isinstance(new_kg_triple, dict):
            return self._check_kg_triple(db, chat_session_id=chat_session_id, new_kg_triple=new_kg_triple)
        if isinstance(new_character_state, dict):
            return self._check_character_state(chat_session_id=chat_session_id, new_character_state=new_character_state)
        return self.resolve_contradiction(decision="accept", reason="no_payload", surface="unknown")

    def record_audit(
        self,
        db: Session,
        *,
        chat_session_id: str,
        surface: str,
        result: dict,
        source: str,
        target_id: int | None = None,
        payload: dict | None = None,
    ) -> None:
        if str(result.get("decision") or "accept") == "accept":
            return

        details = {
            "policy_version": result.get("policy_version") or CONSISTENCY_POLICY_VERSION,
            "surface": surface,
            "decision": result.get("decision"),
            "reason": result.get("reason"),
            "conflicting_record_id": result.get("conflicting_record_id"),
            "conflicting_record_ids": result.get("conflicting_record_ids") or [],
            "details": result.get("details") or {},
        }
        if isinstance(payload, dict) and payload:
            details["payload"] = payload

        db.add(
            AuditLog(
                event_type="consistency_gate",
                chat_session_id=chat_session_id,
                target_type=surface,
                target_id=target_id or result.get("conflicting_record_id"),
                summary=f"{surface} {result.get('decision')}: {result.get('reason')}",
                details_json=json.dumps(details, ensure_ascii=False),
                source=source,
            )
        )

    def _check_memory(self, db: Session, *, chat_session_id: str, new_memory: dict) -> dict:
        summary_text = _normalize_text(new_memory.get("turn_summary"))
        if not summary_text:
            return self.resolve_contradiction(
                decision="reject",
                reason="empty_turn_summary",
                surface="memory",
            )

        rows = (
            db.query(Memory.id, Memory.summary_json)
            .filter(Memory.chat_session_id == chat_session_id)
            .order_by(Memory.turn_index.desc().nullslast(), Memory.id.desc())
            .limit(150)
            .all()
        )
        for row_id, existing_summary_json in rows:
            if _load_summary_text(existing_summary_json) == summary_text:
                return self.resolve_contradiction(
                    decision="supersede_old",
                    reason="duplicate_memory_summary",
                    surface="memory",
                    conflicting_record_id=row_id,
                )

        return self.resolve_contradiction(decision="accept", reason="ok", surface="memory")

    def _check_kg_triple(self, db: Session, *, chat_session_id: str, new_kg_triple: dict) -> dict:
        subject = str(new_kg_triple.get("subject") or "").strip()
        predicate = str(new_kg_triple.get("predicate") or "").strip()
        obj = str(new_kg_triple.get("object") or "").strip()
        valid_to = new_kg_triple.get("valid_to")

        if not subject or not predicate or not obj:
            return self.resolve_contradiction(
                decision="reject",
                reason="missing_required_fields",
                surface="kg_triple",
                details={"subject": subject, "predicate": predicate, "object": obj},
            )

        if valid_to not in (None, ""):
            return self.resolve_contradiction(decision="accept", reason="historical_row", surface="kg_triple")

        active_rows = (
            db.query(KGTriple)
            .filter(KGTriple.chat_session_id == chat_session_id)
            .filter(KGTriple.subject == subject)
            .filter(KGTriple.predicate == predicate)
            .filter(KGTriple.valid_to.is_(None))
            .all()
        )

        if not active_rows:
            return self.resolve_contradiction(decision="accept", reason="ok", surface="kg_triple")

        identical = [row.id for row in active_rows if str(row.object or "") == obj]
        if identical:
            return self.resolve_contradiction(
                decision="reject",
                reason="duplicate_active_fact",
                surface="kg_triple",
                conflicting_record_id=identical[0],
                conflicting_record_ids=identical,
            )

        return self.resolve_contradiction(
            decision="supersede_old",
            reason="active_fact_conflict",
            surface="kg_triple",
            conflicting_record_id=active_rows[0].id,
            conflicting_record_ids=[row.id for row in active_rows],
        )

    def _check_character_state(self, *, chat_session_id: str, new_character_state: dict) -> dict:
        del chat_session_id
        name = str(new_character_state.get("name") or new_character_state.get("character_name") or "").strip()
        if not name:
            return self.resolve_contradiction(
                decision="reject",
                reason="missing_character_name",
                surface="character_state",
            )

        relationships = new_character_state.get("relationships")
        if not isinstance(relationships, list) or not relationships:
            return self.resolve_contradiction(decision="accept", reason="ok", surface="character_state")

        normalized_name = _normalize_name(name)
        seen_targets: dict[str, str] = {}
        for relation in relationships:
            if not isinstance(relation, dict):
                continue
            target = str(relation.get("target") or "").strip()
            if not target:
                return self.resolve_contradiction(
                    decision="reject",
                    reason="missing_relationship_target",
                    surface="character_state",
                )
            normalized_target = _normalize_name(target)
            if normalized_target == normalized_name:
                return self.resolve_contradiction(
                    decision="reject",
                    reason="self_relationship_target",
                    surface="character_state",
                    details={"character_name": name, "target": target},
                )

            signature = json.dumps(relation, ensure_ascii=False, sort_keys=True)
            previous_signature = seen_targets.get(normalized_target)
            if previous_signature is not None and previous_signature != signature:
                return self.resolve_contradiction(
                    decision="flag_for_review",
                    reason="conflicting_relationship_targets",
                    surface="character_state",
                    details={"character_name": name, "target": target},
                )
            seen_targets[normalized_target] = signature

        return self.resolve_contradiction(decision="accept", reason="ok", surface="character_state")