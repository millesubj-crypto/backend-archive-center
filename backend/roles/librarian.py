import logging

from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.critic_sanitizer import sanitize_assistant_chatlog_text
from backend.database import SessionLocal
from backend.models import (
    ActiveState,
    CanonicalStateLayer,
    ArcSummary,
    ChapterSummary,
    CharacterEvent,
    CharacterState,
    ChatLog,
    CriticFeedback,
    DirectEvidenceRecord,
    DEFAULT_SESSION_ID,
    EpisodeSummary,
    EffectiveInputLog,
    SagaDigest,
    GuidanceCompactRecord,
    GuidancePlanState,
    KGTriple,
    MaintenancePassState,
    Memory,
    PendingThread,
    SessionActiveScopeCache,
    Storyline,
    WorldRule,
)

logger = logging.getLogger(__name__)


def _build_turn_role_payloads(user_content: str, assistant_content: str) -> tuple[list[tuple[str, str]], str, str]:
    safe_user_content = str(user_content or "")
    safe_assistant_content = sanitize_assistant_chatlog_text(assistant_content)
    role_payloads: list[tuple[str, str]] = []
    if safe_user_content:
        role_payloads.append(("user", safe_user_content))
    if safe_assistant_content:
        role_payloads.append(("assistant", safe_assistant_content))
    return role_payloads, safe_user_content, safe_assistant_content


class Librarian:

    def save_turn(
        self,
        turn_index: int,
        user_content: str,
        assistant_content: str,
        chat_session_id: str | None = None,
    ) -> dict:
        """유저/어시스턴트 원문을 chat_logs에 저장한다.

        둘 다 비어 있으면 no-op. 실제 commit 성공 여부를 호출자에게 반환한다.
        """
        session_id = (chat_session_id or "").strip() or DEFAULT_SESSION_ID
        role_payloads, _, _ = _build_turn_role_payloads(user_content, assistant_content)
        result = {
            "ok": True,
            "status": "ignored",
            "turn_index": turn_index,
            "chat_session_id": session_id,
            "inserted_roles": [],
            "error": None,
        }

        if not role_payloads:
            logger.debug("save_turn(%d): 내용 없음, 건너뜀", turn_index)
            return result

        db: Session = SessionLocal()
        try:
            inserted_roles: list[str] = []
            for role, content in role_payloads:
                db.add(ChatLog(
                    chat_session_id=session_id,
                    turn_index=turn_index,
                    role=role,
                    content=content,
                ))
                inserted_roles.append(role)
            db.commit()
            result["status"] = "saved"
            result["inserted_roles"] = inserted_roles
            logger.info("save_turn(%d, session=%s) 완료", turn_index, session_id)
        except Exception as exc:
            db.rollback()
            result["ok"] = False
            result["status"] = "error"
            result["error"] = str(exc)
            logger.error("save_turn(%d) 실패: %s", turn_index, exc)
        finally:
            db.close()
        return result

    def repair_turn(
        self,
        turn_index: int,
        user_content: str,
        assistant_content: str,
        chat_session_id: str | None = None,
        dry_run: bool = False,
    ) -> dict:
        """chat_logs 누락을 점검하고, 없는 role만 idempotent 하게 복구한다."""
        session_id = (chat_session_id or "").strip() or DEFAULT_SESSION_ID
        role_payloads, _, _ = _build_turn_role_payloads(user_content, assistant_content)
        result = {
            "ok": True,
            "status": "ignored",
            "turn_index": turn_index,
            "chat_session_id": session_id,
            "dry_run": bool(dry_run),
            "provided_roles": [role for role, _ in role_payloads],
            "existing_roles": [],
            "missing_roles": [],
            "already_present_roles": [],
            "repaired_roles": [],
            "conflict_roles": [],
            "conflicts": [],
            "error": None,
        }
        if not role_payloads:
            return result

        db: Session = SessionLocal()
        try:
            existing_rows = (
                db.query(ChatLog)
                .filter(ChatLog.chat_session_id == session_id)
                .filter(ChatLog.turn_index == turn_index)
                .all()
            )
            existing_by_role: dict[str, list[str]] = {}
            for row in existing_rows:
                existing_by_role.setdefault(str(row.role or ""), []).append(row.content or "")
            result["existing_roles"] = sorted([role for role in existing_by_role.keys() if role])

            for role, content in role_payloads:
                existing_contents = existing_by_role.get(role, [])
                if content in existing_contents:
                    result["already_present_roles"].append(role)
                    continue
                if existing_contents:
                    result["conflict_roles"].append(role)
                    result["conflicts"].append({
                        "role": role,
                        "existing_count": len(existing_contents),
                        "existing_preview": [text[:120] for text in existing_contents[:2]],
                    })
                    continue
                result["missing_roles"].append(role)

            if result["missing_roles"] and not dry_run:
                for role, content in role_payloads:
                    if role not in result["missing_roles"]:
                        continue
                    db.add(ChatLog(
                        chat_session_id=session_id,
                        turn_index=turn_index,
                        role=role,
                        content=content,
                    ))
                db.commit()
                result["repaired_roles"] = list(result["missing_roles"])

            if result["repaired_roles"] and result["conflict_roles"]:
                result["status"] = "partial_repaired_conflict"
            elif result["repaired_roles"]:
                result["status"] = "repaired"
            elif result["missing_roles"] and dry_run and result["conflict_roles"]:
                result["status"] = "missing_with_conflict"
            elif result["missing_roles"] and dry_run:
                result["status"] = "missing_detected"
            elif result["conflict_roles"]:
                result["status"] = "conflict"
            elif result["already_present_roles"]:
                result["status"] = "already_present"
        except Exception as exc:
            db.rollback()
            result["ok"] = False
            result["status"] = "error"
            result["error"] = str(exc)
            logger.error("repair_turn(%d) 실패: %s", turn_index, exc)
        finally:
            db.close()
        return result

    def rollback(self, from_turn_index: int, chat_session_id: str | None = None) -> dict:
        """from_turn_index 이상의 현재 session turn 산출물을 보수적으로 삭제한다.

        L-3b: guidance layer(GuidancePlanState, GuidanceCompactRecord, MaintenancePassState)도
        함께 정리한다.
          - GuidancePlanState.last_turn >= from_turn_index → state_status="empty" 로 무효화
            (story_plan_json / director_json은 보존하여 stale 참조 가능, last_turn=-1 리셋)
          - GuidanceCompactRecord.compacted_turn >= from_turn_index → 삭제
            (rollback 이후 존재해서는 안 될 compact 기록)
          - MaintenancePassState.turn_index >= from_turn_index → 삭제

        삭제 결과를 dict로 반환한다. 실패 시에도 안전한 dict를 반환한다.
        """
        session_id = (chat_session_id or "").strip() or DEFAULT_SESSION_ID

        result = {
            "from_turn_index": from_turn_index,
            "chat_session_id": session_id,
            "deleted_chat_logs": 0,
            "deleted_effective_inputs": 0,
            "deleted_memories": 0,
            "deleted_direct_evidence_records": 0,
            "deleted_kg_triples": 0,
            "deleted_active_states": 0,
            "deleted_canonical_state_layers": 0,
            "deleted_episode_summaries": 0,
            "deleted_chapter_summaries": 0,
            "deleted_arc_summaries": 0,
            "deleted_saga_digests": 0,
            "deleted_storylines": 0,
            "deleted_character_states": 0,
            "deleted_character_events": 0,
            "deleted_world_rules": 0,
            "deleted_pending_threads": 0,
            "deleted_feedbacks": 0,
            "deleted_active_scope_cache": 0,
            # L-3b: guidance layer 정리 결과
            "guidance_state_invalidated": False,
            "deleted_compact_records": 0,
            "deleted_maintenance_passes": 0,
        }

        db: Session = SessionLocal()
        try:
            memory_ids = [
                row[0]
                for row in (
                    db.query(Memory.id)
                    .filter(Memory.chat_session_id == session_id)
                    .filter(Memory.turn_index >= from_turn_index)
                    .all()
                )
            ]
            kg_ids = [
                row[0]
                for row in (
                    db.query(KGTriple.id)
                    .filter(KGTriple.chat_session_id == session_id)
                    .filter(KGTriple.source_turn >= from_turn_index)
                    .all()
                )
            ]

            result["deleted_chat_logs"] = (
                db.query(ChatLog)
                .filter(ChatLog.chat_session_id == session_id)
                .filter(ChatLog.turn_index >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_effective_inputs"] = (
                db.query(EffectiveInputLog)
                .filter(EffectiveInputLog.chat_session_id == session_id)
                .filter(EffectiveInputLog.turn_index >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_memories"] = (
                db.query(Memory)
                .filter(Memory.chat_session_id == session_id)
                .filter(Memory.turn_index >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_direct_evidence_records"] = (
                db.query(DirectEvidenceRecord)
                .filter(DirectEvidenceRecord.chat_session_id == session_id)
                .filter(or_(
                    DirectEvidenceRecord.source_turn_end >= from_turn_index,
                    DirectEvidenceRecord.turn_anchor >= from_turn_index,
                ))
                .delete(synchronize_session=False)
            )
            result["deleted_kg_triples"] = (
                db.query(KGTriple)
                .filter(KGTriple.chat_session_id == session_id)
                .filter(KGTriple.source_turn >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_active_states"] = (
                db.query(ActiveState)
                .filter(ActiveState.chat_session_id == session_id)
                .filter(ActiveState.turn_index >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_canonical_state_layers"] = (
                db.query(CanonicalStateLayer)
                .filter(CanonicalStateLayer.chat_session_id == session_id)
                .filter(CanonicalStateLayer.turn_index >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_episode_summaries"] = (
                db.query(EpisodeSummary)
                .filter(EpisodeSummary.chat_session_id == session_id)
                .filter(EpisodeSummary.to_turn >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_chapter_summaries"] = (
                db.query(ChapterSummary)
                .filter(ChapterSummary.chat_session_id == session_id)
                .filter(ChapterSummary.to_turn >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_arc_summaries"] = (
                db.query(ArcSummary)
                .filter(ArcSummary.chat_session_id == session_id)
                .filter(ArcSummary.to_turn >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_saga_digests"] = (
                db.query(SagaDigest)
                .filter(SagaDigest.chat_session_id == session_id)
                .filter(SagaDigest.to_turn >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_storylines"] = (
                db.query(Storyline)
                .filter(Storyline.chat_session_id == session_id)
                .filter(or_(
                    Storyline.first_turn >= from_turn_index,
                    Storyline.last_turn >= from_turn_index,
                    Storyline.last_evidence_turn >= from_turn_index,
                ))
                .delete(synchronize_session=False)
            )
            result["deleted_character_states"] = (
                db.query(CharacterState)
                .filter(CharacterState.chat_session_id == session_id)
                .filter(CharacterState.turn_index >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_character_events"] = (
                db.query(CharacterEvent)
                .filter(CharacterEvent.chat_session_id == session_id)
                .filter(CharacterEvent.turn_index >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_world_rules"] = (
                db.query(WorldRule)
                .filter(WorldRule.chat_session_id == session_id)
                .filter(WorldRule.source_turn >= from_turn_index)
                .delete(synchronize_session=False)
            )
            result["deleted_pending_threads"] = (
                db.query(PendingThread)
                .filter(PendingThread.chat_session_id == session_id)
                .filter(or_(
                    PendingThread.source_turn >= from_turn_index,
                    PendingThread.last_seen_turn >= from_turn_index,
                ))
                .delete(synchronize_session=False)
            )
            if memory_ids:
                result["deleted_feedbacks"] += (
                    db.query(CriticFeedback)
                    .filter(CriticFeedback.chat_session_id == session_id)
                    .filter(CriticFeedback.target_type == "memory")
                    .filter(CriticFeedback.target_id.in_(memory_ids))
                    .delete(synchronize_session=False)
                )
            if kg_ids:
                result["deleted_feedbacks"] += (
                    db.query(CriticFeedback)
                    .filter(CriticFeedback.chat_session_id == session_id)
                    .filter(CriticFeedback.target_type == "kg_triple")
                    .filter(CriticFeedback.target_id.in_(kg_ids))
                    .delete(synchronize_session=False)
                )

            # ── L-3b: guidance layer 정리 ────────────────────────────────
            # 1. GuidancePlanState 무효화
            #    last_turn >= from_turn_index면 rollback 이후의 state → empty로 리셋
            #    story_plan_json / director_json은 보존 (stale이지만 참조 가능)
            try:
                gps: GuidancePlanState | None = (
                    db.query(GuidancePlanState)
                    .filter(GuidancePlanState.chat_session_id == session_id)
                    .first()
                )
                if gps is not None and (gps.last_turn is not None) and gps.last_turn >= from_turn_index:
                    gps.story_plan_json = None
                    gps.director_json = None
                    gps.warnings_json = None
                    gps.state_status = "empty"
                    gps.last_turn = -1
                    result["guidance_state_invalidated"] = True
            except Exception as g_exc:
                logger.warning("L-3b GuidancePlanState 무효화 실패 (non-fatal): %s", g_exc)

            # 1-b. SessionActiveScopeCache 삭제
            try:
                result["deleted_active_scope_cache"] = (
                    db.query(SessionActiveScopeCache)
                    .filter(SessionActiveScopeCache.chat_session_id == session_id)
                    .delete(synchronize_session=False)
                )
            except Exception as g_exc:
                logger.warning("rollback SessionActiveScopeCache 삭제 실패 (non-fatal): %s", g_exc)

            # 2. GuidanceCompactRecord 삭제 (compacted_turn >= from_turn_index)
            try:
                result["deleted_compact_records"] = (
                    db.query(GuidanceCompactRecord)
                    .filter(GuidanceCompactRecord.chat_session_id == session_id)
                    .filter(GuidanceCompactRecord.compacted_turn >= from_turn_index)
                    .delete(synchronize_session=False)
                )
            except Exception as g_exc:
                logger.warning("L-3b GuidanceCompactRecord 삭제 실패 (non-fatal): %s", g_exc)

            # 3. MaintenancePassState 삭제 (turn_index >= from_turn_index)
            try:
                result["deleted_maintenance_passes"] = (
                    db.query(MaintenancePassState)
                    .filter(MaintenancePassState.chat_session_id == session_id)
                    .filter(MaintenancePassState.turn_index >= from_turn_index)
                    .delete(synchronize_session=False)
                )
            except Exception as g_exc:
                logger.warning("L-3b MaintenancePassState 삭제 실패 (non-fatal): %s", g_exc)
            # ─────────────────────────────────────────────────────────────

            db.commit()
            logger.info(
                "rollback(%d, session=%s) 완료 — logs:%d, effective:%d, memories:%d, evidence:%d, kg:%d, active:%d, canonical:%d, episodes:%d, chapters:%d, arcs:%d, sagas:%d, storylines:%d, chars:%d, events:%d, rules:%d, hooks:%d, feedback:%d, scope:%d | guidance: invalidated=%s compact=%d mnt=%d",
                from_turn_index,
                session_id,
                result["deleted_chat_logs"],
                result["deleted_effective_inputs"],
                result["deleted_memories"],
                result["deleted_direct_evidence_records"],
                result["deleted_kg_triples"],
                result["deleted_active_states"],
                result["deleted_canonical_state_layers"],
                result["deleted_episode_summaries"],
                result["deleted_chapter_summaries"],
                result["deleted_arc_summaries"],
                result["deleted_saga_digests"],
                result["deleted_storylines"],
                result["deleted_character_states"],
                result["deleted_character_events"],
                result["deleted_world_rules"],
                result["deleted_pending_threads"],
                result["deleted_feedbacks"],
                result["deleted_active_scope_cache"],
                result["guidance_state_invalidated"],
                result["deleted_compact_records"],
                result["deleted_maintenance_passes"],
            )
        except Exception as exc:
            db.rollback()
            logger.error("rollback(%d) 실패: %s", from_turn_index, exc)
        finally:
            db.close()

        return result
