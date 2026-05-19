import json
import logging

from sqlalchemy.orm import Session

from backend.archive.bridge import ArchiveBridge, _normalize_kg_entity_label, _safe_session_id
from backend.database import SessionLocal
from backend.integrity.consistency_checker import ConsistencyChecker
from backend.integrity.memory_merge import MemoryMerger
from backend.models import AuditLog, KGTriple, Memory

logger = logging.getLogger(__name__)


class ArchiveStoreBridge(ArchiveBridge):
    def save_memory(
        self,
        turn_index: int,
        summary_json: dict,
        wing: str | None = None,
        room: str | None = None,
        chat_session_id: str | None = None,
    ) -> None:
        if isinstance(summary_json, str):
            try:
                summary_json = json.loads(summary_json)
            except Exception:
                logger.warning("ArchiveStoreBridge.save_memory string parse failed, wrapping")
                summary_json = {"turn_summary": summary_json, "importance_score": 5}
        if not isinstance(summary_json, dict):
            logger.warning(
                "ArchiveStoreBridge.save_memory skipped: summary_json is not a dict (type=%s)",
                type(summary_json).__name__,
            )
            return

        session_id = _safe_session_id(chat_session_id)
        archive_hint = self._extract_archive_hint(summary_json)

        importance = summary_json.get("importance_score", 5)
        try:
            importance = float(importance)
        except Exception:
            importance = 5.0

        save_wing = wing or archive_hint.get("wing") or "wing_general"
        save_room = room or archive_hint.get("room") or "hall_events"
        embed_text = summary_json.get("turn_summary") or json.dumps(summary_json, ensure_ascii=False)
        embedding_vec = self._get_embedding(embed_text)
        embedding_json = json.dumps(embedding_vec) if embedding_vec is not None else None

        # Sprint 3-C-3: embedding model provenance
        embedding_model_name = None
        try:
            from backend.settings import settings as _s
            embedding_model_name = (_s.PROJECT_EMBEDDING_MODEL or "").strip() or None
        except Exception:
            pass

        # Phase 5-1: evidence_excerpts, emotional_intensity, narrative_significance
        evidence_json = None
        emotional_intensity_val = None
        narrative_significance_val = None
        try:
            ev = summary_json.get("evidence_excerpts")
            if isinstance(ev, list) and ev:
                evidence_json = json.dumps(ev, ensure_ascii=False)
            ei = summary_json.get("emotional_intensity")
            if ei is not None:
                emotional_intensity_val = max(0.0, min(1.0, float(ei)))
            ns = summary_json.get("narrative_significance")
            if ns is not None:
                narrative_significance_val = max(0.0, min(1.0, float(ns)))
        except Exception as exc:
            logger.warning("Phase 5-1 field extraction failed (non-fatal): %s", exc)

        # Phase 5-2: emotional intensity threshold → importance boost
        base_importance = importance
        emotional_boost_val = 0.0
        try:
            if emotional_intensity_val is not None:
                if emotional_intensity_val >= 0.90:
                    emotional_boost_val = 2.0
                elif emotional_intensity_val >= 0.70:
                    emotional_boost_val = 1.0
                if emotional_boost_val > 0:
                    importance = min(10.0, importance + emotional_boost_val)
                    logger.info(
                        "Phase 5-2 importance boost: base=%.1f, boost=+%.0f, final=%.1f (emotional=%.2f, turn=%d)",
                        base_importance, emotional_boost_val, importance, emotional_intensity_val, turn_index,
                    )
        except Exception as exc:
            logger.warning("Phase 5-2 importance boost failed (non-fatal): %s", exc)
            emotional_boost_val = 0.0

        db: Session = SessionLocal()
        try:
            consistency_checker = ConsistencyChecker()
            memory_gate = consistency_checker.check_before_save(
                db,
                chat_session_id=session_id,
                new_memory={
                    "turn_summary": summary_json.get("turn_summary"),
                },
            )
            consistency_checker.record_audit(
                db,
                chat_session_id=session_id,
                surface="memory",
                result=memory_gate,
                source="archive_store",
                payload={"turn_index": turn_index},
            )
            if memory_gate.get("decision") == "reject":
                db.commit()
                logger.info(
                    "ArchiveStoreBridge.save_memory skipped by consistency gate (turn=%d, session=%s, reason=%s)",
                    turn_index,
                    session_id,
                    memory_gate.get("reason"),
                )
                return

            merger = MemoryMerger()
            merge_candidate = merger.should_merge(
                db,
                chat_session_id=session_id,
                turn_summary=summary_json.get("turn_summary") or embed_text,
                embedding=embedding_vec,
                archive_wing=save_wing,
                archive_room=save_room,
                cosine_fn=self._cosine,
            )
            if merge_candidate.get("decision") == "merge" and isinstance(merge_candidate.get("match"), Memory):
                merge_result = merger.merge_memories(
                    db,
                    existing_row=merge_candidate["match"],
                    turn_index=turn_index,
                    summary_json=summary_json,
                    importance=importance,
                    embedding_json=embedding_json,
                    embedding_model_name=embedding_model_name,
                    archive_wing=save_wing,
                    archive_room=save_room,
                    evidence_json=evidence_json,
                    emotional_intensity=emotional_intensity_val,
                    narrative_significance=narrative_significance_val,
                    emotional_boost=emotional_boost_val if emotional_boost_val > 0 else None,
                    similarity=float(merge_candidate.get("similarity") or 0.0),
                    merge_reason=str(merge_candidate.get("reason") or "semantic_similarity"),
                    source="archive_store",
                )
                db.commit()
                logger.info(
                    "ArchiveStoreBridge.save_memory merged (turn=%d, session=%s, memory_id=%s, reason=%s, similarity=%.3f)",
                    turn_index,
                    session_id,
                    merge_result.get("memory_id"),
                    merge_result.get("merge_reason"),
                    float(merge_result.get("similarity") or 0.0),
                )
                return

            row = Memory(
                chat_session_id=session_id,
                turn_index=turn_index,
                summary_json=json.dumps(summary_json, ensure_ascii=False),
                embedding=embedding_json,
                embedding_model=embedding_model_name,
                importance=importance,
                emotional_boost=emotional_boost_val if emotional_boost_val > 0 else None,
                evidence=evidence_json,
                emotional_intensity=emotional_intensity_val,
                narrative_significance=narrative_significance_val,
            )
            row.archive_wing = save_wing
            row.archive_room = save_room
            db.add(row)
            db.commit()
            logger.info(
                "ArchiveStoreBridge.save_memory complete (turn=%d, session=%s, wing=%s, room=%s)",
                turn_index,
                session_id,
                save_wing,
                save_room,
            )
        except Exception as exc:
            db.rollback()
            logger.error("ArchiveStoreBridge.save_memory failed: %s", exc)
        finally:
            db.close()

    def save_kg_triples(self, triples: list[dict], source_turn: int, chat_session_id: str | None = None) -> None:
        if not isinstance(triples, list) or not triples:
            return

        session_id = _safe_session_id(chat_session_id)

        db: Session = SessionLocal()
        try:
            inserted_count = 0
            skipped_duplicate_count = 0
            superseded_count = 0
            consistency_checker = ConsistencyChecker()
            for triple in triples:
                if not isinstance(triple, dict):
                    continue
                subject = _normalize_kg_entity_label(triple.get("subject"), allow_player_token=True)
                predicate = str(triple.get("predicate", "") or "").strip()[:255]
                obj = _normalize_kg_entity_label(triple.get("object"), allow_player_token=True)
                valid_from = self._coerce_turn_value(triple.get("valid_from"))
                valid_to = self._coerce_turn_value(triple.get("valid_to"))

                if not subject or not predicate or not obj:
                    skipped_duplicate_count += 1
                    continue

                kg_gate = consistency_checker.check_before_save(
                    db,
                    chat_session_id=session_id,
                    new_kg_triple={
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "valid_to": valid_to,
                    },
                )
                consistency_checker.record_audit(
                    db,
                    chat_session_id=session_id,
                    surface="kg_triple",
                    result=kg_gate,
                    source="critic",
                    payload={
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "source_turn": source_turn,
                    },
                )
                if kg_gate.get("decision") == "reject":
                    skipped_duplicate_count += 1
                    continue

                if subject and predicate and valid_to is None:
                    active_rows = (
                        db.query(KGTriple)
                        .filter(KGTriple.chat_session_id == session_id)
                        .filter(KGTriple.subject == subject)
                        .filter(KGTriple.predicate == predicate)
                        .filter(KGTriple.valid_to.is_(None))
                        .all()
                    )

                    if any((existing.object or "") == obj for existing in active_rows):
                        skipped_duplicate_count += 1
                        continue

                    for existing in active_rows:
                        existing_source_turn = self._coerce_turn_value(existing.source_turn)
                        if existing_source_turn is not None and existing_source_turn > source_turn:
                            continue
                        if (existing.object or "") == obj:
                            continue
                        existing.valid_to = source_turn
                        db.add(
                            AuditLog(
                                event_type="kg_triple_superseded",
                                chat_session_id=session_id,
                                target_type="kg_triple",
                                target_id=existing.id,
                                summary=f"Superseded KG triple {subject}:{predicate}",
                                details_json=json.dumps(
                                    {
                                        "policy_version": "gap1a.v1",
                                        "subject": subject,
                                        "predicate": predicate,
                                        "previous_object": existing.object,
                                        "next_object": obj,
                                        "source_turn": source_turn,
                                    },
                                    ensure_ascii=False,
                                ),
                                source="critic",
                            )
                        )
                        superseded_count += 1

                db.add(
                    KGTriple(
                        chat_session_id=session_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        valid_from=valid_from,
                        valid_to=valid_to,
                        source_turn=source_turn,
                    )
                )
                inserted_count += 1
            db.commit()
            logger.info(
                "ArchiveStoreBridge.save_kg_triples complete (turn=%d, session=%s, inserted=%d, superseded=%d, skipped_duplicates=%d)",
                source_turn,
                session_id,
                inserted_count,
                superseded_count,
                skipped_duplicate_count,
            )
        except Exception as exc:
            db.rollback()
            logger.error("ArchiveStoreBridge.save_kg_triples failed: %s", exc)
        finally:
            db.close()

    def _coerce_turn_value(self, value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except Exception:
            return None
