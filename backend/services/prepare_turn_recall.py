"""backend/services/prepare_turn_recall.py

Prepare-turn recall bundle builder.
Extracted from backend/main.py as part of Step 21.5 accelerated slice.

This module MUST NOT import backend.main.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class BundleRecallDeps:
    """Dependency bundle for prepare-turn recall."""
    storyteller_search: Callable[..., Any]
    storyteller_format_for_injection: Callable[..., Any]
    palace_recall_kg_triples: Callable[..., Any]
    search_episode_summaries: Callable[..., Any]
    search_chapter_summaries: Callable[..., Any]
    search_arc_summaries: Callable[..., Any]
    search_saga_digests: Callable[..., Any]
    simple_extract_entities: Callable[..., Any]
    build_ann_candidate_snapshot: Callable[..., Any]
    build_recall_intent_contract: Callable[..., Any]
    get_intent_routing_runtime_config: Callable[..., Any]
    execute_per_intent_retrieval_shadow: Callable[..., Any]
    build_tier_priority_verification: Callable[..., Any]
    build_captured_session_replay_gate: Callable[..., Any]
    build_guarded_default_takeover: Callable[..., Any]
    build_enforced_default_takeover: Callable[..., Any]
    apply_ann_default_takeover: Callable[..., Any]
    build_takeover_recall_payload: Callable[..., Any]
    bundle_retrieval_documents: Callable[..., Any]
    update_retrieval_index: Callable[..., Any]
    build_validation_gates: Callable[..., Any]
    refresh_step21_validation_contract: Callable[..., Any]
    s1b_version: str
    s1c_version: str
    s1d_version: str
    t1b_version: str
    s1g_version: str
    t1d_version: str
    retrieval_document_schema: dict
    q3c_intent_priority_order: tuple


def bundle_recall(
    sid: str,
    user_input: str,
    continuity_query: str,
    top_k: int,
    *,
    continuity_trigger_mode: str = "none",
    context_window_profile: str = "mid_context_300k",
    context_window_profile_resolution_source: str = "max_injection_chars_proxy",
    max_injection_chars: int = 3000,
    max_input_context_chars: int = 800,
    session_state_data=None,
    narrative_control_data=None,
    continuity_pack_data=None,
    resume_pack_data=None,
    client_meta: dict | None = None,
    deps: BundleRecallDeps,
    logger,
) -> dict | None:
    """M-2b/P-2c/P-3e/P-4/Q-1a: semantic search + hierarchy recall???섎굹??dict濡?臾띠뼱 諛섑솚.

    諛섑솚 ?뺤떇:
    {
      "search":  { items, injection_text, memory_count, fallback_count, total_count }
      "kg":      { items, count, entities_sent }
    "episode": { items, count }
    "chapter": { items, count }
    "arc":     { items, count }
    "saga":    { items, count }
        "intent_contract": { version, routing_mode, intents }
    "document_schema": { version, tiers, fields }
    "documents": [ unified retrieval docs... ]
    "document_count": int
      "entities_extracted": [...]
    }
    ?ㅽ뙣 ??None 諛섑솚 (plugin fallback?쇰줈 泥섎━).
    """
    try:
        effective_query = (continuity_query.strip() or user_input.strip())[:2000]
        if not effective_query:
            return None

        # ?? 1. semantic search ?????????????????????????????????????????????
        search_items: list[dict] = []
        injection_text: str = ""
        memory_count = 0
        fallback_count = 0
        try:
            raw_items = deps.storyteller_search(
                user_input=effective_query,
                wing=None,
                top_k=top_k,
                chat_session_id=sid,
            )
            search_items = raw_items
            injection_text = deps.storyteller_format_for_injection(raw_items)
            memory_count = sum(1 for i in raw_items if i.get("source") == "memory")
            fallback_count = sum(1 for i in raw_items if i.get("source") == "chat_log")
        except Exception as exc:
            logger.warning("[prepare-turn] bundle search failed: %s", exc)

        # ?? 2. KG recall ???????????????????????????????????????????????????
        entities = deps.simple_extract_entities(user_input or continuity_query)
        kg_items: list[dict] = []
        if entities:
            try:
                kg_items = deps.palace_recall_kg_triples(
                    chat_session_id=sid,
                    entities=entities[:20],
                    limit=20,
                    current_turn=None,
                )
            except Exception as exc:
                logger.warning("[prepare-turn] bundle kg_recall failed: %s", exc)

        # ?? 3. episode recall ??????????????????????????????????????????????
        episode_items: list[dict] = []
        try:
            episodes = deps.search_episode_summaries(
                query=effective_query[:2000],
                chat_session_id=sid,
                top_k=top_k,  # _bundle_recall ?몄텧?먯뿉???꾨떖??媛??ъ슜
            )
            episode_items = episodes or []
        except Exception as exc:
            logger.warning("[prepare-turn] bundle episode_recall failed: %s", exc)

        # ?? 4. chapter recall (P-2c: read-only bundle exposure only) ?????
        chapter_items: list[dict] = []
        try:
            chapters = deps.search_chapter_summaries(
                query=effective_query[:2000],
                chat_session_id=sid,
                top_k=top_k,
            )
            chapter_items = chapters or []
        except Exception as exc:
            logger.warning("[prepare-turn] bundle chapter_recall failed: %s", exc)

        # ?? 5. arc recall (P-3e: read-only bundle exposure only) ?????????
        arc_items: list[dict] = []
        try:
            arcs = deps.search_arc_summaries(
                query=effective_query[:2000],
                chat_session_id=sid,
                top_k=top_k,
            )
            arc_items = arcs or []
        except Exception as exc:
            logger.warning("[prepare-turn] bundle arc_recall failed: %s", exc)

        # ?? 6. saga recall (Q-1a: unified retrieval schema??saga tier ?ы븿) ??
        saga_items: list[dict] = []
        try:
            sagas = deps.search_saga_digests(
                query=effective_query[:2000],
                chat_session_id=sid,
                top_k=top_k,
            )
            saga_items = sagas or []
        except Exception as exc:
            logger.warning("[prepare-turn] bundle saga_recall failed: %s", exc)

        ann_candidate_snapshot = deps.build_ann_candidate_snapshot(
            sid,
            effective_query[:2000],
            top_k=top_k,
            context_window_profile=context_window_profile,
        )
        intent_contract = deps.build_recall_intent_contract(
            top_k=top_k,
            user_input=user_input,
            continuity_query=continuity_query,
            continuity_trigger_mode=continuity_trigger_mode,
            context_window_profile=context_window_profile,
            context_window_profile_resolution_source=context_window_profile_resolution_source,
            max_injection_chars=max_injection_chars,
            max_input_context_chars=max_input_context_chars,
            intent_routing_runtime=deps.get_intent_routing_runtime_config(),
            session_state_data=session_state_data,
            narrative_control_data=narrative_control_data,
            continuity_pack_data=continuity_pack_data,
            resume_pack_data=resume_pack_data,
            ann_candidate_snapshot=ann_candidate_snapshot,
        )
        intent_execution_shadow = deps.execute_per_intent_retrieval_shadow(
            sid=sid,
            intent_contract=intent_contract,
        )
        shadow_budget_detail = (intent_execution_shadow or {}).get("budget_enforcement") or {}
        if isinstance(intent_contract, dict):
            intent_contract["routing_shadow_execution"] = {
                "version": deps.s1b_version,
                "status": str((intent_execution_shadow or {}).get("status") or "off"),
                "mode": "shadow_execution_only",
                "runtime_mode": str((intent_execution_shadow or {}).get("runtime_mode") or "single_query_shared"),
                "executed_intent_count": int((intent_execution_shadow or {}).get("executed_intent_count") or 0),
                "total_candidates": int((intent_execution_shadow or {}).get("total_candidates") or 0),
                "warnings_count": len((intent_execution_shadow or {}).get("warnings") or []),
            }
            shadow_merge = (intent_execution_shadow or {}).get("cross_intent_merge") or {}
            intent_contract["routing_shadow_merge"] = {
                "version": deps.s1c_version,
                "status": str(shadow_merge.get("status") or "off"),
                "mode": "shadow_merge_only",
                "runtime_mode": str((intent_execution_shadow or {}).get("runtime_mode") or "single_query_shared"),
                "strategy": "document_id_then_source_row",
                "priority_order": [str(name) for name in (shadow_merge.get("priority_order") or [])],
                "selected_count": int(shadow_merge.get("selected_count") or 0),
                "suppressed_count": int(shadow_merge.get("suppressed_count") or 0),
            }
            shadow_trace = (intent_execution_shadow or {}).get("trace") or {}
            shadow_trace_summary = shadow_trace.get("summary") or {}
            intent_contract["routing_shadow_trace"] = {
                "version": deps.s1d_version,
                "status": str(shadow_trace.get("status") or "off"),
                "mode": "shadow_trace_only",
                "runtime_mode": str((intent_execution_shadow or {}).get("runtime_mode") or "single_query_shared"),
                "selection_event_count": len((shadow_trace.get("selection_events") or [])),
                "budget_event_count": len((shadow_trace.get("budget_events") or [])),
                "selected_count": int(shadow_trace_summary.get("selected_count") or 0),
                "suppressed_count": int(shadow_trace_summary.get("suppressed_count") or 0),
            }
            shadow_budget = shadow_budget_detail
            intent_contract["routing_shadow_budget"] = {
                "version": deps.t1b_version,
                "status": str(shadow_budget.get("status") or "off"),
                "mode": str(shadow_budget.get("mode") or "policy_only"),
                "runtime_mode": str((intent_execution_shadow or {}).get("runtime_mode") or "single_query_shared"),
                "selected_count_before": int(shadow_budget.get("selected_count_before") or 0),
                "selected_count_after": int(shadow_budget.get("selected_count_after") or 0),
                "dropped_count": int(shadow_budget.get("dropped_count") or 0),
                "event_count": len((shadow_budget.get("events") or [])),
            }
            shadow_temporal = (intent_execution_shadow or {}).get("temporal_scoring") or {}
            intent_contract["routing_shadow_temporal"] = {
                "version": deps.s1g_version,
                "status": str(shadow_temporal.get("status") or "off"),
                "mode": str(shadow_temporal.get("mode") or "shadow_temporal_scoring_only"),
                "runtime_mode": str((intent_execution_shadow or {}).get("runtime_mode") or "single_query_shared"),
                "context_window_profile": str(shadow_temporal.get("context_window_profile") or "mid_context_300k"),
                "applied_intent_count": int(shadow_temporal.get("applied_intent_count") or 0),
                "reordered_intent_count": int(shadow_temporal.get("reordered_intent_count") or 0),
            }
            tier_priority_verification = deps.build_tier_priority_verification(
                runtime_mode=str((intent_execution_shadow or {}).get("runtime_mode") or "single_query_shared"),
                context_window_profile=str(
                    (intent_contract.get("routing_shadow_temporal") or {}).get("context_window_profile")
                    or context_window_profile
                    or "mid_context_300k"
                ),
                packet_budget_policy=intent_contract.get("packet_budget_policy") or {},
                budget_enforcement=shadow_budget_detail,
            )
            intent_contract["routing_shadow_tier_priority"] = {
                "version": deps.t1d_version,
                "status": str(tier_priority_verification.get("status") or "off"),
                "mode": str(tier_priority_verification.get("mode") or "verification_only"),
                "runtime_mode": str(tier_priority_verification.get("runtime_mode") or "single_query_shared"),
                "context_window_profile": str(tier_priority_verification.get("context_window_profile") or "mid_context_300k"),
                "saga_collision_policy": str(tier_priority_verification.get("saga_collision_policy") or "none"),
                "saga_priority_candidate": bool(tier_priority_verification.get("saga_priority_candidate") or False),
                "has_mixed_target_tiers": bool(tier_priority_verification.get("has_mixed_target_tiers") or False),
                "tier_event_count": int(tier_priority_verification.get("tier_event_count") or 0),
                "requires_manual_review": bool(tier_priority_verification.get("requires_manual_review") or False),
                "priority_verdict": str(tier_priority_verification.get("priority_verdict") or "not_applicable"),
                "reason": str(tier_priority_verification.get("reason") or ""),
            }
            replay_gate = deps.build_captured_session_replay_gate(
                runtime_mode=str((intent_execution_shadow or {}).get("runtime_mode") or "single_query_shared"),
                context_window_profile=str(
                    (intent_contract.get("routing_shadow_temporal") or {}).get("context_window_profile")
                    or context_window_profile
                    or "mid_context_300k"
                ),
                routing_shadow_execution=intent_contract.get("routing_shadow_execution") or {},
                routing_shadow_budget=intent_contract.get("routing_shadow_budget") or {},
                routing_shadow_temporal=intent_contract.get("routing_shadow_temporal") or {},
                client_meta=client_meta,
            )
            intent_contract["routing_shadow_replay_gate"] = replay_gate
            guarded_takeover = deps.build_guarded_default_takeover(
                runtime_mode=str((intent_execution_shadow or {}).get("runtime_mode") or "single_query_shared"),
                routing_mode=str(intent_contract.get("routing_mode") or "single_query_shared"),
                routing_shadow_execution=intent_contract.get("routing_shadow_execution") or {},
                routing_shadow_merge=intent_contract.get("routing_shadow_merge") or {},
                routing_shadow_budget=intent_contract.get("routing_shadow_budget") or {},
                routing_shadow_temporal=intent_contract.get("routing_shadow_temporal") or {},
                routing_shadow_replay_gate=replay_gate,
            )
            intent_contract["routing_shadow_takeover"] = guarded_takeover
            enforced_takeover = deps.build_enforced_default_takeover(
                runtime_mode=str((intent_execution_shadow or {}).get("runtime_mode") or "single_query_shared"),
                routing_shadow_takeover=guarded_takeover,
                routing_shadow_budget=shadow_budget_detail,
            )
            intent_contract["routing_shadow_enforced_takeover"] = {
                key: value
                for key, value in enforced_takeover.items()
                if key != "selected_candidates"
            }
            if isinstance(intent_execution_shadow, dict):
                intent_execution_shadow["replay_gate"] = replay_gate
                intent_execution_shadow["guarded_takeover"] = guarded_takeover
                intent_execution_shadow["tier_priority_verification"] = tier_priority_verification
                intent_execution_shadow["enforced_takeover"] = {
                    key: value
                    for key, value in enforced_takeover.items()
                    if key != "selected_candidates"
                }
        episode_items, chapter_items, arc_items, saga_items, ann_candidate_snapshot = deps.apply_ann_default_takeover(
            ann_candidate_snapshot=ann_candidate_snapshot,
            episode_items=episode_items,
            chapter_items=chapter_items,
            arc_items=arc_items,
            saga_items=saga_items,
            top_k=top_k,
            context_window_profile=context_window_profile,
        )

        if isinstance(intent_contract, dict):
            enforced_takeover_payload = (intent_contract.get("routing_shadow_enforced_takeover") or {})
            if bool(enforced_takeover_payload.get("takeover_applied")):
                selected_candidates = [
                    item
                    for item in ((shadow_budget_detail.get("selected_candidates") or []))
                    if isinstance(item, dict)
                ]
                takeover_payload = deps.build_takeover_recall_payload(
                    sid=sid,
                    selected_candidates=selected_candidates,
                )
                takeover_document_count = int(takeover_payload.get("document_count") or 0)
                if takeover_document_count > 0:
                    search_items = list(takeover_payload.get("search_items") or [])
                    episode_items = list(takeover_payload.get("episode_items") or [])
                    chapter_items = list(takeover_payload.get("chapter_items") or [])
                    arc_items = list(takeover_payload.get("arc_items") or [])
                    saga_items = list(takeover_payload.get("saga_items") or [])
                    try:
                        injection_text = deps.storyteller_format_for_injection(search_items)
                    except Exception:
                        injection_text = ""
                    memory_count = int(takeover_payload.get("memory_count") or 0)
                    fallback_count = int(takeover_payload.get("fallback_count") or 0)

                    packet_budget_policy = intent_contract.get("packet_budget_policy")
                    if isinstance(packet_budget_policy, dict):
                        packet_budget_policy["mode"] = "enforced"

                    enforced_takeover_payload["selected_document_ids"] = list(
                        takeover_payload.get("selected_document_ids") or []
                    )
                    enforced_takeover_payload["metrics"] = {
                        **(enforced_takeover_payload.get("metrics") or {}),
                        "applied_document_count": takeover_document_count,
                    }
                    intent_contract["routing_shadow_enforced_takeover"] = enforced_takeover_payload
                    if isinstance(intent_execution_shadow, dict):
                        intent_execution_shadow["enforced_takeover"] = enforced_takeover_payload
                else:
                    enforced_takeover_payload["status"] = "blocked"
                    enforced_takeover_payload["decision"] = "hold"
                    enforced_takeover_payload["reason"] = "takeover_payload_empty"
                    enforced_takeover_payload["takeover_applied"] = False
                    enforced_takeover_payload["effective_routing_mode"] = "single_query_shared"
                    intent_contract["routing_shadow_enforced_takeover"] = enforced_takeover_payload
                    if isinstance(intent_execution_shadow, dict):
                        intent_execution_shadow["enforced_takeover"] = enforced_takeover_payload

        retrieval_documents = deps.bundle_retrieval_documents(
            sid=sid,
            search_items=search_items,
            episode_items=episode_items,
            chapter_items=chapter_items,
            arc_items=arc_items,
            saga_items=saga_items,
        )
        index_snapshot = deps.update_retrieval_index(sid, retrieval_documents)
        validation_gates = deps.build_validation_gates(
            client_meta=client_meta,
            packet_budget_policy=(intent_contract or {}).get("packet_budget_policy") or {},
            retrieval_documents=retrieval_documents,
        )
        if isinstance(intent_contract, dict):
            deps.refresh_step21_validation_contract(
                intent_contract=intent_contract,
                client_meta=client_meta,
                validation_gates=validation_gates,
                routing_shadow_replay_gate=intent_contract.get("routing_shadow_replay_gate") or {},
            )

        return {
            "search": {
                "items": search_items,
                "injection_text": injection_text,
                "memory_count": memory_count,
                "fallback_count": fallback_count,
                "total_count": len(search_items),
            },
            "kg": {
                "items": kg_items,
                "count": len(kg_items),
                "entities_sent": len(entities),
            },
            "episode": {
                "items": episode_items,
                "count": len(episode_items),
            },
            "chapter": {
                "items": chapter_items,
                "count": len(chapter_items),
            },
            "arc": {
                "items": arc_items,
                "count": len(arc_items),
            },
            "saga": {
                "items": saga_items,
                "count": len(saga_items),
            },
            "intent_contract": intent_contract,
            "intent_execution_shadow": intent_execution_shadow,
            "document_schema": dict(deps.retrieval_document_schema),
            "documents": retrieval_documents,
            "document_count": len(retrieval_documents),
            "ann_candidate_snapshot": ann_candidate_snapshot,
            "index_snapshot": index_snapshot,
            "entities_extracted": entities,
            "validation_gates": validation_gates,
        }
    except Exception as exc:
        logger.warning("[prepare-turn] _bundle_recall outer failed: %s", exc)
        return None

