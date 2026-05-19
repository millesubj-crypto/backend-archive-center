"""backend/services/prepare_turn_trace.py

Prepare-turn trace preview builder.
Extracted from backend/main.py as part of Step 21.5 accelerated slice.

This module MUST NOT import backend.main.
"""

from typing import Callable


# ---------------------------------------------------------------------------
# Q-3d intent hit preview (pure helper, no DB)
# ---------------------------------------------------------------------------

def _build_intent_hit_preview_q3d(
    recall_data,
    *,
    session_state_data=None,
    narrative_control_data=None,
    continuity_pack_data=None,
    resume_pack_data=None,
) -> dict:
    contract = ((recall_data or {}).get("intent_contract") or {}) if recall_data else {}
    intents = contract.get("intents") or {}
    if not intents:
        return {
            "status": "unavailable",
            "mode": "none",
            "routing_mode": "none",
            "shared_execution_query_source": "unknown",
            "retrieval_hit_count": 0,
            "support_only_count": 0,
            "miss_count": 0,
            "intents": {},
        }

    search_payload = (recall_data or {}).get("search") or {}
    retrieval_counts = {
        "memory": int(search_payload.get("memory_count") or 0),
        "episode": int(((recall_data or {}).get("episode") or {}).get("count") or 0),
        "chapter": int(((recall_data or {}).get("chapter") or {}).get("count") or 0),
        "arc": int(((recall_data or {}).get("arc") or {}).get("count") or 0),
        "saga": int(((recall_data or {}).get("saga") or {}).get("count") or 0),
    }
    support_ready = {
        "session_state": bool(session_state_data),
        "narrative_control": bool(narrative_control_data),
        "continuity_pack": bool(continuity_pack_data),
        "resume_pack": bool(resume_pack_data and resume_pack_data.get("pack_status") == "ready"),
        "search": int(search_payload.get("total_count") or 0) > 0,
        "chapter": retrieval_counts["chapter"] > 0,
        "arc": retrieval_counts["arc"] > 0,
        "saga": retrieval_counts["saga"] > 0,
    }

    intent_preview: dict[str, dict] = {}
    retrieval_hit_count = 0
    support_only_count = 0
    miss_count = 0
    for intent_name in contract.get("intent_order") or list(intents.keys()):
        payload = intents.get(intent_name) or {}
        retrieval_tiers = [
            tier
            for tier in (payload.get("retrieval_tiers") or [])
            if retrieval_counts.get(str(tier), 0) > 0
        ]
        ready_surfaces = [
            surface
            for surface in (payload.get("support_surfaces") or [])
            if support_ready.get(str(surface), False)
        ]
        if retrieval_tiers:
            status = "retrieval_hit"
            retrieval_hit_count += 1
        elif ready_surfaces:
            status = "support_only"
            support_only_count += 1
        else:
            status = "miss"
            miss_count += 1

        intent_preview[intent_name] = {
            "status": status,
            "retrieval_hit": bool(retrieval_tiers),
            "matched_tiers": retrieval_tiers,
            "support_surfaces_ready": ready_surfaces,
            "execution_mode": payload.get("execution_mode") or "unknown",
        }

    return {
        "status": "ready",
        "mode": "shared_query_preview",
        "routing_mode": contract.get("routing_mode") or "single_query_shared",
        "shared_execution_query_source": contract.get("shared_execution_query_source") or "unknown",
        "intent_priority_order": list(contract.get("intent_order") or list(intents.keys())),
        "retrieval_hit_count": retrieval_hit_count,
        "support_only_count": support_only_count,
        "miss_count": miss_count,
        "intents": intent_preview,
    }


# ---------------------------------------------------------------------------
# Public: bundle trace preview
# ---------------------------------------------------------------------------

def bundle_trace_preview(
    reads_ok: int,
    source: str,
    session_state_data,
    narrative_control_data,
    recall_data,
    supervisor_pack_data,
    continuity_pack_data=None,
    injection_pack_data=None,
    resume_pack_data=None,
    progression_ledger_data=None,
    autonomy_plan_data=None,
    micro_beat_proposal_data=None,
    scene_step_proposal_data=None,
    combined_proposal_data=None,
    *,
    logger,
) -> dict | None:
    """M-2d / M-3a / P-2c / P-3e: prepare-turn trace preview shape builder.
    Pure read/compute with no DB access.
    """
    try:
        bundle_sources = []
        if session_state_data is not None:
            bundle_sources.append("session_state")
        if narrative_control_data is not None:
            bundle_sources.append("narrative_control")
        if recall_data is not None:
            bundle_sources.append("recall")
        if supervisor_pack_data is not None:
            bundle_sources.append("supervisor_pack")
        if injection_pack_data is not None:
            bundle_sources.append("injection_pack")
        if resume_pack_data is not None:
            bundle_sources.append("resume_pack")

        recall_hit = False
        chapter_count = 0
        arc_count = 0
        saga_count = 0
        index_ready = False
        index_doc_count = 0
        index_partition_count = 0
        index_status = "empty"
        index_dirty = False
        index_last_event = "none"
        index_runtime_mode = "shadow"
        index_shadow_write_enabled = True
        ann_candidate_ready = False
        ann_candidate_status = "empty"
        ann_candidate_count = 0
        ann_candidate_tier_count = 0
        ann_rerank_applied = False
        ann_filtered_out_count = 0
        ann_merge_applied = False
        ann_default_takeover_applied = False
        ann_default_takeover_reason = "not_evaluated"
        ann_benchmark_status = "empty"
        ann_benchmark_latency_improved = False
        ann_benchmark_overlap_ratio = 0.0
        ann_benchmark_diversity_guard_ok = True

        if recall_data:
            search_count = (recall_data.get("search") or {}).get("total_count", 0)
            kg_count = (recall_data.get("kg") or {}).get("count", 0)
            episode_count = (recall_data.get("episode") or {}).get("count", 0)
            chapter_count = (recall_data.get("chapter") or {}).get("count", 0)
            arc_count = (recall_data.get("arc") or {}).get("count", 0)
            saga_count = (recall_data.get("saga") or {}).get("count", 0)
            index_snapshot = recall_data.get("index_snapshot") or {}
            index_status = str(index_snapshot.get("status") or "empty")
            index_ready = index_status == "ready"
            index_doc_count = int(index_snapshot.get("document_count") or 0)
            index_partition_count = int(index_snapshot.get("partition_count") or 0)
            index_dirty = bool(index_snapshot.get("dirty"))
            index_last_event = str(index_snapshot.get("last_event") or "none")
            index_runtime_mode = str(index_snapshot.get("runtime_mode") or "shadow")
            index_shadow_write_enabled = bool(index_snapshot.get("shadow_write_enabled", True))
            ann_candidate_snapshot = recall_data.get("ann_candidate_snapshot") or {}
            ann_candidate_status = str(ann_candidate_snapshot.get("status") or "empty")
            ann_candidate_ready = ann_candidate_status == "ready"
            ann_candidate_count = int(ann_candidate_snapshot.get("candidate_count") or 0)
            ann_candidate_tier_count = sum(
                1
                for count in (ann_candidate_snapshot.get("tier_counts") or {}).values()
                if int(count or 0) > 0
            )
            ann_rerank_applied = bool(ann_candidate_snapshot.get("rerank_applied"))
            ann_filtered_out_count = int(ann_candidate_snapshot.get("filtered_out_count") or 0)
            ann_merge_applied = bool(ann_candidate_snapshot.get("merge_applied"))
            ann_default_takeover_applied = bool(ann_candidate_snapshot.get("default_takeover_applied"))
            ann_default_takeover_reason = str(ann_candidate_snapshot.get("default_takeover_reason") or "not_evaluated")
            ann_benchmark_status = str((ann_candidate_snapshot.get("benchmark") or {}).get("status") or "empty")
            ann_benchmark_latency_improved = bool((ann_candidate_snapshot.get("benchmark") or {}).get("latency_improved"))
            ann_benchmark_overlap_ratio = round(float((ann_candidate_snapshot.get("benchmark") or {}).get("overlap_ratio") or 0.0), 4)
            ann_benchmark_diversity_guard_ok = bool((ann_candidate_snapshot.get("benchmark") or {}).get("diversity_guard_ok", True))
            recall_hit = bool(search_count or kg_count or episode_count or chapter_count or arc_count or saga_count)

        # Momentum / guide / injection / resume / ledger defaults
        momentum_fetched = bool((supervisor_pack_data or {}).get("momentum_packet"))
        guide_suffix_ready = bool((supervisor_pack_data or {}).get("guide_suffix"))
        sv_pack_ok = bool(supervisor_pack_data)
        injection_pack_ok = bool(injection_pack_data)
        injection_text_ok = bool((injection_pack_data or {}).get("memory_text") or (injection_pack_data or {}).get("kg_text"))
        input_context_ok = bool((injection_pack_data or {}).get("input_context_text"))
        canon_injection_ok = bool((injection_pack_data or {}).get("canon_injection_ok"))
        canon_drift_status = str((injection_pack_data or {}).get("canon_drift_status") or "none")
        canon_drift_ratio = round(float((injection_pack_data or {}).get("canon_drift_ratio") or 0.0), 4)
        canon_priority_applied = bool((injection_pack_data or {}).get("canon_priority_applied"))
        canon_drift_dimension_count = int((injection_pack_data or {}).get("canon_drift_dimension_count") or 0)
        canon_drift_covered_count = int((injection_pack_data or {}).get("canon_drift_covered_count") or 0)
        context_window_profile = str((injection_pack_data or {}).get("context_window_profile") or "mid_context_300k")
        packet_composition_status = str((injection_pack_data or {}).get("packet_composition_status") or "none")
        packet_composition_active_profile = str((injection_pack_data or {}).get("packet_composition_active_profile") or "mid_context_300k")
        packet_composition_section_order = (injection_pack_data or {}).get("packet_composition_section_order") or []
        packet_composition_anchor_layer = str((injection_pack_data or {}).get("packet_composition_anchor_layer") or "none")
        saga_priority_applied = bool((injection_pack_data or {}).get("saga_priority_applied"))
        saga_injection_ok = bool((injection_pack_data or {}).get("saga_injection_ok"))
        saga_input_ok = bool((injection_pack_data or {}).get("saga_input_ok"))
        resume_pack_ready = bool((resume_pack_data or {}).get("pack_status") == "ready")
        resume_pack_layer_count = int((resume_pack_data or {}).get("layer_count") or 0)
        resume_pack_trigger = (resume_pack_data or {}).get("trigger") or "none"
        apply_mode_echo = (injection_pack_data or {}).get("apply_mode_reflected") or "shadow"
        apply_verdict_rule_echo = (injection_pack_data or {}).get("apply_verdict_rule") or "trace_only"
        ledger_ready = bool(progression_ledger_data)
        consumed_beats_count = len((progression_ledger_data or {}).get("consumed_beats") or [])
        pending_beats_count = len((progression_ledger_data or {}).get("pending_beats") or [])
        unresolved_tensions_count = len((progression_ledger_data or {}).get("unresolved_tensions") or [])
        consequences_count = len((progression_ledger_data or {}).get("consequences") or [])
        payoffs_count = len((progression_ledger_data or {}).get("payoffs") or [])
        scene_deltas_count = len((progression_ledger_data or {}).get("scene_deltas") or [])
        story_ledger_mode = str((progression_ledger_data or {}).get("ledger_mode") or "deterministic_no_llm")
        story_ledger_policy_version = str((progression_ledger_data or {}).get("ledger_policy_version") or "lw1h.v1")
        world_pressure = (progression_ledger_data or {}).get("world_pressure") or {}
        world_pressure_policy_version = str((progression_ledger_data or {}).get("world_pressure_policy_version") or "lw1d.v1")
        world_pressure_ready = bool(world_pressure)
        supporting_precedence_guard = (progression_ledger_data or {}).get("supporting_precedence_guard") or {}
        continuity_precedence_policy_version = str(
            (progression_ledger_data or {}).get("continuity_precedence_policy_version") or "lw1e.v1"
        )
        supporting_precedence_guard_ready = bool(supporting_precedence_guard)
        supporting_precedence_supporting_only = bool(supporting_precedence_guard.get("supporting_only", True))
        supporting_precedence_blocks_user_input_override = bool(
            supporting_precedence_guard.get("cannot_override_current_user_input", True)
        )
        supporting_precedence_blocks_direct_evidence_override = bool(
            supporting_precedence_guard.get("cannot_override_verified_direct_evidence", True)
        )
        compatibility_contract = (progression_ledger_data or {}).get("compatibility_contract") or {}
        compatibility_policy_version = str(
            (progression_ledger_data or {}).get("compatibility_policy_version") or "lw1f.v1"
        )
        compatibility_ready = bool(compatibility_contract)
        compatibility_targets_count = len((compatibility_contract.get("targets") or []))
        compatibility_consumer_safe = bool(compatibility_contract.get("consumer_safe", True))
        lifecycle_model = (progression_ledger_data or {}).get("lifecycle_model") or {}
        lifecycle_policy_version = str((progression_ledger_data or {}).get("lifecycle_policy_version") or "lw1g.v1")
        lifecycle_ready = bool(lifecycle_model)
        lifecycle_states_count = len((lifecycle_model.get("states") or []))
        lifecycle_decay_rules_count = len((lifecycle_model.get("decay_rules") or {}))
        lifecycle_entry_count = 0
        lifecycle_state_counts = {key: 0 for key in ("latent", "active", "escalating", "aftermath", "resolved", "dormant")}
        for entry_group_key in ("unresolved_tensions", "consequences", "payoffs", "scene_deltas"):
            for item in ((progression_ledger_data or {}).get(entry_group_key) or []):
                lifecycle_entry_count += 1
                state_key = str((item or {}).get("lifecycle_state") or "").strip().lower()
                if state_key in lifecycle_state_counts:
                    lifecycle_state_counts[state_key] += 1
        for bucket_key in ("factions", "regions", "offscreen_threads", "public_pressure"):
            for item in (world_pressure.get(bucket_key) or []):
                lifecycle_entry_count += 1
                state_key = str((item or {}).get("pressure_state") or "").strip().lower()
                if state_key in lifecycle_state_counts:
                    lifecycle_state_counts[state_key] += 1
        world_pressure_factions_count = len((world_pressure.get("factions") or []))
        world_pressure_regions_count = len((world_pressure.get("regions") or []))
        world_pressure_offscreen_threads_count = len((world_pressure.get("offscreen_threads") or []))
        world_pressure_public_pressure_count = len((world_pressure.get("public_pressure") or []))
        world_pressure_timeline_count = len((world_pressure.get("timeline") or []))
        payoff_completed_count = sum(
            1 for item in ((progression_ledger_data or {}).get("payoffs") or [])
            if str((item or {}).get("payoff_state") or "") == "completed"
        )
        payoff_invalid_count = sum(
            1 for item in ((progression_ledger_data or {}).get("payoffs") or [])
            if str((item or {}).get("payoff_state") or "") == "invalid"
        )
        payoff_pending_count = sum(
            1 for item in ((progression_ledger_data or {}).get("payoffs") or [])
            if str((item or {}).get("payoff_state") or "") == "pending"
        )
        do_not_resolve_guard = (progression_ledger_data or {}).get("do_not_resolve_guard") or {}
        do_not_resolve_policy_version = str(
            (progression_ledger_data or {}).get("do_not_resolve_policy_version") or "lw1h.v1"
        )
        do_not_resolve_guard_ready = bool(do_not_resolve_guard)
        do_not_resolve_unresolved_count = sum(
            1 for item in ((progression_ledger_data or {}).get("unresolved_tensions") or [])
            if bool((item or {}).get("do_not_resolve_yet"))
        )
        do_not_resolve_payoff_pending_count = sum(
            1 for item in ((progression_ledger_data or {}).get("payoffs") or [])
            if str((item or {}).get("payoff_state") or "") == "pending"
            and bool((item or {}).get("do_not_resolve_yet"))
        )
        do_not_resolve_protected_count = do_not_resolve_unresolved_count + do_not_resolve_payoff_pending_count
        autonomy_mode = (autonomy_plan_data or {}).get("mode", "pass")
        autonomy_gate_reason = (autonomy_plan_data or {}).get("gate_reason", "no_plan")
        autonomy_suggested_beat = (autonomy_plan_data or {}).get("suggested_beat")
        micro_beat_ready = bool(
            micro_beat_proposal_data
            and micro_beat_proposal_data.get("hint_text")
        )
        scene_step_ready = bool(
            scene_step_proposal_data
            and scene_step_proposal_data.get("gate_reason") == "ok"
            and scene_step_proposal_data.get("scope_beats")
        )
        combined_ready = bool(
            combined_proposal_data
            and combined_proposal_data.get("combined_hint_text")
        )
        combined_confidence = (combined_proposal_data or {}).get("proposal_confidence", "none")
        if combined_ready:
            proposal_source = "combined"
        elif scene_step_ready:
            proposal_source = "scene_step"
        elif micro_beat_ready:
            proposal_source = "micro_beat"
        else:
            proposal_source = "none"
        proposal_plan_signals = (combined_proposal_data or {}).get("plan_signals") or []
        proposal_max_step = (scene_step_proposal_data or {}).get("max_step", 2)
        proposal_forbidden_count = len(
            (scene_step_proposal_data or {}).get("excluded_beats") or []
        )
        intent_hit_preview = _build_intent_hit_preview_q3d(
            recall_data,
            session_state_data=session_state_data,
            narrative_control_data=narrative_control_data,
            continuity_pack_data=continuity_pack_data,
            resume_pack_data=resume_pack_data,
        )
        return {
            "reads_ok": reads_ok,
            "reads_total": 5,
            "source": source,
            "bundle_sources": bundle_sources,
            "recall_hit": recall_hit,
            "chapter_count": chapter_count,
            "arc_count": arc_count,
            "saga_count": saga_count,
            "index_ready": index_ready,
            "index_status": index_status,
            "index_doc_count": index_doc_count,
            "index_partition_count": index_partition_count,
            "index_dirty": index_dirty,
            "index_last_event": index_last_event,
            "index_runtime_mode": index_runtime_mode,
            "index_shadow_write_enabled": index_shadow_write_enabled,
            "ann_candidate_ready": ann_candidate_ready,
            "ann_candidate_status": ann_candidate_status,
            "ann_candidate_count": ann_candidate_count,
            "ann_candidate_tier_count": ann_candidate_tier_count,
            "ann_rerank_applied": ann_rerank_applied,
            "ann_filtered_out_count": ann_filtered_out_count,
            "ann_merge_applied": ann_merge_applied,
            "ann_default_takeover_applied": ann_default_takeover_applied,
            "ann_default_takeover_reason": ann_default_takeover_reason,
            "ann_benchmark_status": ann_benchmark_status,
            "ann_benchmark_latency_improved": ann_benchmark_latency_improved,
            "ann_benchmark_overlap_ratio": ann_benchmark_overlap_ratio,
            "ann_benchmark_diversity_guard_ok": ann_benchmark_diversity_guard_ok,
            "momentum_fetched": momentum_fetched,
            "guide_suffix_ready": guide_suffix_ready,
            "sv_pack_ok": sv_pack_ok,
            "injection_pack_ok": injection_pack_ok,
            "injection_text_ok": injection_text_ok,
            "input_context_ok": input_context_ok,
            "canon_injection_ok": canon_injection_ok,
            "canon_drift_status": canon_drift_status,
            "canon_drift_ratio": canon_drift_ratio,
            "canon_priority_applied": canon_priority_applied,
            "canon_drift_dimension_count": canon_drift_dimension_count,
            "canon_drift_covered_count": canon_drift_covered_count,
            "context_window_profile": context_window_profile,
            "packet_composition_status": packet_composition_status,
            "packet_composition_active_profile": packet_composition_active_profile,
            "packet_composition_section_order": packet_composition_section_order,
            "packet_composition_anchor_layer": packet_composition_anchor_layer,
            "saga_priority_applied": saga_priority_applied,
            "saga_injection_ok": saga_injection_ok,
            "saga_input_ok": saga_input_ok,
            "resume_pack_ready": resume_pack_ready,
            "resume_pack_layer_count": resume_pack_layer_count,
            "resume_pack_trigger": resume_pack_trigger,
            "apply_mode_echo": apply_mode_echo,
            "apply_verdict_rule_echo": apply_verdict_rule_echo,
            "ledger_ready": ledger_ready,
            "consumed_beats_count": consumed_beats_count,
            "pending_beats_count": pending_beats_count,
            "unresolved_tensions_count": unresolved_tensions_count,
            "consequences_count": consequences_count,
            "payoffs_count": payoffs_count,
            "scene_deltas_count": scene_deltas_count,
            "story_ledger_mode": story_ledger_mode,
            "story_ledger_policy_version": story_ledger_policy_version,
            "world_pressure_ready": world_pressure_ready,
            "world_pressure_policy_version": world_pressure_policy_version,
            "continuity_precedence_policy_version": continuity_precedence_policy_version,
            "supporting_precedence_guard_ready": supporting_precedence_guard_ready,
            "supporting_precedence_supporting_only": supporting_precedence_supporting_only,
            "supporting_precedence_blocks_user_input_override": supporting_precedence_blocks_user_input_override,
            "supporting_precedence_blocks_direct_evidence_override": supporting_precedence_blocks_direct_evidence_override,
            "compatibility_policy_version": compatibility_policy_version,
            "compatibility_ready": compatibility_ready,
            "compatibility_targets_count": compatibility_targets_count,
            "compatibility_consumer_safe": compatibility_consumer_safe,
            "lifecycle_policy_version": lifecycle_policy_version,
            "lifecycle_ready": lifecycle_ready,
            "lifecycle_states_count": lifecycle_states_count,
            "lifecycle_decay_rules_count": lifecycle_decay_rules_count,
            "lifecycle_entry_count": lifecycle_entry_count,
            "lifecycle_latent_count": lifecycle_state_counts["latent"],
            "lifecycle_active_count": lifecycle_state_counts["active"],
            "lifecycle_escalating_count": lifecycle_state_counts["escalating"],
            "lifecycle_aftermath_count": lifecycle_state_counts["aftermath"],
            "lifecycle_resolved_count": lifecycle_state_counts["resolved"],
            "lifecycle_dormant_count": lifecycle_state_counts["dormant"],
            "world_pressure_factions_count": world_pressure_factions_count,
            "world_pressure_regions_count": world_pressure_regions_count,
            "world_pressure_offscreen_threads_count": world_pressure_offscreen_threads_count,
            "world_pressure_public_pressure_count": world_pressure_public_pressure_count,
            "world_pressure_timeline_count": world_pressure_timeline_count,
            "payoff_completed_count": payoff_completed_count,
            "payoff_invalid_count": payoff_invalid_count,
            "payoff_pending_count": payoff_pending_count,
            "do_not_resolve_policy_version": do_not_resolve_policy_version,
            "do_not_resolve_guard_ready": do_not_resolve_guard_ready,
            "do_not_resolve_protected_count": do_not_resolve_protected_count,
            "do_not_resolve_unresolved_count": do_not_resolve_unresolved_count,
            "do_not_resolve_payoff_pending_count": do_not_resolve_payoff_pending_count,
            "autonomy_mode": autonomy_mode,
            "autonomy_gate_reason": autonomy_gate_reason,
            "autonomy_suggested_beat": autonomy_suggested_beat,
            "micro_beat_ready": micro_beat_ready,
            "scene_step_ready": scene_step_ready,
            "combined_ready": combined_ready,
            "combined_confidence": combined_confidence,
            "proposal_source": proposal_source,
            "proposal_plan_signals": proposal_plan_signals,
            "proposal_max_step": proposal_max_step,
            "proposal_forbidden_count": proposal_forbidden_count,
            "intent_hit_preview": intent_hit_preview,
        }
    except Exception as exc:
        logger.warning("[prepare-turn] bundle_trace_preview failed: %s", exc)
        return None
