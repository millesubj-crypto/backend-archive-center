"""Prepare-turn service: route body extraction from backend/main.py.

Extracted as part of Step 21.5 accelerated core slice.
This module must NOT import backend.main to avoid circular dependencies.
All main.py dependencies are injected via keyword-only parameters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter


def handle_prepare_turn(
    *,
    req,
    logger,
    Utf8JSONResponse,
    PrepareTurnResponse,
    bundle_read_session_state,
    bundle_read_narrative_control,
    bundle_read_continuity_pack,
    build_progression_ledger,
    build_autonomy_plan,
    build_micro_beat_proposal,
    build_scene_step_proposal,
    bundle_resume_pack,
    resolve_context_window_profile_details,
    bundle_recall,
    bundle_supervisor_input_pack,
    build_combined_planner,
    build_canonical_ledger,
    bundle_injection_assembly,
    build_truth_boundary_replay_gate,
    refresh_step21_validation_contract,
    bundle_writeback_preview,
    bundle_trace_preview,
    build_long_session_health,
    write_health_snapshot,
    build_generation_packet,
):
    """M-2b/P-2c: prepare-turn 세션-state / narrative-control / continuity-pack + recall 번들.

    source="partial": 1개 이상 read 성공
    source="skeleton": 전체 실패
    fallback_reason="": 전체 성공
    fallback_reason="partial_reads": 일부만 성공
    fallback_reason="skeleton_only": 전체 실패 (M-1b 명시)
    """
    sid = (req.chat_session_id or "").strip()
    if not sid:
        return Utf8JSONResponse(
            content={
                "status": "error",
                "source": "off",
                "fallback_reason": "session_missing",
                "chat_session_id": "",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "warnings": ["chat_session_id is required"],
            },
            status_code=400,
        )

    generated_at = datetime.now(timezone.utc).isoformat()

    # M-2a: bundle reads - session-state / narrative-control / continuity-pack
    session_state_data = bundle_read_session_state(sid)
    narrative_control_data = bundle_read_narrative_control(sid)
    continuity_pack_data = bundle_read_continuity_pack(sid)

    progression_ledger_data = None
    if isinstance(narrative_control_data, dict):
        progression_ledger_data = narrative_control_data.get("progression_ledger")
        if progression_ledger_data is None:
            progression_ledger_data = build_progression_ledger(
                story_plan_data=narrative_control_data.get("story_plan"),
                director_data=narrative_control_data.get("director"),
                last_turn=narrative_control_data.get("last_advanced_turn"),
                state_status=narrative_control_data.get("state_status"),
            )
    if progression_ledger_data is None:
        progression_ledger_data = build_progression_ledger(
            story_plan_data=None,
            director_data=None,
            last_turn=None,
            state_status="skeleton",
        )

    # N-2b: autonomy plan - progression ledger 기반 planner 결합
    autonomy_plan_data = build_autonomy_plan(progression_ledger_data)

    # N-3a: micro-beat proposal - autonomy mode=assist 기반 hint 결합
    micro_beat_proposal_data = build_micro_beat_proposal(
        autonomy_plan=autonomy_plan_data,
        narrative_control=narrative_control_data,
    )

    # N-3b: bounded scene-step proposal - micro_beat ready일 때만 scope 결합
    scene_step_proposal_data = build_scene_step_proposal(
        autonomy_plan=autonomy_plan_data,
        micro_beat_proposal=micro_beat_proposal_data,
        narrative_control=narrative_control_data,
    )

    resume_pack_data = bundle_resume_pack(
        sid=sid,
        continuity_trigger_mode=req.continuity_trigger_mode or "none",
    )
    context_window_profile, context_window_profile_resolution_source = resolve_context_window_profile_details(
        settings_obj=req.settings,
        client_meta=req.client_meta,
    )
    max_injection_chars = max(300, int((getattr(req.settings, "max_injection_chars", None) or 3000)))
    max_input_context_chars = max(120, int((getattr(req.settings, "max_input_context_chars", None) or 800)))

    # M-2b/P-2c: recall bundle - semantic search + KG + episode + chapter(read-only)
    recall_started_at = perf_counter()
    recall_data = bundle_recall(
        sid=sid,
        user_input=req.raw_user_input or "",
        continuity_query=req.continuity_query or "",
        top_k=req.settings.top_k if req.settings else 5,
        continuity_trigger_mode=req.continuity_trigger_mode or "none",
        context_window_profile=context_window_profile,
        context_window_profile_resolution_source=context_window_profile_resolution_source,
        max_injection_chars=max_injection_chars,
        max_input_context_chars=max_input_context_chars,
        session_state_data=session_state_data,
        narrative_control_data=narrative_control_data,
        continuity_pack_data=continuity_pack_data,
        resume_pack_data=resume_pack_data,
        client_meta=req.client_meta,
    )
    recall_latency_ms = round((perf_counter() - recall_started_at) * 1000.0, 3)

    # M-2c: supervisor input pack - persistent guidance + guide/initiative suffix + momentum
    supervisor_pack_data = bundle_supervisor_input_pack(
        sid=sid,
        settings_obj=req.settings,
        nc_data=narrative_control_data,
        continuity_trigger_mode=req.continuity_trigger_mode or "none",
        raw_user_input=req.raw_user_input or "",
    )

    # N-3c (N-3e): combined planner - story plan + director + maintenance signal 결합
    # supervisor_pack_data가 None일 때 maintenance_hint 반영 가능 (N-3e 완성 후)
    combined_proposal_data = build_combined_planner(
        micro_beat_proposal=micro_beat_proposal_data,
        scene_step_proposal=scene_step_proposal_data,
        narrative_control=narrative_control_data,
        supervisor_pack=supervisor_pack_data,
        raw_user_input=req.raw_user_input or "",
    )

    canonical_ledger_data = build_canonical_ledger(
        session_state_data=session_state_data,
        continuity_pack_data=continuity_pack_data,
        narrative_control_data=narrative_control_data,
    )
    # R-1d/M-3a/M-3c: canon-first packet assembly를 위해 canonical ledger를 최초로 계산한다.
    injection_pack_data = bundle_injection_assembly(
        recall_data,
        chat_session_id=sid,
        settings_obj=req.settings,
        raw_user_input=req.raw_user_input or "",
        session_state_data=session_state_data,
        continuity_pack_data=continuity_pack_data,
        canonical_ledger_data=canonical_ledger_data,
        resume_pack_data=resume_pack_data,
        context_window_profile=context_window_profile,
    )
    if isinstance(recall_data, dict):
        validation_gates = recall_data.get("validation_gates")
        if not isinstance(validation_gates, dict):
            validation_gates = {}
            recall_data["validation_gates"] = validation_gates
        validation_gates["truth_boundary_replay"] = build_truth_boundary_replay_gate(
            client_meta=req.client_meta,
            packet_composition=(injection_pack_data or {}).get("packet_composition") if injection_pack_data else None,
        )
        intent_contract = recall_data.get("intent_contract")
        if isinstance(intent_contract, dict):
            refresh_step21_validation_contract(
                intent_contract=intent_contract,
                client_meta=req.client_meta,
                validation_gates=validation_gates,
                routing_shadow_replay_gate=intent_contract.get("routing_shadow_replay_gate") or {},
            )

    all_reads = [session_state_data, narrative_control_data, continuity_pack_data, recall_data, supervisor_pack_data]
    reads_ok = sum(1 for x in all_reads if x is not None)
    source = "partial" if reads_ok > 0 else "skeleton"
    fallback_reason = "" if reads_ok == 5 else ("partial_reads" if reads_ok > 0 else "skeleton_only")
    warnings: list[str] = []
    if reads_ok < 5:
        warnings.append(f"prepare-turn M-3a: {reads_ok}/5 bundle reads succeeded")

    # M-2d: writeback_preview + trace_preview (H-4 non-debug groundwork)
    writeback_preview_data = bundle_writeback_preview(
        session_state_data=session_state_data,
        narrative_control_data=narrative_control_data,
        recall_data=recall_data,
        supervisor_pack_data=supervisor_pack_data,
        settings_obj=req.settings,
    )
    trace_preview_data = bundle_trace_preview(
        reads_ok=reads_ok,
        source=source,
        session_state_data=session_state_data,
        narrative_control_data=narrative_control_data,
        recall_data=recall_data,
        continuity_pack_data=continuity_pack_data,
        supervisor_pack_data=supervisor_pack_data,
        injection_pack_data=injection_pack_data,
        resume_pack_data=resume_pack_data,
        progression_ledger_data=progression_ledger_data,
        autonomy_plan_data=autonomy_plan_data,
        micro_beat_proposal_data=micro_beat_proposal_data,
        scene_step_proposal_data=scene_step_proposal_data,
        combined_proposal_data=combined_proposal_data,
    )
    long_session_health_data = build_long_session_health(
        recall_data=recall_data,
        continuity_pack_data=continuity_pack_data,
        trace_preview_data=trace_preview_data,
        recall_latency_ms=recall_latency_ms,
        context_window_profile=context_window_profile,
        resume_pack_data=resume_pack_data,
        bundle_read_source=source,
        bundle_fallback_reason=fallback_reason,
    )
    if isinstance(trace_preview_data, dict):
        health_summary = (long_session_health_data or {}).get("summary") or {}
        trace_preview_data["long_session_health_status"] = str(
            (long_session_health_data or {}).get("status") or "empty"
        )
        trace_preview_data["long_session_recall_latency_ms"] = round(
            float(health_summary.get("recall_latency_ms") or 0.0),
            3,
        )
        trace_preview_data["long_session_hit_quality_score"] = round(
            float(health_summary.get("recall_hit_quality_score") or 0.0),
            4,
        )
        trace_preview_data["long_session_hook_survival_ratio"] = round(
            float(health_summary.get("hook_survival_ratio") or 0.0),
            4,
        )
        trace_preview_data["long_session_callback_recovery_rate"] = round(
            float(health_summary.get("callback_recovery_rate") or 0.0),
            4,
        )
        trace_preview_data["long_session_resume_quality_score"] = round(
            float(health_summary.get("resume_quality_score") or 0.0),
            4,
        )
        trace_preview_data["long_session_resume_ready"] = bool(
            health_summary.get("resume_ready") or False
        )

    # R-3d: 계산된 long_session_health를 per-session 캐시에 기록한다 (fail-open)
    write_health_snapshot(sid, long_session_health_data, req.turn_index)

    logger.info(
        "[prepare-turn] M-3a: session=%s type=%s source=%s reads=%d/5 sv_pack=%s recall_hit=%s ip=%s",
        sid,
        req.request_type,
        source,
        reads_ok,
        "ok" if supervisor_pack_data else "skip",
        trace_preview_data.get("recall_hit", False) if trace_preview_data else "n/a",
        "ok" if injection_pack_data else "skip",
    )

    # O-1b: generation packet 결합 - 모든 데이터를 종합하여 최종 키값
    generation_packet_data = build_generation_packet(
        injection_pack=injection_pack_data,
        combined_proposal=combined_proposal_data,
        autonomy_plan=autonomy_plan_data,
        narrative_control=narrative_control_data,
        source=source,
        takeover_mode=(req.settings.takeover_mode if req.settings else "off"),
        raw_user_input=req.raw_user_input or "",
    )

    resp = PrepareTurnResponse(
        status="ok",
        source=source,
        chat_session_id=sid,
        generated_at=generated_at,
        session_state=session_state_data,
        narrative_control=narrative_control_data,
        progression_ledger=progression_ledger_data,
        autonomy_plan=autonomy_plan_data,
        micro_beat_proposal=micro_beat_proposal_data,
        scene_step_proposal=scene_step_proposal_data,
        combined_proposal=combined_proposal_data,
        generation_packet=generation_packet_data,
        continuity_pack=continuity_pack_data,
        resume_pack=resume_pack_data,
        canonical_ledger=canonical_ledger_data,
        recall_result=recall_data,
        supervisor_input_pack=supervisor_pack_data,
        injection_pack=injection_pack_data,
        packet_composition=(injection_pack_data or {}).get("packet_composition") if injection_pack_data else None,
        long_session_health=long_session_health_data,
        writeback_preview=writeback_preview_data,
        trace_preview=trace_preview_data,
        fallback_reason=fallback_reason,
        warnings=warnings,
    )
    return Utf8JSONResponse(content=resp.model_dump())
