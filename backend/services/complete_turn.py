"""Complete-turn service: OOC guard, route-input helpers, and turn completion handler.

Extracted from backend/main.py as part of Step 21.5 complete_turn_m4 service extraction.
This module must NOT import backend.main to avoid circular dependencies.
All main.py dependencies are injected via keyword-only parameters.
"""

from __future__ import annotations

from datetime import datetime, timezone


def prepare_complete_turn_route_inputs(
    *,
    user_input: str,
    assistant_content: str,
    context_messages: list[dict],
    sanitize_context_messages_for_critic,
) -> dict[str, object]:
    """Trim, truncate, and sanitize complete-turn route inputs.

    Returns a dict with:
      - safe_user: str
      - safe_content: str
      - safe_ctx_for_ooc: list[dict]
      - has_persistable_turn_content: bool
    """
    safe_user = (user_input or "").strip()[:2000]
    safe_content = (assistant_content or "").strip()
    safe_ctx_for_ooc = sanitize_context_messages_for_critic(context_messages or [])
    has_persistable_turn_content = bool(safe_user or safe_content)
    return {
        "safe_user": safe_user,
        "safe_content": safe_content,
        "safe_ctx_for_ooc": safe_ctx_for_ooc,
        "has_persistable_turn_content": has_persistable_turn_content,
    }


def should_apply_complete_turn_ooc_guard(
    *,
    user_input: str,
    assistant_content: str = "",
    context_messages: list[dict],
    looks_like_ooc_text,
) -> bool:
    """Return True if the turn should be skipped due to OOC content."""
    if looks_like_ooc_text(user_input):
        return True

    if looks_like_ooc_text(assistant_content):
        return True

    if not isinstance(context_messages, list):
        return False

    for item in reversed(context_messages[-3:]):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role != "user":
            continue
        if looks_like_ooc_text(item.get("content") or ""):
            return True
    return False


def resolve_complete_turn_ooc_guard_cleanup(
    *,
    chat_session_id: str,
    turn_index: int,
    purge_turn_artifacts_for_ooc,
) -> dict:
    """Resolve OOC cleanup, returning a status dict."""
    sid = (chat_session_id or "").strip()
    if not sid:
        return {"status": "off", "reason": "session_missing"}
    return purge_turn_artifacts_for_ooc(sid, turn_index)


def build_complete_turn_ooc_guard_response(
    *,
    response_model_cls,
    chat_session_id: str,
    turn_index: int,
    generated_at: str,
    response_warnings: list[str],
    ooc_guard_cleanup: dict,
    critic_pipeline_version: str = "",
    critic_preview_pass_version: str = "",
    direct_evidence_retention_policy_version: str = "",
) -> object:
    """Build an M4CompleteTurnResponse (or compatible model) for an OOC-guarded turn.

    The returned object matches the existing OOC branch shape in
    ``complete_turn_m4`` so the plugin handoff stays stable.
    """
    updated_warnings = list(response_warnings or [])
    updated_warnings.append("ooc_turn_guard_applied")

    return response_model_cls(
        status="ok",
        chat_session_id=chat_session_id,
        turn_index=turn_index,
        generated_at=generated_at,
        save_ok=True,
        save_error="skipped_by_ooc_guard",
        critic_triggered=False,
        critic_result=None,
        episode_result={
            "checked": False,
            "triggered": False,
            "range": None,
            "reason": "ooc_turn_guard",
        },
        chapter_result={
            "checked": False,
            "triggered": False,
            "range": None,
            "reason": "ooc_turn_guard",
        },
        summary_fallback=None,
        maintenance_enqueued=False,
        fail_reasons=[],
        trace_handoff={
            "skeleton": False,
            "turn_index": turn_index,
            "generated_at": generated_at,
            "save_ok": True,
            "critic_triggered": False,
            "ooc_turn_guard_applied": True,
            "ooc_turn_guard_cleanup": ooc_guard_cleanup,
            "critic_pipeline_version": critic_pipeline_version,
            "critic_pipeline_split_enabled": True,
            "critic_pipeline_all_in_single_call": False,
            "critic_pipeline_extractor_stage": "critic._extract",
            "critic_pipeline_reducer_stage": "critic.run_deterministic_reducer",
            "critic_pipeline_compactor_stage": "summary_compactor_background",
            "critic_pipeline_compactor_owner": "complete_turn.background_tasks",
            "critic_preview_pass_version": critic_preview_pass_version,
            "critic_preview_pass_enabled": True,
            "critic_preview_pass_scope": "recent_raw_and_verified_direct_evidence",
            "critic_preview_compaction_mode": "hint_only",
            "direct_evidence_retention_policy_version": direct_evidence_retention_policy_version,
            "direct_evidence_retention_enabled": True,
            "direct_evidence_retention_mode": "importance_lineage_ttl",
            "episode_checked": False,
            "episode_triggered": False,
            "chapter_checked": False,
            "chapter_triggered": False,
            "summary_fallback_status": "off",
            "summary_fallback_layer_count": 0,
            "summary_fallback_available_layers": [],
            "summary_fallback_staleness_status": "off",
            "summary_fallback_stale_layer_count": 0,
            "summary_fallback_stale_layers": [],
            "summary_retry_enqueue_status": "off",
            "summary_retry_enqueue_reason": "not_applicable",
            "summary_retry_enqueue_layer_count": 0,
            "summary_retry_enqueue_layers": [],
            "summary_failure_trace_status": "off",
            "summary_failure_trace_warning_count": 0,
            "summary_failure_trace_warning_codes": [],
            "maintenance_enqueued": False,
            "maintenance_queue_status": "skipped",
            "maintenance_queue_depth": 0,
            "maintenance_refresh_enabled": False,
        },
        warnings=list(dict.fromkeys(updated_warnings)),
    )


def resolve_complete_turn_ooc_guard_surface(
    *,
    chat_session_id: str,
    turn_index: int,
    generated_at: str,
    response_warnings: list[str],
    resolve_complete_turn_ooc_guard_cleanup,
    build_complete_turn_ooc_guard_response,
) -> object:
    """Resolve the OOC guard surface: cleanup + response build.

    Returns the built response model instance.
    """
    ooc_guard_cleanup = resolve_complete_turn_ooc_guard_cleanup(
        chat_session_id=chat_session_id,
        turn_index=turn_index,
    )
    return build_complete_turn_ooc_guard_response(
        chat_session_id=chat_session_id,
        turn_index=turn_index,
        generated_at=generated_at,
        response_warnings=response_warnings,
        ooc_guard_cleanup=ooc_guard_cleanup,
    )


def handle_complete_turn_m4(
    *,
    req,
    background_tasks,
    logger,
    librarian,
    critic,
    M4CompleteTurnResponse,
    MaintenanceEnqueueRequest,
    MaintenanceEnqueueResponse,
    sanitize_context_messages_for_critic,
    sanitize_text_for_critic,
    should_apply_ooc_guard,
    purge_turn_artifacts_for_ooc,
    mark_retrieval_index_dirty,
    check_episode_interval,
    run_episode_bg,
    run_chapter_bg,
    preview_chapter_generation,
    enqueue_maintenance_job,
    default_retry_enqueue_status,
    default_summary_failure_trace,
    resolve_summary_staleness_thresholds,
    build_last_good_summary_fallback,
    build_summary_retry_enqueue_plan,
    build_summary_failure_trace,
    pipeline_version: str,
    preview_version: str,
    retention_policy_version: str,
) -> object:
    """M-4b: complete-turn save / critic / episode / chapter / maintenance handler.

    All external dependencies are injected to avoid importing backend.main.
    The route wrapper in main.py is responsible for binding these dependencies.
    """
    now = datetime.now(timezone.utc).isoformat()

    fail_reasons: list[str] = []
    save_ok = False
    save_error: str | None = None
    critic_triggered = False
    episode_result: dict | None = None
    chapter_result: dict | None = None
    summary_fallback: dict | None = None
    summary_retry_enqueue: dict = default_retry_enqueue_status()
    summary_failure_trace: dict = default_summary_failure_trace()
    response_warnings: list[str] = []
    maintenance_enqueued = False
    maintenance_queue_status = "skipped"
    maintenance_queue_depth = 0
    maintenance_refresh_plan: dict | None = None

    logger.info(
        "[complete-turn][M-4b] session=%s turn=%d type=%s",
        (req.chat_session_id or "?")[:12],
        req.turn_index,
        req.request_type,
    )

    # Step 1: turn save (synchronous)
    route_inputs = prepare_complete_turn_route_inputs(
        user_input=req.user_input,
        assistant_content=req.assistant_content,
        context_messages=req.context_messages,
        sanitize_context_messages_for_critic=sanitize_context_messages_for_critic,
    )
    safe_user = str(route_inputs.get("safe_user") or "")
    safe_content = str(route_inputs.get("safe_content") or "")
    safe_ctx_for_ooc = route_inputs.get("safe_ctx_for_ooc")
    if not isinstance(safe_ctx_for_ooc, list):
        safe_ctx_for_ooc = []
    has_persistable_turn_content = bool(route_inputs.get("has_persistable_turn_content"))

    if should_apply_ooc_guard(
        user_input=safe_user,
        assistant_content=safe_content,
        context_messages=safe_ctx_for_ooc,
    ):
        return resolve_complete_turn_ooc_guard_surface(
            chat_session_id=req.chat_session_id,
            turn_index=req.turn_index,
            generated_at=now,
            response_warnings=response_warnings,
            resolve_complete_turn_ooc_guard_cleanup=lambda chat_session_id, turn_index: resolve_complete_turn_ooc_guard_cleanup(
                chat_session_id=chat_session_id,
                turn_index=turn_index,
                purge_turn_artifacts_for_ooc=purge_turn_artifacts_for_ooc,
            ),
            build_complete_turn_ooc_guard_response=lambda **kwargs: build_complete_turn_ooc_guard_response(
                response_model_cls=M4CompleteTurnResponse,
                critic_pipeline_version=pipeline_version,
                critic_preview_pass_version=preview_version,
                direct_evidence_retention_policy_version=retention_policy_version,
                **kwargs,
            ),
        )

    if has_persistable_turn_content:
        try:
            save_result = librarian.save_turn(
                req.turn_index,
                safe_user,
                safe_content,
                chat_session_id=req.chat_session_id or None,
            )
            if not save_result.get("ok"):
                raise RuntimeError(str(save_result.get("error") or "save_turn_failed"))
            save_ok = True
            if req.chat_session_id:
                mark_retrieval_index_dirty(
                    req.chat_session_id,
                    reason="complete_turn_saved",
                    turn_index=req.turn_index,
                )
            logger.info(
                "[complete-turn][M-4b] save OK session=%s turn=%d",
                (req.chat_session_id or "?")[:12], req.turn_index,
            )
        except Exception as exc:
            logger.error("[complete-turn][M-4b] save failed turn=%d: %s", req.turn_index, exc)
            save_error = str(exc)
            fail_reasons.append("save")
    else:
        save_error = "empty user_input and assistant_content — skip save"
        fail_reasons.append("save")

    # Step 2: critic (background task)
    try:
        safe_ctx = safe_ctx_for_ooc
        critic_user = sanitize_text_for_critic(safe_user)
        critic_content = sanitize_text_for_critic(safe_content)
        turn_content = f"{{{{user}}}}: {critic_user[:500]}\n{{{{char}}}}: {critic_content[:500]}"
        it = req.improvement_trace or {}
        if (it.get("payloadReplaced")
                and isinstance(it.get("keepCandidates"), list)
                and it["keepCandidates"]):
            keep_hint = ", ".join(str(k) for k in it["keepCandidates"][:5])
            turn_content += f"\n[개선 힌트: 보존=\"{keep_hint}\"]"
        background_tasks.add_task(
            critic.run,
            req.turn_index,
            turn_content,
            safe_ctx,
            req.chat_session_id or None,
            req.output_language_override,
        )
        critic_triggered = True
    except Exception as exc:
        logger.error("[complete-turn][M-4b] critic trigger failed turn=%d: %s", req.turn_index, exc)
        fail_reasons.append("critic")

    # Step 3: episode interval check + background generation
    try:
        raw_interval = (req.client_meta or {}).get("episode_interval_turns", 10)
        interval = max(1, min(100, int(raw_interval or 10)))
        ep_info = check_episode_interval(req.turn_index, interval)
        episode_result = ep_info
        if ep_info["reason"] == "interval_boundary" and req.chat_session_id:
            fr, to = ep_info["range"]
            background_tasks.add_task(
                run_episode_bg,
                req.chat_session_id,
                fr,
                to,
            )
            episode_result = {**ep_info, "triggered": True, "reason": "triggered_bg"}
            logger.info(
                "[complete-turn][M-4b] episode bg enqueued session=%s range=%d-%d",
                (req.chat_session_id or "?")[:12], fr, to,
            )
    except Exception as exc:
        logger.warning("[complete-turn][M-4b] episode check failed turn=%d: %s", req.turn_index, exc)
        fail_reasons.append("episode_check")
        episode_result = {"checked": False, "triggered": False, "range": None, "reason": "exception"}

    # Step 4: chapter interval check + opt-in background generation
    try:
        chapter_auto_enabled = bool((req.client_meta or {}).get("chapter_auto_enabled"))
        if chapter_auto_enabled and req.chat_session_id:
            raw_chapter_interval = (req.client_meta or {}).get("chapter_interval_turns", 60)
            preview = preview_chapter_generation(
                chat_session_id=req.chat_session_id,
                turn_index=req.turn_index,
                interval=int(raw_chapter_interval or 60),
                force=False,
            )
            candidate = preview.get("candidate_range") or {}
            blocking_reasons = list(preview.get("blocking_reasons") or [])
            warnings = list(preview.get("warnings") or [])
            ready = bool(preview.get("ready"))
            turn_span_recommended = bool((preview.get("input_stats") or {}).get("turn_span_recommended"))
            candidate_range = None
            if candidate:
                candidate_range = [candidate.get("from_turn"), candidate.get("to_turn")]

            reason = (
                blocking_reasons[0]
                if blocking_reasons else (
                    "not_recommended_turn_span" if not turn_span_recommended else (
                        ((preview.get("interval_check") or {}).get("reason")) or "not_ready"
                    )
                )
            )
            chapter_result = {
                "checked": True,
                "triggered": False,
                "range": candidate_range,
                "reason": reason,
                "ready": ready,
                "blocking_reasons": blocking_reasons,
                "warnings": warnings,
                "episode_count": (preview.get("input_stats") or {}).get("episode_count", 0),
            }

            if ready and turn_span_recommended and candidate_range:
                ensure_episode_range = None
                if episode_result and episode_result.get("triggered") and isinstance(episode_result.get("range"), list):
                    ensure_episode_range = episode_result.get("range")

                background_tasks.add_task(
                    run_chapter_bg,
                    req.chat_session_id,
                    candidate_range[0],
                    candidate_range[1],
                    ensure_episode_range,
                )
                chapter_result = {
                    **chapter_result,
                    "triggered": True,
                    "reason": "triggered_bg",
                }
                logger.info(
                    "[complete-turn][P-2a] chapter bg enqueued session=%s range=%d-%d",
                    (req.chat_session_id or "?")[:12], candidate_range[0], candidate_range[1],
                )
    except Exception as exc:
        logger.warning("[complete-turn][P-2a] chapter check failed turn=%d: %s", req.turn_index, exc)
        fail_reasons.append("chapter_check")
        chapter_result = {"checked": False, "triggered": False, "range": None, "reason": "exception"}

    # Step 5: maintenance queue enqueue + R-3c summary refresh plan
    try:
        sid = (req.chat_session_id or "").strip()
        if sid and save_ok:
            client_meta = req.client_meta or {}
            refresh_enabled = bool(client_meta.get("long_session_refresh_enabled"))

            chapter_refresh_enabled = (
                refresh_enabled
                and bool(client_meta.get("chapter_auto_enabled"))
                and not bool(chapter_result and chapter_result.get("triggered"))
            )
            arc_refresh_enabled = refresh_enabled and bool(client_meta.get("arc_auto_enabled", True))
            saga_refresh_enabled = refresh_enabled and bool(client_meta.get("saga_auto_enabled", True))

            maintenance_refresh_plan = {
                "enabled": refresh_enabled,
                "version": "r3c.v1",
                "mode": "maintenance_enqueue",
                "layers": {
                    "chapter": {
                        "enabled": chapter_refresh_enabled,
                        "interval_turns": int(client_meta.get("chapter_interval_turns", 60) or 60),
                    },
                    "arc": {
                        "enabled": arc_refresh_enabled,
                        "interval_turns": int(client_meta.get("arc_interval_turns", 240) or 240),
                    },
                    "saga": {
                        "enabled": saga_refresh_enabled,
                        "interval_turns": int(client_meta.get("saga_interval_turns", 960) or 960),
                    },
                },
            }

            mq_resp = enqueue_maintenance_job(
                MaintenanceEnqueueRequest(
                    chat_session_id=sid,
                    turn_index=req.turn_index,
                    shadow_only=True,
                    assistant_response=safe_content,
                    recent_responses=[],
                    supervisor_result=None,
                    summary_refresh=maintenance_refresh_plan,
                )
            )

            if isinstance(mq_resp, MaintenanceEnqueueResponse):
                maintenance_queue_status = mq_resp.status
                maintenance_queue_depth = mq_resp.queue_depth
                maintenance_enqueued = mq_resp.status in {"enqueued", "queued_pending", "replaced_pending"}
            else:
                maintenance_queue_status = "skipped"
                maintenance_queue_depth = 0
                maintenance_enqueued = False

            logger.info(
                "[complete-turn][R-3c] maintenance enqueue session=%s turn=%d status=%s depth=%d refresh_enabled=%s",
                sid[:12],
                req.turn_index,
                maintenance_queue_status,
                maintenance_queue_depth,
                str((maintenance_refresh_plan or {}).get("enabled", False)).lower(),
            )
        elif sid and not save_ok:
            maintenance_queue_status = "skipped_save_failed"
    except Exception as exc:
        logger.warning("[complete-turn][R-3c] maintenance enqueue failed turn=%d: %s", req.turn_index, exc)
        fail_reasons.append("maintenance_enqueue")

    # Response assembly
    if not fail_reasons:
        status = "ok"
    elif save_ok or critic_triggered:
        status = "partial"
    else:
        status = "error"

    # U-1a: summary check failure -> last-good fallback snapshot (read-only)
    summary_failure_reasons = [
        reason for reason in fail_reasons
        if reason in {"episode_check", "chapter_check"}
    ]
    if summary_failure_reasons:
        summary_staleness_thresholds = resolve_summary_staleness_thresholds(req.client_meta)
        summary_fallback = build_last_good_summary_fallback(
            req.chat_session_id,
            fail_reasons=summary_failure_reasons,
            reference_turn=req.turn_index,
            staleness_thresholds=summary_staleness_thresholds,
        )

        # U-1c: stale layer exists -> force-regenerate refresh via maintenance queue
        sid = (req.chat_session_id or "").strip()
        staleness_snapshot = (summary_fallback or {}).get("staleness") or {}
        retry_plan = build_summary_retry_enqueue_plan(
            staleness_snapshot,
            turn_index=req.turn_index,
            thresholds=summary_staleness_thresholds,
        )
        requested_layers = [
            name
            for name in ("chapter", "arc", "saga")
            if ((retry_plan.get("layers") or {}).get(name) or {}).get("enabled")
        ]
        summary_retry_enqueue = {
            **summary_retry_enqueue,
            "requested_layers": requested_layers,
            "requested_layer_count": len(requested_layers),
        }

        if not sid:
            summary_retry_enqueue = {
                **summary_retry_enqueue,
                "status": "off",
                "reason": "session_missing",
            }
        elif not save_ok:
            summary_retry_enqueue = {
                **summary_retry_enqueue,
                "status": "skipped",
                "reason": "save_failed",
            }
        elif (staleness_snapshot or {}).get("status") != "stale_detected":
            summary_retry_enqueue = {
                **summary_retry_enqueue,
                "status": "skipped",
                "reason": "stale_not_detected",
            }
        elif not retry_plan.get("enabled"):
            summary_retry_enqueue = {
                **summary_retry_enqueue,
                "status": "skipped",
                "reason": "no_retryable_stale_layers",
            }
        else:
            try:
                retry_resp = enqueue_maintenance_job(
                    MaintenanceEnqueueRequest(
                        chat_session_id=sid,
                        turn_index=req.turn_index,
                        shadow_only=True,
                        assistant_response=safe_content,
                        recent_responses=[],
                        supervisor_result=None,
                        summary_refresh=retry_plan,
                    )
                )
                queue_status = str(getattr(retry_resp, "status", "skipped") or "skipped")
                queue_depth = int(getattr(retry_resp, "queue_depth", 0) or 0)
                summary_retry_enqueue = {
                    **summary_retry_enqueue,
                    "status": "enqueued" if queue_status in {"enqueued", "queued_pending", "replaced_pending"} else "skipped",
                    "reason": "stale_force_regenerate_enqueued",
                    "enqueued": queue_status in {"enqueued", "queued_pending", "replaced_pending"},
                    "queue_status": queue_status,
                    "queue_depth": queue_depth,
                }
            except Exception as retry_exc:
                logger.warning(
                    "[complete-turn][U-1c] retry enqueue failed turn=%d: %s",
                    req.turn_index,
                    retry_exc,
                )
                summary_retry_enqueue = {
                    **summary_retry_enqueue,
                    "status": "error",
                    "reason": "retry_enqueue_failed",
                    "queue_status": "error",
                    "error": str(retry_exc)[:160],
                }

        if isinstance(summary_fallback, dict):
            summary_fallback["retry_enqueue"] = summary_retry_enqueue
            summary_failure_trace = build_summary_failure_trace(
                fail_reasons=summary_failure_reasons,
                staleness_snapshot=staleness_snapshot,
                retry_enqueue=summary_retry_enqueue,
            )
            summary_fallback["failure_trace"] = summary_failure_trace
            response_warnings.extend(list(summary_failure_trace.get("warning_codes") or []))

    response_warnings = list(dict.fromkeys(response_warnings))

    return M4CompleteTurnResponse(
        status=status,
        chat_session_id=req.chat_session_id,
        turn_index=req.turn_index,
        generated_at=now,
        save_ok=save_ok,
        save_error=save_error,
        critic_triggered=critic_triggered,
        critic_result=None,
        episode_result=episode_result,
        chapter_result=chapter_result,
        summary_fallback=summary_fallback,
        maintenance_enqueued=maintenance_enqueued,
        fail_reasons=fail_reasons,
        trace_handoff={
            "skeleton": False,
            "turn_index": req.turn_index,
            "generated_at": now,
            "save_ok": save_ok,
            "critic_triggered": critic_triggered,
            "critic_pipeline_version": pipeline_version,
            "critic_pipeline_split_enabled": True,
            "critic_pipeline_all_in_single_call": False,
            "critic_pipeline_extractor_stage": "critic._extract",
            "critic_pipeline_reducer_stage": "critic.run_deterministic_reducer",
            "critic_pipeline_compactor_stage": "summary_compactor_background",
            "critic_pipeline_compactor_owner": "complete_turn.background_tasks",
            "critic_preview_pass_version": preview_version,
            "critic_preview_pass_enabled": True,
            "critic_preview_pass_scope": "recent_raw_and_verified_direct_evidence",
            "critic_preview_compaction_mode": "hint_only",
            "direct_evidence_retention_policy_version": retention_policy_version,
            "direct_evidence_retention_enabled": True,
            "direct_evidence_retention_mode": "importance_lineage_ttl",
            "episode_checked": (episode_result or {}).get("checked", False),
            "episode_triggered": (episode_result or {}).get("triggered", False),
            "chapter_checked": bool(chapter_result and chapter_result.get("checked")),
            "chapter_triggered": bool(chapter_result and chapter_result.get("triggered")),
            "summary_fallback_status": (summary_fallback or {}).get("status", "off"),
            "summary_fallback_layer_count": int((summary_fallback or {}).get("layer_count") or 0),
            "summary_fallback_available_layers": list((summary_fallback or {}).get("available_layers") or []),
            "summary_fallback_staleness_status": ((summary_fallback or {}).get("staleness") or {}).get("status", "off"),
            "summary_fallback_stale_layer_count": int((((summary_fallback or {}).get("staleness") or {}).get("stale_layer_count") or 0)),
            "summary_fallback_stale_layers": list((((summary_fallback or {}).get("staleness") or {}).get("stale_layers") or [])),
            "summary_retry_enqueue_status": (summary_retry_enqueue or {}).get("status", "off"),
            "summary_retry_enqueue_reason": (summary_retry_enqueue or {}).get("reason", "not_applicable"),
            "summary_retry_enqueue_layer_count": int((summary_retry_enqueue or {}).get("requested_layer_count") or 0),
            "summary_retry_enqueue_layers": list((summary_retry_enqueue or {}).get("requested_layers") or []),
            "summary_failure_trace_status": (summary_failure_trace or {}).get("status", "off"),
            "summary_failure_trace_warning_count": int((summary_failure_trace or {}).get("warning_count") or 0),
            "summary_failure_trace_warning_codes": list((summary_failure_trace or {}).get("warning_codes") or []),
            "maintenance_enqueued": maintenance_enqueued,
            "maintenance_queue_status": maintenance_queue_status,
            "maintenance_queue_depth": maintenance_queue_depth,
            "maintenance_refresh_enabled": bool((maintenance_refresh_plan or {}).get("enabled")),
        },
        warnings=response_warnings,
    )
