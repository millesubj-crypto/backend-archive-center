from __future__ import annotations

import json

from backend.retrieval_contracts import RetrievalDocumentQ1A


def _safe_json_loads_q1a(raw: object):
    if isinstance(raw, (dict, list)):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _clean_retrieval_text_q1a(raw: object, *, max_len: int = 600) -> str:
    text = " ".join(str(raw or "").split()).strip()
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _to_float_q1a(value: object) -> float | None:
    try:
        return round(float(value), 4)
    except Exception:
        return None


def _build_retrieval_document_q1a(tier: str, item: dict, sid: str) -> dict | None:
    if not isinstance(item, dict):
        return None

    source_type = str(item.get("source_type") or item.get("source") or tier).strip().lower() or tier
    source_row_id = item.get("id") if isinstance(item.get("id"), int) else None
    turn_index = item.get("turn_index") if isinstance(item.get("turn_index"), int) else None
    from_turn = item.get("from_turn") if isinstance(item.get("from_turn"), int) else None
    to_turn = item.get("to_turn") if isinstance(item.get("to_turn"), int) else None
    title = None
    text = ""
    source_subtype = None
    source_table = None
    metadata: dict = {}

    if tier == "memory":
        raw_summary = item.get("summary_json") or item.get("summary") or item.get("content") or ""
        parsed = _safe_json_loads_q1a(raw_summary)
        if isinstance(parsed, dict):
            text = _clean_retrieval_text_q1a(
                parsed.get("turn_summary") or parsed.get("summary") or parsed.get("text") or raw_summary
            )
        else:
            text = _clean_retrieval_text_q1a(item.get("content") or raw_summary)
        if turn_index is not None:
            from_turn = turn_index
            to_turn = turn_index
        title = (
            f"Memory @ turn {turn_index}"
            if source_type == "memory"
            else f"Chat Log @ turn {turn_index}"
        )
        source_subtype = "semantic_search" if source_type == "memory" else "fallback_verbatim"
        source_table = "memories" if source_type == "memory" else "chat_logs"
        metadata = {
            "importance": item.get("importance"),
            "final_score": item.get("final_score"),
            "semantic_rank_score": _to_float_q1a(item.get("semantic_rank_score")),
            "keyword_overlap_score": _to_float_q1a(item.get("keyword_overlap_score")),
            "hybrid_baseline_score": _to_float_q1a(item.get("hybrid_baseline_score")),
            "soft_bias_score": _to_float_q1a(item.get("soft_bias_score")),
            "speaker_bias_score": _to_float_q1a(item.get("speaker_bias_score")),
            "location_bias_score": _to_float_q1a(item.get("location_bias_score")),
            "storyline_bias_score": _to_float_q1a(item.get("storyline_bias_score")),
            "keyword_overlap_terms": item.get("keyword_overlap_terms"),
            "speaker_bias_terms": item.get("speaker_bias_terms"),
            "location_bias_terms": item.get("location_bias_terms"),
            "storyline_bias_terms": item.get("storyline_bias_terms"),
            "hybrid_baseline_policy_version": item.get("hybrid_baseline_policy_version"),
            "soft_bias_policy_version": item.get("soft_bias_policy_version"),
            "tail_budget_policy_version": item.get("tail_budget_policy_version"),
            "tail_budget_original_rank": item.get("tail_budget_original_rank"),
            "tail_budget_promoted": item.get("tail_budget_promoted"),
            "tail_budget_reason": item.get("tail_budget_reason"),
            "tail_budget_score_gap": _to_float_q1a(item.get("tail_budget_score_gap")),
            "archive_wing": item.get("archive_wing") or item.get("library_wing"),
            "archive_room": item.get("archive_room") or item.get("library_room"),
            "role": item.get("role"),
        }
    elif tier == "episode":
        text = _clean_retrieval_text_q1a(item.get("summary_text"))
        title = f"Episode {from_turn}-{to_turn}" if from_turn is not None and to_turn is not None else "Episode Summary"
        source_subtype = "semantic_search"
        source_table = "episode_summaries"
        metadata = {
            "key_entities": item.get("key_entities"),
            "key_events": item.get("key_events"),
            "open_loops_json": item.get("open_loops_json"),
            "relationship_changes_json": item.get("relationship_changes_json"),
        }
    elif tier == "chapter":
        text = _clean_retrieval_text_q1a(item.get("resume_text") or item.get("summary_text"))
        chapter_index = item.get("chapter_index")
        title = item.get("chapter_title") or (
            f"Chapter {chapter_index}" if chapter_index is not None else "Chapter Summary"
        )
        source_subtype = "legacy_keyword_fallback" if source_type == "chat_log_fallback" else "semantic_search"
        source_table = "chat_logs" if source_type == "chat_log_fallback" else "chapter_summaries"
        metadata = {
            "chapter_index": chapter_index,
            "open_loops_json": item.get("open_loops_json"),
            "relationship_changes_json": item.get("relationship_changes_json"),
            "world_changes_json": item.get("world_changes_json"),
            "callback_candidates_json": item.get("callback_candidates_json"),
        }
    elif tier == "arc":
        text = _clean_retrieval_text_q1a(item.get("arc_resume_text") or item.get("core_conflict"))
        arc_index = item.get("arc_index")
        title = item.get("arc_name") or (f"Arc {arc_index}" if arc_index is not None else "Arc Summary")
        source_subtype = "semantic_search"
        source_table = "arc_summaries"
        metadata = {
            "arc_index": arc_index,
            "arc_status": item.get("arc_status"),
            "active_promises_json": item.get("active_promises_json"),
            "unresolved_debts_json": item.get("unresolved_debts_json"),
            "resolved_payoffs_json": item.get("resolved_payoffs_json"),
            "callback_candidates_json": item.get("callback_candidates_json"),
            "irreversible_turns_json": item.get("irreversible_turns_json"),
            "callback_debts_json": item.get("callback_debts_json"),
            "relationship_pivots_json": item.get("relationship_pivots_json"),
            "future_payoff_candidates_json": item.get("future_payoff_candidates_json"),
        }
    elif tier == "saga":
        text = _clean_retrieval_text_q1a(item.get("resume_pack_text") or item.get("saga_summary"), max_len=900)
        title = item.get("era_label") or (
            f"Saga {from_turn}-{to_turn}" if from_turn is not None and to_turn is not None else "Saga Digest"
        )
        source_subtype = "semantic_search"
        source_table = "saga_digests"
        metadata = {
            "persistent_facts_json": item.get("persistent_facts_json"),
            "never_drop_candidates_json": item.get("never_drop_candidates_json"),
        }
    else:
        return None

    ann_metadata = {
        "base_similarity": _to_float_q1a(item.get("ann_base_similarity") or item.get("similarity")),
        "rerank_score": _to_float_q1a(item.get("ann_rerank_score")),
        "recency_score": _to_float_q1a(item.get("ann_recency_score")),
        "importance_score": _to_float_q1a(item.get("ann_importance_score")),
        "pinned_hint": item.get("ann_pinned_hint") if isinstance(item.get("ann_pinned_hint"), bool) else None,
        "pinned_score": _to_float_q1a(item.get("ann_pinned_score")),
        "status": item.get("ann_status"),
        "status_score": _to_float_q1a(item.get("ann_status_score")),
        "bucket_distance": item.get("ann_bucket_distance") if isinstance(item.get("ann_bucket_distance"), int) else None,
        "candidate_rank": item.get("ann_candidate_rank") if isinstance(item.get("ann_candidate_rank"), int) else None,
        "merge_rank": item.get("ann_merge_rank") if isinstance(item.get("ann_merge_rank"), int) else None,
        "merge_pass": item.get("ann_merge_pass") if isinstance(item.get("ann_merge_pass"), str) else None,
    }
    ann_metadata = {key: value for key, value in ann_metadata.items() if value is not None}
    if ann_metadata:
        metadata["ann"] = ann_metadata

    chroma_live_metadata = {
        "contract_version": item.get("chroma_live_contract_version"),
        "state": item.get("chroma_live_state"),
        "mode": item.get("chroma_live_mode"),
        "candidate_limit": item.get("chroma_live_candidate_limit") if isinstance(item.get("chroma_live_candidate_limit"), int) else None,
        "candidate_count": item.get("chroma_live_candidate_count") if isinstance(item.get("chroma_live_candidate_count"), int) else None,
        "hydrated_count": item.get("chroma_live_hydrated_count") if isinstance(item.get("chroma_live_hydrated_count"), int) else None,
        "hydrated": item.get("chroma_live_hydrated") if isinstance(item.get("chroma_live_hydrated"), bool) else None,
        "truth_authority": item.get("chroma_live_truth_authority"),
        "supporting_only": item.get("chroma_live_supporting_only") if isinstance(item.get("chroma_live_supporting_only"), bool) else None,
        "support_surface_mode": item.get("chroma_live_support_surface_mode"),
        "cannot_override_current_user_input": item.get("chroma_live_cannot_override_current_user_input") if isinstance(item.get("chroma_live_cannot_override_current_user_input"), bool) else None,
        "cannot_override_verified_direct_evidence": item.get("chroma_live_cannot_override_verified_direct_evidence") if isinstance(item.get("chroma_live_cannot_override_verified_direct_evidence"), bool) else None,
        "precedence_ceiling": item.get("chroma_live_precedence_ceiling"),
        "hydration_source_table": item.get("chroma_live_hydration_source_table"),
        "hydration_source_row_id": item.get("chroma_live_hydration_source_row_id") if isinstance(item.get("chroma_live_hydration_source_row_id"), int) else None,
        "merge_candidate_limit": item.get("chroma_live_merge_candidate_limit") if isinstance(item.get("chroma_live_merge_candidate_limit"), int) else None,
        "merged": item.get("chroma_live_merged") if isinstance(item.get("chroma_live_merged"), bool) else None,
        "merge_rank": item.get("chroma_live_merge_rank") if isinstance(item.get("chroma_live_merge_rank"), int) else None,
        "merge_reason": item.get("chroma_live_merge_reason"),
        "merge_score_gap": _to_float_q1a(item.get("chroma_live_merge_score_gap")),
        "merge_policy_version": item.get("chroma_live_merge_policy_version"),
        "candidate_rank": item.get("chroma_live_candidate_rank") if isinstance(item.get("chroma_live_candidate_rank"), int) else None,
        "distance": _to_float_q1a(item.get("chroma_live_distance")),
        "reason": item.get("chroma_live_reason"),
    }
    chroma_live_metadata = {key: value for key, value in chroma_live_metadata.items() if value is not None}
    if chroma_live_metadata:
        metadata["chroma_live"] = chroma_live_metadata

    metadata = {key: value for key, value in metadata.items() if value is not None}
    source_key = source_row_id if source_row_id is not None else (turn_index if turn_index is not None else f"{from_turn}-{to_turn}")
    doc = RetrievalDocumentQ1A(
        document_id=f"{tier}:{source_key}",
        tier=tier,
        source_type=source_type,
        source_subtype=source_subtype,
        source_row_id=source_row_id,
        source_table=source_table,
        chat_session_id=str(item.get("chat_session_id") or sid),
        from_turn=from_turn,
        to_turn=to_turn,
        turn_index=turn_index,
        title=str(title) if title is not None else None,
        text=text,
        similarity=_to_float_q1a(item.get("similarity")),
        created_at=item.get("created_at"),
        query_matched=True,
        metadata=metadata,
    )
    return doc.model_dump()
