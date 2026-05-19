from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import func

from backend.database import SessionLocal
from backend.direct_evidence import (
    is_within_retention_window,
    normalize_archive_state,
    normalize_capture_verification,
    normalize_committed_gate,
)
from backend.models import ChatLog, DirectEvidenceRecord
from backend.prepare_turn_assembly import (
    _format_chapter_text_m3a,
    _format_entity_anchor_text_m3a,
    _format_entity_digest_text_m3a,
    _format_episode_text_m3a,
    _format_fallback_text_m3a,
    _format_kg_text_m3a,
    _format_memory_text_m3a,
)
from backend.turn_contracts import PrepareTurnSettings


@dataclass(frozen=True, slots=True)
class BundleInjectionDeps:
    logger: logging.Logger
    clean_canonical_text: Callable[..., str]
    build_injection_budget_decisions: Callable[..., dict]
    build_profile_layer_rank_map: Callable[..., dict[str, int]]
    build_temporal_support_packet: Callable[..., dict]
    context_window_profiles: set[str]
    r2d_packet_composition_surface_version: str


def bundle_injection_assembly(
    recall_result: dict | None,
    chat_session_id: str | None = None,
    settings_obj: PrepareTurnSettings | None = None,
    raw_user_input: str = "",
    session_state_data: dict | None = None,
    continuity_pack_data: dict | None = None,
    canonical_ledger_data: dict | None = None,
    resume_pack_data: dict | None = None,
    context_window_profile: str = "mid_context_300k",
    *,
    deps: BundleInjectionDeps,
) -> dict | None:
    """M-3a / M-3b / M-3c: recall_result?먯꽌 injection???띿뒪??釉붾줉??議곕┰.

    諛섑솚 ?뺤떇:
    {
      # M-3a: 媛쒕퀎 釉붾줉 ?띿뒪??
            "latest_direct_evidence_text": str,  # EA-1g: 理쒖떊 gate-approved direct evidence anchor
            "recent_raw_turn_text": str,         # EA-1g: 理쒖떊 raw turn anchor (user/assistant)
      "memory_text":   str,   # plugin formatMemoryBlock ?숈씪 洹쒖튃
      "kg_text":       str,   # plugin formatKGBlock ?숈씪 洹쒖튃
      "episode_text":  str,   # plugin buildEpisodeBlock ?숈씪 洹쒖튃
            "chapter_text":  str,   # P-2d: real chapter summary fallback block (low-priority)
            "saga_text":     str,   # P-4d: ultra-long profile?먯꽌 ?곗꽑 二쇱엯?섎뒗 saga digest block
      "fallback_text": str,   # plugin formatFallbackBlock ?숈씪 洹쒖튃 (議곌굔遺)
            "entity_digest_text": str,  # ?몃Ъ/?μ냼/臾쇳뭹 媛꾨떒 ?붿빟 (異붽?)
      # M-3b: 議곕┰??recall 二쇱엯 ?띿뒪??+ ?명뭼 留λ씫
      "injection_text":      str,   # recall 釉붾줉留?議곕┰??pre-assembled ?띿뒪??
      "input_context_text":  str,   # ?명뭼 吏곸쟾 留λ씫 ?듭빱 (?곗냽????理쒓렐 memory ?붿빟)
      # M-3c: apply_mode ?쒕㎤??洹쒖튃 ?꾨뱶
      "apply_mode_reflected": str,  # ?꾩옱 apply_mode ?ㅼ젙???먯퐫 (off/shadow/reviewed_apply)
      "apply_verdict_rule":   str,  # plugin ?쒕㎤??洹쒖튃 (no_apply/trace_only/apply_if_approved)
    # T-1c: budget decision read-only surface
    "budget_decisions":     dict,
      # Plugin Main / Sub Review 怨꾩빟 placeholder (M-3d+?먯꽌 梨꾩썙吏?
      "effective_user_input": None,
      "apply_verdict":        None,
      "was_improved":         False,
      "improvement_source":   "none",
      "sources_used":         [...],
      "assembly_note":        str,
    }
    ?ㅽ뙣 ??None 諛섑솚.
    """
    import json as _json

    _VR18_SCOPED_VERBATIM_POLICY = {
        "version": "vr18a.v1",
        "max_items": 3,
        "max_total_chars": 720,
        "max_excerpt_chars": 160,
        "surface_label": "Scoped Verbatim Recall (support surface)",
        "source_kind": "direct_evidence",
        "support_surface_first": True,
        "prompt_injection_strategy": "latest_anchor_only",
        "surface_route": "scoped_verbatim_support",
        "surface_priority": ("latest_direct_evidence", "recent_raw_turn"),
    }

    logger = deps.logger
    _clean_canonical_text_r1a = deps.clean_canonical_text
    _build_injection_budget_decisions_t1c = deps.build_injection_budget_decisions
    _build_profile_layer_rank_map_r2c = deps.build_profile_layer_rank_map
    _build_temporal_support_packet_sc19 = deps.build_temporal_support_packet
    _CONTEXT_WINDOW_PROFILES_P4D = deps.context_window_profiles
    _R2D_PACKET_COMPOSITION_SURFACE_VERSION = deps.r2d_packet_composition_surface_version

    if not recall_result:
        return None
    try:
        sid = " ".join(str(chat_session_id or "").split()).strip()

        def _clean_priority_text_ea1g(value: object, *, max_len: int = 240) -> str:
            text = " ".join(str(value or "").split()).strip()
            if not text:
                return ""
            if len(text) > max_len:
                return text[: max_len - 3] + "..."
            return text

        def _collect_gate_approved_direct_evidence_rows_ea1g(*, session_id: str) -> tuple[int | None, list[dict]]:
            if not session_id:
                return None, []
            db = SessionLocal()
            try:
                latest_turn = (
                    db.query(func.max(ChatLog.turn_index))
                    .filter(ChatLog.chat_session_id == session_id)
                    .scalar()
                )
                if latest_turn is None:
                    return None, []

                rows = (
                    db.query(DirectEvidenceRecord)
                    .filter(DirectEvidenceRecord.chat_session_id == session_id)
                    .order_by(DirectEvidenceRecord.source_turn_end.desc(), DirectEvidenceRecord.id.desc())
                    .limit(24)
                    .all()
                )

                collected: list[dict] = []
                for row in rows:
                    archive_state = normalize_archive_state(row.archive_state)
                    verification = normalize_capture_verification(row.capture_verification)
                    gate = normalize_committed_gate(row.committed_gate)
                    if bool(row.tombstoned):
                        continue
                    if row.superseded_by_id is not None:
                        continue
                    if archive_state != "verified_direct":
                        continue
                    if verification != "verified":
                        continue
                    if bool(row.repair_needed):
                        continue
                    if gate not in {"finalize", "manual"}:
                        continue
                    if not is_within_retention_window(
                        source_turn_end=row.source_turn_end,
                        latest_turn_index=int(latest_turn),
                        archive_state=archive_state,
                        capture_verification=verification,
                        repair_needed=row.repair_needed,
                        committed_gate=gate,
                        tombstoned=row.tombstoned,
                        lineage=row.lineage_json,
                    ):
                        continue

                    cleaned_text = _clean_priority_text_ea1g(
                        row.evidence_text,
                        max_len=max(320, int(_VR18_SCOPED_VERBATIM_POLICY["max_excerpt_chars"])),
                    )
                    if not cleaned_text:
                        continue
                    anchor_turn = row.turn_anchor or row.source_turn_end or row.source_turn_start
                    start_turn = int(row.source_turn_start) if row.source_turn_start is not None else None
                    end_turn = int(row.source_turn_end) if row.source_turn_end is not None else None
                    anchor_turn_value = int(anchor_turn) if anchor_turn is not None else None
                    scope = "single_turn" if start_turn is not None and end_turn is not None and start_turn == end_turn else "turn_window"
                    turns = (
                        str(start_turn)
                        if start_turn is not None and end_turn is not None and start_turn == end_turn
                        else (
                            f"{start_turn}-{end_turn}"
                            if start_turn is not None and end_turn is not None
                            else (str(end_turn) if end_turn is not None else (str(start_turn) if start_turn is not None else "?"))
                        )
                    )
                    collected.append(
                        {
                            "anchor_turn": anchor_turn_value,
                            "source_turn_start": start_turn,
                            "source_turn_end": end_turn,
                            "scope": scope,
                            "turns": turns,
                            "evidence_kind": str(row.evidence_kind or "fact_event").strip() or "fact_event",
                            "cleaned_text": cleaned_text,
                        }
                    )

                return int(latest_turn), collected
            except Exception as exc_rows:
                logger.debug("[prepare-turn] EA-1g direct evidence row collect failed (non-fatal): %s", exc_rows)
                return None, []
            finally:
                db.close()

        def _build_recent_raw_turn_text_ea1g(*, session_id: str) -> str:
            if not session_id:
                return ""
            db = SessionLocal()
            try:
                latest_turn = (
                    db.query(func.max(ChatLog.turn_index))
                    .filter(ChatLog.chat_session_id == session_id)
                    .scalar()
                )
                if latest_turn is None:
                    return ""

                rows = (
                    db.query(ChatLog)
                    .filter(
                        ChatLog.chat_session_id == session_id,
                        ChatLog.turn_index == int(latest_turn),
                    )
                    .order_by(ChatLog.id.asc())
                    .all()
                )
                if not rows:
                    return ""

                user_text = ""
                assistant_text = ""
                for row in rows:
                    role = str(row.role or "").strip().lower()
                    content = _clean_priority_text_ea1g(row.content, max_len=180)
                    if not content:
                        continue
                    if role == "user" and not user_text:
                        user_text = content
                    elif role == "assistant" and not assistant_text:
                        assistant_text = content

                lines: list[str] = [f"[Turn {int(latest_turn)}]"]
                if user_text:
                    lines.append("U: " + user_text)
                if assistant_text:
                    lines.append("A: " + assistant_text)
                if len(lines) <= 1:
                    return ""
                return "\n".join(lines)
            except Exception as exc_raw:
                logger.debug("[prepare-turn] EA-1g recent_raw_turn anchor build failed (non-fatal): %s", exc_raw)
                return ""
            finally:
                db.close()

        def _build_latest_direct_evidence_text_ea1g(*, session_id: str) -> str:
            _latest_turn, evidence_rows = _collect_gate_approved_direct_evidence_rows_ea1g(session_id=session_id)
            if not evidence_rows:
                return ""
            latest = evidence_rows[0]
            evidence_text = _clean_priority_text_ea1g(latest.get("cleaned_text"), max_len=320)
            if not evidence_text:
                return ""
            anchor_turn = latest.get("anchor_turn")
            if anchor_turn is not None:
                return f"[Turn {int(anchor_turn)}] {evidence_text}"
            return evidence_text

        def _build_scoped_verbatim_support_vr18(*, session_id: str) -> dict:
            latest_turn, evidence_rows = _collect_gate_approved_direct_evidence_rows_ea1g(session_id=session_id)
            lines: list[str] = []
            items: list[dict] = []
            total_chars = 0
            for row in evidence_rows:
                if len(items) >= int(_VR18_SCOPED_VERBATIM_POLICY["max_items"]):
                    break
                excerpt = _clean_priority_text_ea1g(
                    row.get("cleaned_text"),
                    max_len=int(_VR18_SCOPED_VERBATIM_POLICY["max_excerpt_chars"]),
                )
                if not excerpt:
                    continue
                source_tag = (
                    f"[source={_VR18_SCOPED_VERBATIM_POLICY['source_kind']} "
                    f"scope={row.get('scope') or 'turn_window'} turns={row.get('turns') or '?'} "
                    f"anchor={row.get('anchor_turn') if row.get('anchor_turn') is not None else '?'} "
                    f"kind={row.get('evidence_kind') or 'fact_event'}]"
                )
                line = f"{source_tag} {excerpt}".strip()
                projected_total = total_chars + (1 if lines else 0) + len(line)
                if projected_total > int(_VR18_SCOPED_VERBATIM_POLICY["max_total_chars"]):
                    break
                lines.append(line)
                total_chars = projected_total
                items.append(
                    {
                        "source_tag": source_tag,
                        "excerpt": excerpt,
                        "scope": row.get("scope") or "turn_window",
                        "turns": row.get("turns") or "?",
                        "anchor_turn": row.get("anchor_turn"),
                        "evidence_kind": row.get("evidence_kind") or "fact_event",
                    }
                )

            return {
                "active": bool(items),
                "policy_version": _VR18_SCOPED_VERBATIM_POLICY["version"],
                "surface_label": _VR18_SCOPED_VERBATIM_POLICY["surface_label"],
                "support_surface_first": bool(_VR18_SCOPED_VERBATIM_POLICY["support_surface_first"]),
                "prompt_injection_strategy": _VR18_SCOPED_VERBATIM_POLICY["prompt_injection_strategy"],
                "surface_route": _VR18_SCOPED_VERBATIM_POLICY["surface_route"],
                "max_items": int(_VR18_SCOPED_VERBATIM_POLICY["max_items"]),
                "max_total_chars": int(_VR18_SCOPED_VERBATIM_POLICY["max_total_chars"]),
                "max_excerpt_chars": int(_VR18_SCOPED_VERBATIM_POLICY["max_excerpt_chars"]),
                "count": len(items),
                "latest_turn_index": latest_turn,
                "text": "\n".join(lines),
                "items": items,
                "surface_priority": list(_VR18_SCOPED_VERBATIM_POLICY["surface_priority"]),
            }

        def _format_canon_text_r1d(ledger: dict | None, profile_name: str) -> str:
            if not isinstance(ledger, dict):
                return ""
            if str(ledger.get("ledger_status") or "empty") == "empty":
                return ""

            max_chars_map = {
                "mid_context_300k": 420,
                "wide_context_500k": 520,
                "ultra_long_1m_plus": 680,
                "extreme_long_2m_plus": 820,
            }
            max_chars = int(max_chars_map.get(profile_name, 420))

            lines: list[str] = ["[Canonical Ledger]"]
            characters = ledger.get("characters") or []
            relationships = ledger.get("relationships") or []
            world_rules = ledger.get("world_rules") or []
            obligations = ledger.get("open_obligations") or []
            compact_history = ledger.get("resolved_compact_history") or []

            for item in characters[:2]:
                name = _clean_canonical_text_r1a(item.get("character_name"), max_len=36)
                status = _clean_canonical_text_r1a(item.get("status_summary"), max_len=96)
                if name and status:
                    lines.append(f"- Character: {name} ??{status}")
            for item in relationships[:2]:
                label = _clean_canonical_text_r1a(item.get("label"), max_len=36)
                summary = _clean_canonical_text_r1a(item.get("summary"), max_len=96)
                if label and summary:
                    lines.append(f"- Relationship: {label} ??{summary}")
            for item in world_rules[:2]:
                scope_name = _clean_canonical_text_r1a(item.get("scope_name") or item.get("scope"), max_len=36)
                key = _clean_canonical_text_r1a(item.get("key"), max_len=32)
                value = _clean_canonical_text_r1a(item.get("value_summary"), max_len=96)
                if key and value:
                    prefix = f"{scope_name} / " if scope_name else ""
                    lines.append(f"- World: {prefix}{key} ??{value}")
            for item in obligations[:3]:
                title = _clean_canonical_text_r1a(item.get("title"), max_len=48)
                summary = _clean_canonical_text_r1a(item.get("summary"), max_len=96)
                if title and summary:
                    lines.append(f"- Open: {title} ??{summary}")
            for item in compact_history[:2]:
                summary = _clean_canonical_text_r1a(item, max_len=110)
                if summary:
                    lines.append(f"- Resolved: {summary}")

            text = "\n".join(lines).strip()
            if text == "[Canonical Ledger]":
                return ""
            return _clean_canonical_text_r1a(text, max_len=max_chars)

        def _build_canon_drift_benchmark_r1e(
            ledger: dict | None,
            canon_payload: str,
            injection_payload: str,
            legacy_chars: int,
        ) -> dict:
            strategy = "canon_first_packet_proxy_v1"
            if not isinstance(ledger, dict) or str(ledger.get("ledger_status") or "empty") == "empty":
                return {
                    "status": "empty",
                    "reason": "no_canonical_ledger",
                    "strategy": strategy,
                    "canon_priority_applied": False,
                    "drift_dimension_count": 0,
                    "covered_dimension_count": 0,
                    "canon_drift_ratio": 0.0,
                    "populated_dimensions": [],
                    "covered_dimensions": [],
                    "canon_chars": 0,
                    "legacy_summary_chars": legacy_chars,
                }

            populated_dimensions: list[str] = []
            if ledger.get("characters"):
                populated_dimensions.append("character")
            if ledger.get("relationships"):
                populated_dimensions.append("relationship")
            if ledger.get("world_rules"):
                populated_dimensions.append("world_rule")
            if ledger.get("open_obligations"):
                populated_dimensions.append("open_obligation")

            if not canon_payload.strip():
                return {
                    "status": "degraded",
                    "reason": "canon_text_missing",
                    "strategy": strategy,
                    "canon_priority_applied": False,
                    "drift_dimension_count": len(populated_dimensions),
                    "covered_dimension_count": 0,
                    "canon_drift_ratio": 1.0 if populated_dimensions else 0.0,
                    "populated_dimensions": populated_dimensions,
                    "covered_dimensions": [],
                    "canon_chars": 0,
                    "legacy_summary_chars": legacy_chars,
                }

            covered_dimensions: list[str] = []
            if "- Character:" in canon_payload:
                covered_dimensions.append("character")
            if "- Relationship:" in canon_payload:
                covered_dimensions.append("relationship")
            if "- World:" in canon_payload:
                covered_dimensions.append("world_rule")
            if "- Open:" in canon_payload:
                covered_dimensions.append("open_obligation")

            canon_priority_applied = injection_payload.startswith("?곣봺 Canonical Ledger ?곣봺")
            covered_dimension_count = len([dim for dim in populated_dimensions if dim in covered_dimensions])
            drift_dimension_count = len(populated_dimensions)
            canon_drift_ratio = 0.0
            if drift_dimension_count:
                canon_drift_ratio = round(1.0 - (covered_dimension_count / drift_dimension_count), 4)

            return {
                "status": "ready" if canon_priority_applied else "degraded",
                "reason": "canon_first_packet" if canon_priority_applied else "canon_block_not_first",
                "strategy": strategy,
                "canon_priority_applied": canon_priority_applied,
                "drift_dimension_count": drift_dimension_count,
                "covered_dimension_count": covered_dimension_count,
                "canon_drift_ratio": canon_drift_ratio,
                "populated_dimensions": populated_dimensions,
                "covered_dimensions": covered_dimensions,
                "canon_chars": len(canon_payload),
                "legacy_summary_chars": legacy_chars,
            }

        memory_text = _format_memory_text_m3a(recall_result)
        kg_text = _format_kg_text_m3a(recall_result)
        episode_text = _format_episode_text_m3a(recall_result)
        search = recall_result.get("search") or {}
        items = search.get("items") or []
        scoped_verbatim_support = _build_scoped_verbatim_support_vr18(session_id=sid)
        latest_direct_evidence_text = _build_latest_direct_evidence_text_ea1g(session_id=sid)
        recent_raw_turn_text = _build_recent_raw_turn_text_ea1g(session_id=sid)

        def _detect_temporal_query_mode_ea1h(text: str) -> tuple[str, int | None]:
            import re as _re

            normalized = " ".join(str(text or "").split()).strip().lower()
            if not normalized:
                return "none", None

            specific_turn: int | None = None
            specific_match = _re.search(r"(?:turn|턴)\s*(\d+)|(\d+)\s*(?:turn|턴)", normalized)
            if specific_match:
                raw_num = specific_match.group(1) or specific_match.group(2)
                try:
                    if raw_num is not None:
                        specific_turn = int(raw_num)
                except Exception:
                    specific_turn = None

            earliest_tokens = ["처음", "최초", "첫", "first", "earliest", "beginning", "start"]
            middle_tokens = ["중간", "중반", "가운데", "middle", "mid", "midway"]
            latest_tokens = ["최근", "최신", "마지막", "요즘", "latest", "recent", "most recent", "current", "now"]

            if specific_turn is not None:
                return "specific_turn", specific_turn
            if any(tok in normalized for tok in earliest_tokens):
                return "earliest", None
            if any(tok in normalized for tok in middle_tokens):
                return "middle", None
            if any(tok in normalized for tok in latest_tokens):
                return "latest", None
            return "none", None

        def _extract_summary_preview_ea1h(raw_value: object, *, max_len: int = 220) -> str:
            if isinstance(raw_value, str):
                try:
                    parsed = _json.loads(raw_value)
                    text = parsed.get("turn_summary") or parsed.get("summary") or raw_value
                except Exception:
                    text = raw_value
            elif isinstance(raw_value, dict):
                text = raw_value.get("turn_summary") or raw_value.get("summary") or _json.dumps(raw_value, ensure_ascii=False)
            else:
                text = str(raw_value or "")
            text = " ".join(str(text).split()).strip()
            if not text:
                return ""
            if len(text) > max_len:
                return text[: max_len - 3] + "..."
            return text

        def _build_temporal_integrity_anchor_ea1h() -> tuple[str, str, str]:
            mode, requested_turn = _detect_temporal_query_mode_ea1h(raw_user_input)
            if mode == "none":
                return "", "", "none"

            candidates: list[dict] = []

            for it in items:
                if not isinstance(it, dict) or it.get("source") != "memory":
                    continue
                turn_index = it.get("turn_index")
                if turn_index is None:
                    continue
                try:
                    ti = int(turn_index)
                except Exception:
                    continue
                summary = _extract_summary_preview_ea1h(it.get("summary_json") or it.get("summary") or "")
                if not summary:
                    continue
                candidates.append({
                    "source": "memory",
                    "turn_center": ti,
                    "from_turn": ti,
                    "to_turn": ti,
                    "summary": summary,
                })

            for ep_item in ((recall_result.get("episode") or {}).get("items") or []):
                if not isinstance(ep_item, dict):
                    continue
                ft = ep_item.get("from_turn")
                tt = ep_item.get("to_turn")
                if ft is None or tt is None:
                    continue
                try:
                    fti = int(ft)
                    tti = int(tt)
                except Exception:
                    continue
                summary = _extract_summary_preview_ea1h(ep_item.get("summary_text") or "")
                if not summary:
                    continue
                candidates.append({
                    "source": "episode",
                    "turn_center": (fti + tti) // 2,
                    "from_turn": fti,
                    "to_turn": tti,
                    "summary": summary,
                })

            if not candidates:
                return "", "", mode

            candidates.sort(key=lambda row: (int(row.get("turn_center") or 0), str(row.get("source") or "")))

            selected = candidates[0]
            if mode == "latest":
                selected = candidates[-1]
            elif mode == "middle":
                selected = candidates[len(candidates) // 2]
            elif mode == "specific_turn" and requested_turn is not None:
                selected = min(candidates, key=lambda row: abs(int(row.get("turn_center") or 0) - int(requested_turn)))

            selected_from = int(selected.get("from_turn") or 0)
            selected_to = int(selected.get("to_turn") or 0)
            selected_center = int(selected.get("turn_center") or selected_from)
            selected_summary = str(selected.get("summary") or "")
            if not selected_summary:
                return "", "", mode

            evidence_min = int(candidates[0].get("turn_center") or 0)
            evidence_max = int(candidates[-1].get("turn_center") or 0)
            requested_line = f"- requested_turn: {int(requested_turn)}\n" if requested_turn is not None else ""
            temporal_text = (
                "Temporal query integrity rule is active. Resolve answer from selected timeline anchor below.\n"
                f"- mode: {mode}\n"
                f"{requested_line}"
                f"- selected_source: {selected.get('source')}\n"
                f"- selected_turn_window: {selected_from}-{selected_to}\n"
                f"- selected_summary: {selected_summary}\n"
                f"- evidence_turn_center_range: {evidence_min}-{evidence_max} ({len(candidates)} candidates)"
            )

            preview = selected_summary
            if len(preview) > 120:
                preview = preview[:117] + "..."
            temporal_anchor = f"[Temporal Anchor:{mode}] turn {selected_center}: {preview}"
            return temporal_text, temporal_anchor, mode

        temporal_integrity_text, temporal_input_anchor_text, temporal_mode = _build_temporal_integrity_anchor_ea1h()
        chapter_text = ""
        if not episode_text:
            chapter_text = _format_chapter_text_m3a(recall_result)
        saga_text = ""
        profile = context_window_profile if context_window_profile in _CONTEXT_WINDOW_PROFILES_P4D else "mid_context_300k"
        saga_priority_applied = False
        canon_text = _format_canon_text_r1d(canonical_ledger_data, profile)

        if profile in {"ultra_long_1m_plus", "extreme_long_2m_plus"}:
            saga_payload = ((resume_pack_data or {}).get("saga") or {}) if resume_pack_data else {}
            raw_saga_text = " ".join(str(saga_payload.get("resume_pack_text") or saga_payload.get("saga_summary") or "").split()).strip()
            if raw_saga_text:
                if len(raw_saga_text) > 900:
                    raw_saga_text = raw_saga_text[:897] + "..."
                saga_label = " ".join(str(saga_payload.get("era_label") or "").split()).strip()
                saga_header = saga_label or f"Saga {saga_payload.get('from_turn')}-{saga_payload.get('to_turn')}"
                saga_text = f"[{saga_header}] {raw_saga_text}"
                saga_priority_applied = True
                if profile == "extreme_long_2m_plus":
                    chapter_text = ""

        # fallback: memory ??ぉ??2媛?誘몃쭔???뚮쭔 ?ы븿 (plugin怨??숈씪 議곌굔)
        mem_count = sum(1 for it in items if isinstance(it, dict) and it.get("source") == "memory")
        has_fallback = any(isinstance(it, dict) and it.get("source") == "chat_log" for it in items)
        fallback_text = _format_fallback_text_m3a(recall_result) if (has_fallback and mem_count < 2) else ""

        # ?몃Ъ/?μ냼/臾쇳뭹 ?붿빟: ??λ맂 援ъ“???곗씠??+ KG瑜?湲곕컲?쇰줈 媛꾨떒 ?붿빟??留뚮뱺??
        entity_digest_text = _format_entity_digest_text_m3a(
            recall_result=recall_result,
            session_state_data=session_state_data,
            continuity_pack_data=continuity_pack_data,
        )
        if entity_digest_text:
            if memory_text:
                memory_text = memory_text + "\n[Entity Digest]\n" + entity_digest_text
            else:
                memory_text = "[Entity Digest]\n" + entity_digest_text

        layer_rank_map = _build_profile_layer_rank_map_r2c(profile)
        unknown_layer_rank = len(layer_rank_map) + 99

        # R-2c: R-2b?먯꽌 ?뺤쓽??profile layer priority瑜??ㅼ젣 packet section 議곕┰ ?쒖꽌??諛섏쁺?쒕떎.
        section_specs = [
            {
                "source": "temporal_integrity",
                "source_alias": "temporal_integrity",
                "layer": "direct_evidence",
                "header": "?곣봺 Temporal Integrity Anchor ?곣봺",
                "text": temporal_integrity_text,
                "stable_order": -3,
            },
            {
                "source": "latest_direct_evidence",
                "source_alias": "latest_direct_evidence",
                "layer": "direct_evidence",
                "header": "?곣봺 Latest Direct Evidence ?곣봺",
                "text": latest_direct_evidence_text,
                "stable_order": -2,
            },
            {
                "source": "recent_raw_turn",
                "source_alias": "recent_raw_turn",
                "layer": "recent_raw_turn",
                "header": "?곣봺 Recent Raw Turn ?곣봺",
                "text": recent_raw_turn_text,
                "stable_order": -1,
            },
            {
                "source": "canon",
                "source_alias": "canon",
                "layer": "canonical_ledger",
                "header": "?곣봺 Canonical Ledger ?곣봺",
                "text": canon_text,
                "stable_order": 0,
            },
            {
                "source": "memory",
                "source_alias": "search",
                "layer": "search",
                "header": "?곣봺 Related Memories ?곣봺",
                "text": memory_text,
                "stable_order": 1,
            },
            {
                "source": "kg",
                "source_alias": "kg",
                "layer": "continuity_pack",
                "header": "?곣봺 Relationship Info ?곣봺",
                "text": kg_text,
                "stable_order": 2,
            },
            {
                "source": "saga",
                "source_alias": "saga",
                "layer": "resume_pack",
                "header": "?곣봺 Saga Digest ?곣봺",
                "text": saga_text,
                "stable_order": 3,
            },
            {
                "source": "episode",
                "source_alias": "episode",
                "layer": "episode",
                "header": "?곣봺 Past Summaries ?곣봺",
                "text": episode_text,
                "stable_order": 4,
            },
            {
                "source": "chapter",
                "source_alias": "chapter",
                "layer": "chapter",
                "header": "?곣봺 Chapter Recall ?곣봺",
                "text": chapter_text,
                "stable_order": 5,
            },
            {
                "source": "fallback",
                "source_alias": "fallback",
                "layer": "fallback",
                "header": "?곣봺 Past Chat Reference ?곣봺",
                "text": fallback_text,
                "stable_order": 6,
            },
        ]
        active_section_specs = [spec for spec in section_specs if spec.get("text")]
        active_section_specs.sort(
            key=lambda spec: (
                int(layer_rank_map.get(str(spec.get("layer") or ""), unknown_layer_rank)),
                int(spec.get("stable_order") or 0),
            )
        )
        section_source_order = [
            str(spec.get("source_alias") or spec.get("source") or "").strip()
            for spec in active_section_specs
            if str(spec.get("source_alias") or spec.get("source") or "").strip()
        ]
        section_layer_order = [
            str(spec.get("layer") or "").strip()
            for spec in active_section_specs
            if str(spec.get("layer") or "").strip()
        ]

        sources_used: list[str] = []
        for spec in active_section_specs:
            source_name = str(spec.get("source_alias") or spec.get("source") or "").strip()
            if source_name and source_name not in sources_used:
                sources_used.append(source_name)
        if entity_digest_text:
            sources_used.append("entity_digest")

        # ?? M-3b: recall 釉붾줉留?議곕┰??injection_text ???????????????????????
        # plugin assembleInjectionWithBudget? 13媛??뚮씪誘명꽣 ?꾩껜瑜?媛?몄빞 ?섎?濡?
        # backend?먯꽌??recall 湲곗썝 釉붾줉留??⑥닚 寃고빀 (budget 愿由щ뒗 plugin?먯꽌 ?섑뻾).
        # R-2c遺?곕뒗 profile layer priority ?쒖꽌瑜??곕Ⅸ??
        recall_sections: list[str] = [
            str(spec.get("header") or "") + "\n" + str(spec.get("text") or "")
            for spec in active_section_specs
        ]
        injection_text = "\n\n".join(recall_sections)
        legacy_summary_chars = sum(
            len(part or "")
            for part in [memory_text, kg_text, episode_text, chapter_text, saga_text, fallback_text]
        )
        canon_drift_benchmark = _build_canon_drift_benchmark_r1e(
            canonical_ledger_data,
            canon_text,
            injection_text,
            legacy_summary_chars,
        )
        summary_surface_present = any(
            bool(part)
            for part in (episode_text, chapter_text, saga_text, fallback_text)
        )
        hierarchy_escape_hatch = {
            "status": "active" if bool(scoped_verbatim_support.get("active")) else "inactive",
            "route": _VR18_SCOPED_VERBATIM_POLICY["surface_route"],
            "reason": (
                "summary_surface_missing"
                if bool(scoped_verbatim_support.get("active")) and not summary_surface_present
                else ("support_route_available" if bool(scoped_verbatim_support.get("active")) else "no_direct_evidence_support")
            ),
            "support_surface_first": bool(_VR18_SCOPED_VERBATIM_POLICY["support_surface_first"]),
            "prompt_injection_strategy": _VR18_SCOPED_VERBATIM_POLICY["prompt_injection_strategy"],
            "support_count": int(scoped_verbatim_support.get("count") or 0),
            "latest_direct_evidence_available": bool(latest_direct_evidence_text),
            "summary_surface_present": summary_surface_present,
            "canon_drift_ratio": float((canon_drift_benchmark or {}).get("canon_drift_ratio") or 0.0),
            "policy_version": _VR18_SCOPED_VERBATIM_POLICY["version"],
        }

        # ?? M-3b: input_context_text ???곗냽???듭빱 (理쒓렐 memory ?붿빟) ??????
        # plugin buildInputContext???뱀뀡 4 "Continuity" ?숈씪 洹쒖튃.
        # active_state / KG / supervisor / userInput ?뱀뀡? plugin?먯꽌 蹂댁셿?쒕떎.
        input_context_text = ""
        previous_anchor_text = ""
        try:
            mem_items_ctx = [
                it for it in items
                if isinstance(it, dict) and it.get("source") == "memory" and it.get("turn_index") is not None
            ]
            if mem_items_ctx:
                mem_items_ctx.sort(key=lambda x: x.get("turn_index", 0), reverse=True)
                latest = mem_items_ctx[0]
                raw = latest.get("summary_json") or latest.get("summary") or ""
                if isinstance(raw, str):
                    try:
                        parsed = _json.loads(raw)
                        raw_summary = parsed.get("turn_summary") or parsed.get("summary") or raw
                    except Exception:
                        raw_summary = raw
                elif isinstance(raw, dict):
                    raw_summary = raw.get("turn_summary") or raw.get("summary") or ""
                else:
                    raw_summary = str(raw)
                raw_summary = " ".join(str(raw_summary).split()).strip()
                if raw_summary:
                    preview = raw_summary[:150]
                    previous_anchor_text = "[Previous] " + preview
        except Exception as exc_ctx:
            logger.debug("[prepare-turn] input_context_text build failed (non-fatal): %s", exc_ctx)

        saga_anchor_text = ""
        if saga_text:
            saga_preview = saga_text.splitlines()[0].strip()
            saga_preview = " ".join(saga_preview.split()).strip()
            if len(saga_preview) > 120:
                saga_preview = saga_preview[:117] + "..."
            saga_anchor_text = "[Saga] " + saga_preview

        chapter_anchor_text = ""
        if chapter_text:
            chapter_preview = chapter_text.splitlines()[0].strip()
            chapter_preview = " ".join(chapter_preview.split()).strip()
            if len(chapter_preview) > 120:
                chapter_preview = chapter_preview[:117] + "..."
            chapter_anchor_text = "[Chapter] " + chapter_preview

        # R-2c: input anchor??profile layer priority??留욎떠 ?곗꽑?쒖쐞瑜??곸슜?쒕떎.
        input_anchor_layer = "none"
        input_anchor_source = "none"
        anchor_specs: list[tuple[str, str, str, int]] = []
        if temporal_input_anchor_text:
            anchor_specs.append(("direct_evidence", "temporal_integrity", temporal_input_anchor_text, -1))
        if previous_anchor_text:
            anchor_specs.append(("search", "search", previous_anchor_text, 0))
        if saga_anchor_text:
            anchor_specs.append(("resume_pack", "saga", saga_anchor_text, 1))
        if chapter_anchor_text:
            anchor_specs.append(("chapter", "chapter", chapter_anchor_text, 2))
        if anchor_specs:
            anchor_specs.sort(key=lambda item: (int(layer_rank_map.get(item[0], unknown_layer_rank)), item[3]))
            selected_anchor = anchor_specs[0]
            input_anchor_layer = selected_anchor[0]
            input_anchor_source = selected_anchor[1]
            input_context_text = selected_anchor[2]

        if temporal_input_anchor_text:
            input_anchor_layer = "direct_evidence"
            input_anchor_source = "temporal_integrity"
            input_context_text = temporal_input_anchor_text

        # ?명뭼 而⑦뀓?ㅽ듃???뷀꽣???듭빱瑜?異붽???user ?낅젰 ?덉쭏??蹂댁“?쒕떎.
        entity_anchor_text = _format_entity_anchor_text_m3a(entity_digest_text)
        if entity_anchor_text:
            if input_context_text:
                input_context_text = input_context_text + "\n" + entity_anchor_text
            else:
                input_context_text = entity_anchor_text

        # M-3c: apply_mode ?쒕㎤??洹쒖튃 ?꾨뱶 議곕┰
        # apply_mode_reflected: ?꾩옱 settings???ㅼ젙??apply_mode瑜??먯퐫 ??plugin??backend? ?몄떇 ?쇱튂 ?뺤씤??
        # apply_verdict_rule: effective_user_input??寃곌뎅 梨꾩썙吏???plugin???대뼸寃??뚮퉬?댁빞 ?섎뒗吏 洹쒖튃
        #   no_apply     ??mode=off   ??plugin? effective_user_input??寃곗퐫 ?곸슜?섏? ?딅뒗??
        #   trace_only   ??mode=shadow ??plugin? trace留??섍퀬 payload 援먯껜?섏? ?딅뒗??
        #   apply_if_approved ??mode=reviewed_apply ??apply_verdict媛 approve/partial???뚮쭔 payload 援먯껜
        _apply_mode = (settings_obj.apply_mode if settings_obj else None) or "shadow"
        _apply_verdict_rule_map = {
            "off":            "no_apply",
            "shadow":         "trace_only",
            "reviewed_apply": "apply_if_approved",
        }
        apply_mode_reflected = _apply_mode
        apply_verdict_rule = _apply_verdict_rule_map.get(_apply_mode, "trace_only")
        temporal_packet = _build_temporal_support_packet_sc19(session_state_data=session_state_data)

        packet_composition = {
            "version": _R2D_PACKET_COMPOSITION_SURFACE_VERSION,
            "surface_status": "ready",
            "mode": "read_only_surface",
            "active_profile": profile,
            "section_count": len(section_source_order),
            "section_source_order": section_source_order,
            "section_layer_order": section_layer_order,
            "input_anchor_layer": input_anchor_layer,
            "input_anchor_source": input_anchor_source,
            "entity_anchor_attached": bool(entity_anchor_text),
            "verbatim_support_active": bool(scoped_verbatim_support.get("active")),
            "verbatim_support_count": int(scoped_verbatim_support.get("count") or 0),
            "verbatim_support_surface_priority": list(scoped_verbatim_support.get("surface_priority") or []),
            "verbatim_support_policy_version": scoped_verbatim_support.get("policy_version"),
            "prompt_injection_strategy": _VR18_SCOPED_VERBATIM_POLICY["prompt_injection_strategy"],
            "hierarchy_escape_hatch_status": hierarchy_escape_hatch["status"],
            "temporal_packet_ready": bool(temporal_packet.get("text")),
            "temporal_packet_status": temporal_packet.get("status") or "empty",
            "temporal_packet_policy_version": temporal_packet.get("version"),
        }
        budget_decisions = _build_injection_budget_decisions_t1c(
            recall_result=recall_result,
        )

        return {
            "canon_text": canon_text,
            "temporal_integrity_text": temporal_integrity_text,
            "scoped_verbatim_support_text": scoped_verbatim_support.get("text") or "",
            "scoped_verbatim_support_count": int(scoped_verbatim_support.get("count") or 0),
            "scoped_verbatim_support_items": scoped_verbatim_support.get("items") or [],
            "verbatim_support": scoped_verbatim_support,
            "hierarchy_escape_hatch": hierarchy_escape_hatch,
            "latest_direct_evidence_text": latest_direct_evidence_text,
            "recent_raw_turn_text": recent_raw_turn_text,
            "temporal_packet": temporal_packet,
            "temporal_packet_text": temporal_packet.get("text") or "",
            "memory_text":    memory_text,
            "kg_text":        kg_text,
            "episode_text":   episode_text,
            "chapter_text":   chapter_text,
            "saga_text":      saga_text,
            "fallback_text":  fallback_text,
            "entity_digest_text": entity_digest_text,
            # M-3b: recall 二쇱엯 ?띿뒪??+ ?명뭼 留λ씫
            "injection_text":     injection_text,
            "input_context_text": input_context_text,
            "canon_drift_benchmark": canon_drift_benchmark,
            # M-3c: apply_mode ?쒕㎤??洹쒖튃 ?꾨뱶
            "apply_mode_reflected": apply_mode_reflected,
            "apply_verdict_rule":   apply_verdict_rule,
            "budget_decisions":     budget_decisions,
            # Plugin Main / Sub Review 怨꾩빟 placeholder (M-3d+?먯꽌 梨꾩썙吏?
            "effective_user_input": None,
            "apply_verdict":        None,
            "was_improved":         False,
            "improvement_source":   "none",
            "context_window_profile": profile,
            "saga_priority_applied": saga_priority_applied,
            "sources_used":         sources_used,
            "packet_composition":   packet_composition,
            "assembly_note":        "R-1d/M-3c/P-4d/EA-1h/SC19d: canon-first block + temporal integrity anchor + temporal support packet + apply_mode_reflected + apply_verdict_rule + entity_digest + profile-aware saga priority",
        }
    except Exception as exc:
        logger.warning("[prepare-turn] _bundle_injection_assembly failed: %s", exc)
        return None




