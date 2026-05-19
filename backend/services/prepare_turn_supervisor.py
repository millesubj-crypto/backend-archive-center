"""
backend/services/prepare_turn_supervisor.py

M-2c / M-2d: prepare-turn supervisor input pack and writeback preview builders.
Extracted from main.py as part of Step 21.5 accelerated core slice.

This module MUST NOT import backend.main.
"""

from typing import Callable

# ---------------------------------------------------------------------------
# Guide mode suffixes (M-2c)
# ---------------------------------------------------------------------------

_GUIDE_MODE_SUFFIXES_M2C: dict[str, str] = {
    "standard": (
        "\n[Narrative Guide —Standard]\n"
        "Focus on: tension pacing, foreshadowing, subplot weaving.\n"
        "Maintain balanced tone between action and dialogue.\n"
        "Avoid abrupt tonal shifts unless dramatically justified."
    ),
    "romantic": (
        "\n[Narrative Guide —Romantic]\n"
        "Emphasize emotional dynamics: affection, longing, jealousy, intimacy.\n"
        "Track relationship tension/affection metrics closely.\n"
        "Slow-burn pacing preferred; let moments breathe.\n"
        "Dialogue subtext > explicit declarations."
    ),
    "action": (
        "\n[Narrative Guide —Action]\n"
        "Prioritize momentum: short sentences, visceral descriptions.\n"
        "Raise stakes progressively; consequences must feel real.\n"
        "Limit exposition during combat/chase; weave it into downtime."
    ),
    "mature_soft": (
        "\n[Narrative Guide —Mature (Sensual)]\n"
        "Adult content is permitted when story-appropriate.\n"
        "Prefer sensory, suggestive, indirect description.\n"
        "Atmosphere and emotion over explicit mechanics.\n"
        "Respect character agency and established boundaries."
    ),
    "mature_direct": (
        "\n[Narrative Guide —Mature (Explicit)]\n"
        "Adult content is permitted; direct and vivid description allowed.\n"
        "Maintain literary quality even in explicit scenes.\n"
        "Character voice and emotional context remain paramount.\n"
        "Do not reduce characters to mere participants; inner thoughts matter."
    ),
}

_NARRATIVE_STANCE_SUFFIXES_M2C: dict[str, str] = {
    "reactive": (
        "\n[Story Initiative —Reactive]\n"
        "Stay close to the user's immediate lead and the current scene.\n"
        "If the user expresses caution, hesitation, or uncertainty, remain in observation, "
        "clarification, or low-risk preparation rather than pushing action.\n"
        "Do not initiate entry, unlock barriers, assign a plan, or commit companions to a risky "
        "move unless the user explicitly asks for that step.\n"
        "Advance existing threads only when the current exchange clearly opens space for it, "
        "and keep any suggestion small and reversible."
    ),
    "proactive": (
        "\n[Story Initiative —Proactive]\n"
        "You may introduce one plausible next beat or complication when continuity supports it.\n"
        "Initiative must grow from existing tensions, hooks, promises, or scene context.\n"
        "When the user is still deciding, propose the next beat rather than executing the decision "
        "on the user's behalf.\n"
        "Do not override the user's intent, skip causal steps, or force abrupt scene changes."
    ),
    "balanced": (
        "\n[Story Initiative —Balanced]\n"
        "You may add one gentle next-beat nudge when it naturally fits the current scene.\n"
        "Keep the response anchored to the user's immediate intent and the current arc, "
        "and suggest rather than execute the next step.\n"
        "Avoid abrupt escalation, forced twists, or hard scene jumps."
    ),
}

_NARRATIVE_STANCE_BOUNDS_M2C: dict[str, dict] = {
    "reactive": {
        "emphasis": ["user-led follow-through", "observation before action", "low-risk option framing"],
        "forbidden_moves": [
            "unlocking barriers or initiating entry without explicit user intent",
            "committing the group to a risky plan on the user's behalf",
            "inventing urgent danger to force motion",
        ],
        "max_new_beats": 0,
        "allow_scene_jump": False,
    },
    "proactive": {
        "emphasis": ["causal next-beat proposal", "continuity-aware tension increase", "bounded steering"],
        "forbidden_moves": [
            "forcing irreversible turns without buildup",
            "overwriting the user's immediate intent",
            "turning a cautious pause into immediate entry or confrontation without buy-in",
        ],
        "max_new_beats": 1,
        "allow_scene_jump": False,
    },
    "balanced": {
        "emphasis": ["gentle next-beat nudges", "continuity-aware escalation", "conversation momentum"],
        "forbidden_moves": [
            "hard scene cut without setup",
            "forcing a dramatic turn too early",
            "executing a risky step before the user agrees to it",
        ],
        "max_new_beats": 1,
        "allow_scene_jump": False,
    },
}


# ---------------------------------------------------------------------------
# M-2c helpers
# ---------------------------------------------------------------------------

def _m2c_build_auto_advance_hint_text(nc_data: dict, trigger: str) -> str:
    """M-2c: Build auto-advance hint text from NC data and explicit runtime trigger.
    Aligns with plugin.js buildAutoAdvanceHint.
    """
    if not trigger or trigger == "none":
        return ""
    if not nc_data or nc_data.get("skeleton_only", True):
        return ""
    sp = nc_data.get("story_plan") or {}
    dr = nc_data.get("director") or {}
    next_beats = sp.get("next_beats") or []
    first_beat = str(next_beats[0]).strip() if next_beats and next_beats[0] else ""
    scene_mandate = str(dr.get("scene_mandate") or "").strip()
    selected_beat = first_beat
    if not selected_beat and not scene_mandate:
        return ""
    trigger_labels = {
        "empty": "empty input",
        "resume": "manual resume",
        "idle": "idle reentry",
    }
    trigger_label = trigger_labels.get(trigger, trigger)
    lines = [
        "\n[Auto-advance Reentry Hint - " + trigger_label + "]",
        "The user's input is absent or signals a scene reentry.",
        "Use this hint to continue naturally from where the story was.",
        "BOUNDS: Do NOT jump to a new scene. Do NOT add unsolicited commitments or dramatic revelations.",
        "Treat this as a gentle nudge only - never override explicit user input.",
    ]
    if selected_beat:
        lines.append("Selected beat: " + selected_beat)
    if scene_mandate:
        lines.append("Current scene focus: " + scene_mandate)
    lines.append("Fallback policy: conservative continuation only; no scene jump.")
    lines.append("(If user provides explicit input, ignore this hint and follow user intent.)")
    return "\n".join(lines)


def build_persistent_guidance_text(
    nc_data: dict | None,
    *,
    build_story_guidance_surface: Callable[[dict | None, dict | None], dict],
) -> str:
    """M-2c: NC data to supervisor persistent_guidance text.
    plugin.js buildPersistentGuidanceSuffix alignment (snake_case).
    """
    if not nc_data or nc_data.get("skeleton_only", True):
        return ""
    sp = nc_data.get("story_plan") or {}
    dr = nc_data.get("director") or {}
    contract = nc_data.get("story_guidance") or build_story_guidance_surface(sp, dr)
    story_frame = contract.get("story_frame") if isinstance(contract, dict) else {}
    turn_directives = contract.get("turn_directives") if isinstance(contract, dict) else {}
    precedence = contract.get("precedence") if isinstance(contract, dict) else {}
    if (
        not sp.get("current_arc") and not sp.get("narrative_goal")
        and not (sp.get("next_beats") or [])
        and not (dr.get("required_outcomes") or [])
        and not (dr.get("forbidden_moves") or [])
    ):
        return ""
    lines = ["\n[Persistent Story Guidance ??multi-turn carry-over]"]
    lines.append("Story frame:")
    if story_frame.get("arc_focus"):
        lines.append("- Arc focus: " + str(story_frame["arc_focus"]))
    if story_frame.get("narrative_drive"):
        lines.append("- Narrative drive: " + str(story_frame["narrative_drive"]))
    tensions = story_frame.get("live_tensions") or []
    if tensions:
        lines.append("- Live tensions: " + " / ".join(str(t) for t in tensions[:3]))
    beat_queue = story_frame.get("beat_queue") or []
    if beat_queue:
        lines.append("- Beat queue:")
        for b in beat_queue[:3]:
            if b:
                lines.append("  * " + str(b))
    anchors = story_frame.get("carry_threads") or []
    if anchors:
        lines.append("- Carry threads: " + " / ".join(str(a) for a in anchors[:3]))
    lines.append("Turn directives:")
    if turn_directives.get("scene_drive"):
        lines.append("- Scene drive: " + str(turn_directives["scene_drive"]))
    required = turn_directives.get("carry_targets") or []
    if required:
        lines.append("- Carry targets:")
        for r in required[:3]:
            if r:
                lines.append("  * " + str(r))
    forbidden = turn_directives.get("blocked_routes") or []
    if forbidden:
        lines.append("- Blocked routes:")
        for f in forbidden[:3]:
            if f:
                lines.append("  * " + str(f))
    pressure = (turn_directives.get("tempo_band") or "").strip()
    if pressure:
        lines.append("- Tempo band: " + pressure)
    ending_requirement = (turn_directives.get("handoff_edge") or "").strip()
    if ending_requirement:
        lines.append("- Handoff edge: " + ending_requirement)
    fail_mode = turn_directives.get("fail_mode") or {}
    fail_mode_label = (fail_mode.get("mode") or "").strip() if isinstance(fail_mode, dict) else ""
    if fail_mode_label:
        lines.append("- Fail mode: " + fail_mode_label.replace("_", " "))
    pg = turn_directives.get("voice_guardrails") or []
    if pg:
        lines.append("- Voice guardrails: " + "; ".join(str(g) for g in pg[:2]))
    wg = turn_directives.get("setting_guardrails") or []
    if wg:
        lines.append("- Setting guardrails: " + "; ".join(str(g) for g in wg[:2]))
    precedence_note = (precedence.get("precedence_note") or "").strip()
    if precedence_note:
        lines.append("Story guidance precedence: " + precedence_note)
    lines.append(
        "(This is persistent story context. Explicit user input takes priority. "
        "Do not reference these notes mechanically.)"
    )
    return "\n".join(lines)


def _resolve_auto_advance_trigger(continuity_trigger_mode: str) -> str:
    """Map continuity trigger mode to auto-advance trigger."""
    ctm = (continuity_trigger_mode or "none").strip().lower()
    if ctm in ("resume", "idle", "empty"):
        return ctm
    if ctm == "empty_input":
        return "empty"
    if ctm == "manual_resume":
        return "resume"
    if ctm == "idle_reentry":
        return "idle"
    return "none"


# ---------------------------------------------------------------------------
# Public: supervisor input pack
# ---------------------------------------------------------------------------

def bundle_supervisor_input_pack(
    sid: str,
    settings_obj,
    nc_data: dict | None,
    continuity_trigger_mode: str,
    raw_user_input: str = "",
    *,
    build_persistent_guidance_text: Callable[[dict | None], str],
    fetch_momentum_packet: Callable[[str], dict | None],
    logger,
) -> dict | None:
    """M-2c: supervisor 호출 전 의존성을 미리 계산하여 dict로 반환.

    반환 형식:
    {
      "persistent_guidance": str,
      "guide_suffix":        str,
      "narrative_stance_suffix":   str,
      "narrative_stance_bounds":   dict,
      "auto_advance_trigger":str,
      "auto_advance_hint":   str,
      "final_guidance_suffix":str,
      "momentum_packet":     dict|None,
    }
    실패 시 None.
    """
    try:
        guide_mode = (getattr(settings_obj, "guide_mode", None) or "off").strip()
        narrative_stance = (getattr(settings_obj, "narrative_stance", None) or "balanced").strip()

        # 1. persistent guidance suffix (callback — may depend on main.py)
        persistent_guidance = build_persistent_guidance_text(nc_data) if nc_data else ""

        # 2. guide suffix
        guide_suffix = _GUIDE_MODE_SUFFIXES_M2C.get(guide_mode, "")

        # 3. initiative suffix + bounds
        narrative_stance_suffix = _NARRATIVE_STANCE_SUFFIXES_M2C.get(
            narrative_stance, _NARRATIVE_STANCE_SUFFIXES_M2C["balanced"]
        )
        narrative_stance_bounds = _NARRATIVE_STANCE_BOUNDS_M2C.get(
            narrative_stance, _NARRATIVE_STANCE_BOUNDS_M2C["balanced"]
        )

        # 4. auto_advance_trigger
        auto_advance_trigger = _resolve_auto_advance_trigger(continuity_trigger_mode)

        # 5. auto_advance_hint
        auto_advance_hint = (
            _m2c_build_auto_advance_hint_text(nc_data, auto_advance_trigger) if nc_data else ""
        )

        # 6. final guidance suffix
        if auto_advance_hint and persistent_guidance:
            final_guidance_suffix = persistent_guidance + "\n" + auto_advance_hint
        elif auto_advance_hint:
            final_guidance_suffix = auto_advance_hint
        else:
            final_guidance_suffix = persistent_guidance

        # 7. momentum_packet (optional, fail-safe; callback — may depend on DB)
        momentum_packet = fetch_momentum_packet(sid)

        return {
            "persistent_guidance": persistent_guidance,
            "guide_suffix": guide_suffix,
            "narrative_stance_suffix": narrative_stance_suffix,
            "narrative_stance_bounds": narrative_stance_bounds,
            "auto_advance_trigger": auto_advance_trigger,
            "auto_advance_hint": auto_advance_hint,
            "final_guidance_suffix": final_guidance_suffix,
            "momentum_packet": momentum_packet,
        }
    except Exception as exc:
        logger.warning("[prepare-turn] bundle_supervisor_input_pack failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public: writeback preview
# ---------------------------------------------------------------------------

def bundle_writeback_preview(
    session_state_data,
    narrative_control_data,
    recall_data,
    supervisor_pack_data,
    settings_obj,
    *,
    logger,
) -> dict | None:
    """M-2d post-turn writeback preview shape."""
    try:
        guide_mode = "off"
        narrative_stance = "reactive"
        if settings_obj:
            guide_mode = getattr(settings_obj, "guide_mode", "off") or "off"
            narrative_stance = getattr(settings_obj, "narrative_stance", "reactive") or "reactive"
        return {
            "turns_to_save": 1,
            "storyline_sync_expected": supervisor_pack_data is not None,
            "guidance_refresh_expected": narrative_control_data is not None,
            "episode_check_expected": recall_data is not None,
            "maintenance_enqueue_expected": True,
            "guide_mode": guide_mode,
            "narrative_stance": narrative_stance,
            "session_state_present": session_state_data is not None,
        }
    except Exception as exc:
        logger.warning("[prepare-turn] bundle_writeback_preview failed: %s", exc)
        return None