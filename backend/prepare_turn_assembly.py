from __future__ import annotations

import json
import logging


logger = logging.getLogger(__name__)

# M-3a injection assembly text formatters.
# These mirror plugin.js formatting rules and accept recall_result dictionaries.

def _format_memory_text_m3a(recall_result: dict) -> str:
    """Format source='memory' items like plugin formatMemoryBlock."""
    import json as _json
    search = recall_result.get("search") or {}
    items = search.get("items") or []
    mem_items = [it for it in items if isinstance(it, dict) and it.get("source") == "memory"]
    if not mem_items:
        return ""
    parts = []
    for i, item in enumerate(mem_items, 1):
        raw = item.get("summary_json") or item.get("summary") or ""
        if isinstance(raw, str):
            try:
                parsed = _json.loads(raw)
                summary = parsed.get("turn_summary") or parsed.get("summary") or raw
            except Exception:
                summary = raw
        elif isinstance(raw, dict):
            summary = raw.get("turn_summary") or raw.get("summary") or _json.dumps(raw)
        else:
            summary = str(raw)
        summary = " ".join(str(summary).split()).strip()
        if len(summary) > 200:
            summary = summary[:197] + "..."
        imp = item.get("importance")
        imp_str = f" (importance:{imp})" if imp is not None else ""
        parts.append(f"[Memory {i}]{imp_str} {summary}")
    return "\n".join(parts)


def _format_kg_text_m3a(recall_result: dict) -> str:
    """Format KG triples like plugin formatKGBlock."""
    kg = recall_result.get("kg") or {}
    items = kg.get("items") or []
    if not items:
        return ""
    parts = []
    for t in items:
        if not isinstance(t, dict):
            continue
        subj = t.get("subject", "?")
        pred = t.get("predicate", "?")
        obj = t.get("object", "?")
        valid_to = t.get("valid_to")
        line = f"{subj} —({pred})→ {obj}"
        if valid_to is not None:
            line += f" [~turn {valid_to}]"
        parts.append(line)
    return "\n".join(parts)


def _format_episode_text_m3a(recall_result: dict) -> str:
    """Format episode summaries like plugin buildEpisodeBlock."""
    ep = recall_result.get("episode") or {}
    items = ep.get("items") or []
    if not items:
        return ""
    parts = []
    for ep_item in items[:5]:
        if not isinstance(ep_item, dict):
            continue
        from_turn = ep_item.get("from_turn")
        to_turn = ep_item.get("to_turn")
        summary = (ep_item.get("summary_text") or "").strip()
        summary = " ".join(summary.split()).strip()
        if not summary:
            continue
        if len(summary) > 300:
            summary = summary[:297] + "..."
        from_str = str(from_turn) if from_turn is not None else "?"
        to_str = str(to_turn) if to_turn is not None else "?"
        parts.append(f"[Turn {from_str}-{to_str}] {summary}")
    return "\n".join(parts)


def _format_chapter_text_m3a(recall_result: dict) -> str:
    """Format real chapter summaries as low-priority injection support.

    Safety rules:
    - use only source_type="chapter" results, excluding chat_log_fallback
    - use at most two chapters
    - prefer resume_text, then summary_text
    """
    ch = recall_result.get("chapter") or {}
    items = ch.get("items") or []
    if not items:
        return ""

    real_items = [
        it for it in items
        if isinstance(it, dict) and (it.get("source_type") or "") == "chapter"
    ]
    if not real_items:
        return ""

    parts = []
    for ch_item in real_items[:2]:
        from_turn = ch_item.get("from_turn")
        to_turn = ch_item.get("to_turn")
        chapter_index = ch_item.get("chapter_index")
        summary = (ch_item.get("resume_text") or ch_item.get("summary_text") or "").strip()
        summary = " ".join(summary.split()).strip()
        if not summary:
            continue
        if len(summary) > 220:
            summary = summary[:217] + "..."
        from_str = str(from_turn) if from_turn is not None else "?"
        to_str = str(to_turn) if to_turn is not None else "?"
        chapter_label = f"Chapter {chapter_index}" if chapter_index is not None else "Chapter"
        parts.append(f"[{chapter_label} | Turn {from_str}-{to_str}] {summary}")
    return "\n".join(parts)


def _format_fallback_text_m3a(recall_result: dict) -> str:
    """Format source='chat_log' fallback items like plugin formatFallbackBlock."""
    search = recall_result.get("search") or {}
    items = search.get("items") or []
    fb_items = [it for it in items if isinstance(it, dict) and it.get("source") == "chat_log"]
    if not fb_items:
        return ""
    parts = []
    for i, it in enumerate(fb_items, 1):
        content = it.get("content") or ""
        preview = " ".join(str(content).split()).strip()
        if len(preview) > 150:
            preview = preview[:147] + "..."
        parts.append(f"[Past Chat {i}] {preview}")
    return "\n".join(parts)


_LOC_PRED_HINTS_M3A = (
    " at", " in", "inside", "located", "location", "region", "area", "place", "near", "toward",
    "position", "위치", "장소", "지역", "체류", "도착",
)
_ITEM_PRED_HINTS_M3A = (
    "has", "have", "owns", "own", "carry", "carries", "held", "holds", "wield", "equip", "use",
    "item", "weapon", "artifact", "tool", "inventory", "소유", "보유", "장비", "무기", "아이템", "획득",
)


def _clean_short_m3a(raw: object, max_len: int = 80) -> str:
    text = " ".join(str(raw or "").split()).strip()
    if not text:
        return ""
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _json_load_maybe_m3a(raw: object) -> object | None:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return None
        try:
            return json.loads(txt)
        except Exception:
            return None
    return None


def _predicate_matches_m3a(predicate: object, hints: tuple[str, ...]) -> bool:
    p = _clean_short_m3a(predicate, 120).lower()
    if not p:
        return False
    return any(h in p for h in hints)


def _world_rule_note_m3a(value_json: object) -> str:
    parsed = _json_load_maybe_m3a(value_json)
    if isinstance(parsed, dict):
        for k, v in parsed.items():
            if isinstance(v, (str, int, float, bool)):
                key = _clean_short_m3a(k, 24)
                val = _clean_short_m3a(v, 48)
                if val:
                    return f"{key}={val}" if key else val
        if parsed:
            keys = []
            for k in list(parsed.keys())[:2]:
                kk = _clean_short_m3a(k, 20)
                if kk:
                    keys.append(kk)
            if keys:
                return ", ".join(keys)
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (str, int, float, bool)):
                v = _clean_short_m3a(item, 48)
                if v:
                    return v
        if parsed:
            return f"{len(parsed)} entries"
    return _clean_short_m3a(value_json, 64)


def _format_entity_digest_text_m3a(
    recall_result: dict,
    session_state_data: dict | None = None,
    continuity_pack_data: dict | None = None,
) -> str:
    """Build concise character/location/item digest from stored structured data."""
    try:
        kg_items = ((recall_result.get("kg") or {}).get("items") or []) if recall_result else []
        entities = (recall_result.get("entities_extracted") or []) if recall_result else []
        session_state = session_state_data or {}

        characters = session_state.get("characters") or []
        world_rules = session_state.get("world_rules") or []
        if not world_rules and isinstance(continuity_pack_data, dict):
            world_rules = continuity_pack_data.get("world_constraints") or []

        entity_hits = {
            _clean_short_m3a(e, 40).lower()
            for e in entities
            if _clean_short_m3a(e, 40)
        }

        # 1) Characters
        names_seen: set[str] = set()
        character_names: list[str] = []
        for row in characters:
            if not isinstance(row, dict):
                continue
            name = _clean_short_m3a(row.get("character_name"), 40)
            if not name:
                continue
            lk = name.lower()
            if lk in names_seen:
                continue
            names_seen.add(lk)
            character_names.append(name)

        character_names.sort(key=lambda n: (0 if n.lower() in entity_hits else 1, n.lower()))
        character_names = character_names[:4]

        # Build name-to-state mapping from status/personality/relationship payloads.
        char_state_map: dict[str, dict] = {}
        for row in characters:
            if not isinstance(row, dict):
                continue
            cname = _clean_short_m3a(row.get("character_name"), 40)
            if cname:
                char_state_map[cname.lower()] = row

        character_lines: list[str] = []
        for name in character_names:
            hints: list[str] = []

            # Extract location/emotion-style hints from status_json.
            row = char_state_map.get(name.lower()) or {}
            raw_status = _json_load_maybe_m3a(row.get("status_json"))
            if isinstance(raw_status, dict):
                for key in ("location", "position", "emotion", "mood", "health"):
                    val = _clean_short_m3a(raw_status.get(key), 30)
                    if val:
                        hints.append(f"{key}={val}")
                        if len(hints) >= 2:
                            break

            # Extract one relationship hint from relationships_json.
            if len(hints) < 2:
                raw_rels = _json_load_maybe_m3a(row.get("relationships_json"))
                if isinstance(raw_rels, list):
                    for rel_entry in raw_rels[:3]:
                        if not isinstance(rel_entry, dict):
                            continue
                        target = _clean_short_m3a(rel_entry.get("target"), 24)
                        sentiment = _clean_short_m3a(rel_entry.get("sentiment") or rel_entry.get("affection"), 16)
                        if target and sentiment:
                            hints.append(f"{target}:{sentiment}")
                            break

            # Supplement with KG triples when hints are still sparse.
            if len(hints) < 2:
                for tr in kg_items:
                    if not isinstance(tr, dict):
                        continue
                    subj = _clean_short_m3a(tr.get("subject"), 40)
                    pred = _clean_short_m3a(tr.get("predicate"), 28)
                    obj = _clean_short_m3a(tr.get("object"), 40)
                    if not subj or not pred or not obj:
                        continue
                    if name.lower() == subj.lower():
                        hints.append(f"{pred}->{obj}")
                    elif name.lower() == obj.lower():
                        hints.append(f"{subj}-{pred}")
                    else:
                        continue
                    if len(hints) >= 2:
                        break

            if hints:
                character_lines.append(f"- {name}: {' | '.join(hints)}")
            else:
                character_lines.append(f"- {name}")

        # 2) Locations
        location_notes: dict[str, str] = {}
        for row in world_rules:
            if not isinstance(row, dict):
                continue
            scope = _clean_short_m3a(row.get("scope"), 20).lower()
            if scope not in ("location", "region", "place", "area"):
                continue
            loc = _clean_short_m3a(row.get("scope_name") or row.get("key"), 48)
            if not loc or loc in location_notes:
                continue
            location_notes[loc] = _world_rule_note_m3a(row.get("value_json"))

        for tr in kg_items:
            if not isinstance(tr, dict):
                continue
            pred = _clean_short_m3a(tr.get("predicate"), 36)
            if not _predicate_matches_m3a(pred, _LOC_PRED_HINTS_M3A):
                continue
            loc = _clean_short_m3a(tr.get("object"), 48)
            subj = _clean_short_m3a(tr.get("subject"), 28)
            if not loc or loc in location_notes:
                continue
            location_notes[loc] = _clean_short_m3a(f"{subj} {pred}", 56) if subj else pred

        location_lines: list[str] = []
        for loc, note in list(location_notes.items())[:4]:
            if note:
                location_lines.append(f"- {loc}: {note}")
            else:
                location_lines.append(f"- {loc}")

        # 3) Items
        item_notes: dict[str, str] = {}
        for tr in kg_items:
            if not isinstance(tr, dict):
                continue
            pred = _clean_short_m3a(tr.get("predicate"), 36)
            if not _predicate_matches_m3a(pred, _ITEM_PRED_HINTS_M3A):
                continue
            item = _clean_short_m3a(tr.get("object"), 48)
            owner = _clean_short_m3a(tr.get("subject"), 30)
            if not item or item in item_notes:
                continue
            item_notes[item] = _clean_short_m3a(f"{owner} {pred}", 56) if owner else pred

        item_lines: list[str] = []
        for item, note in list(item_notes.items())[:4]:
            if note:
                item_lines.append(f"- {item}: {note}")
            else:
                item_lines.append(f"- {item}")

        sections: list[str] = []
        if character_lines:
            sections.append("[Characters]\n" + "\n".join(character_lines))
        if location_lines:
            sections.append("[Locations]\n" + "\n".join(location_lines))
        if item_lines:
            sections.append("[Items]\n" + "\n".join(item_lines))

        digest = "\n\n".join(sections)
        if len(digest) > 700:
            digest = digest[:697] + "..."
        return digest
    except Exception as exc:
        logger.debug("[prepare-turn] entity digest build failed (non-fatal): %s", exc)
        return ""


def _format_entity_anchor_text_m3a(entity_digest_text: str) -> str:
    """Build one-line anchor for input context from entity digest text."""
    if not entity_digest_text:
        return ""

    buckets = {"characters": [], "locations": [], "items": []}
    current: str | None = None
    for raw_line in entity_digest_text.splitlines():
        line = raw_line.strip()
        low = line.lower()
        if low == "[characters]":
            current = "characters"
            continue
        if low == "[locations]":
            current = "locations"
            continue
        if low == "[items]":
            current = "items"
            continue
        if current and line.startswith("- "):
            name = _clean_short_m3a(line[2:].split(":", 1)[0], 20)
            if name and name not in buckets[current]:
                buckets[current].append(name)

    parts: list[str] = []
    if buckets["characters"]:
        parts.append("C:" + ", ".join(buckets["characters"][:2]))
    if buckets["locations"]:
        parts.append("L:" + ", ".join(buckets["locations"][:2]))
    if buckets["items"]:
        parts.append("I:" + ", ".join(buckets["items"][:2]))

    if not parts:
        return ""
    return "[Entity] " + " | ".join(parts)


