import json
import logging
import re

import httpx

from backend.prompt_store import read_prompt
from backend.settings import settings

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(self) -> None:
        self.endpoint = getattr(settings, "PROJECT_SUPERVISOR_ENDPOINT", "") or ""
        self.api_key = getattr(settings, "PROJECT_SUPERVISOR_API_KEY", "") or ""
        self.model = getattr(settings, "PROJECT_SUPERVISOR_MODEL", "") or ""
        self.timeout = getattr(settings, "SUPERVISOR_TIMEOUT", 30) or 30
        self.prompt_name = "supervisor_system"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def get_directive(
        self,
        context_messages: list[dict],
        wake_up_context: str = "",
        chat_session_id: str | None = None,
        storylines_context: str = "",
        guide_suffix: str = "",
        narrative_stance: str = "balanced",
        narrative_stance_suffix: str = "",
        narrative_stance_bounds: dict | None = None,
        momentum_packet: dict | None = None,
        persistent_guidance: str = "",
    ) -> dict | str:
        """Return a directive packet dict, or a safe default on any failure."""
        logger.info("Supervisor.get_directive started (session=%s)", chat_session_id or "default")
        try:
            if not isinstance(context_messages, list):
                context_messages = []
            result = self._extract_directive(
                context_messages,
                wake_up_context or "",
                storylines_context or "",
                guide_suffix or "",
                narrative_stance or "balanced",
                narrative_stance_suffix or "",
                narrative_stance_bounds if isinstance(narrative_stance_bounds, dict) else {},
                momentum_packet if isinstance(momentum_packet, dict) else None,
                persistent_guidance or "",
            )
            if result is None:
                logger.info("Supervisor: extraction failed, using default directive")
                return self._default_directive()
            return result
        except Exception as exc:
            logger.warning("Supervisor.get_directive failed: %s", exc)
            return self._default_directive()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _extract_directive(
        self,
        context_messages: list[dict],
        wake_up_context: str = "",
        storylines_context: str = "",
        guide_suffix: str = "",
        narrative_stance: str = "balanced",
        narrative_stance_suffix: str = "",
        narrative_stance_bounds: dict | None = None,
        momentum_packet: dict | None = None,
        persistent_guidance: str = "",
    ) -> dict | None:
        try:
            system_prompt = self._read_system_prompt()
            if not system_prompt:
                return None

            # E-5: guide mode suffix—system prompt 끝에 붙임
            if guide_suffix and guide_suffix.strip():
                system_prompt = system_prompt + "\n" + guide_suffix.strip()

            if narrative_stance_suffix and narrative_stance_suffix.strip():
                system_prompt = system_prompt + "\n" + narrative_stance_suffix.strip()

            narrative_stance_bounds_text = self._format_narrative_stance_bounds(narrative_stance, narrative_stance_bounds)
            if narrative_stance_bounds_text:
                system_prompt = system_prompt + "\n" + narrative_stance_bounds_text

            # I-2c: momentum packet — packet_status가 ready 또는 partial일 때만 주입
            momentum_suffix = self._format_momentum_suffix(momentum_packet)
            if momentum_suffix:
                system_prompt = system_prompt + "\n" + momentum_suffix

            # K-4a: persistent guidance suffix — 비어있지 않으면 마지막에 주입
            # momentum packet 이후, payload 직전에 위치시켜 가장 최신 컨텍스트를 유지한다.
            if persistent_guidance and persistent_guidance.strip():
                system_prompt = system_prompt + "\n" + persistent_guidance.strip()

            if not self.endpoint or not self.api_key or not self.model:
                logger.warning("Supervisor settings missing; extraction skipped.")
                return None

            payload = self._build_payload(system_prompt, context_messages, wake_up_context, storylines_context)
            raw_text = self._request_completion(payload)
            if not raw_text:
                return None

            parsed = self._parse_response_json(raw_text)
            if parsed is None:
                logger.warning("Supervisor JSON parse failed.")
                return None

            logger.info("Supervisor JSON parse succeeded.")
            return self._normalize_directive(parsed)
        except Exception as exc:
            logger.warning("Supervisor extraction failed: %s", exc)
            return None

    def _format_momentum_suffix(self, momentum_packet: dict | None) -> str:
        """I-2c: momentum packet을 supervisor system prompt suffix로 변환한다.

        packet_status가 ready 또는 partial일 때만 suffix를 생성한다.
        empty / None / skeleton_only 이면 빈 문자열 반환.
        """
        if not momentum_packet or not isinstance(momentum_packet, dict):
            return ""
        status = momentum_packet.get("packet_status", "empty")
        if status not in ("ready", "partial"):
            return ""

        lines: list[str] = ["[Story Momentum — Current Session]"]

        def _fmt_items(slot_name: str, heading: str) -> None:
            items = momentum_packet.get(slot_name) or []
            if not items:
                return
            lines.append(heading)
            for item in items[:3]:
                label = item.get("label", "") if isinstance(item, dict) else str(item)
                if label:
                    lines.append(f"- {label}")

        _fmt_items("next_pressure",    "Next pressure points:")
        _fmt_items("payoff_candidates", "Ready to resolve:")
        _fmt_items("tension_to_reuse", "Forgotten threads to weave back in:")
        _fmt_items("beats_to_avoid",   "Avoid repeating these beats:")

        if len(lines) <= 1:
            return ""
        lines.append("(Use the above as narrative context — do not reference it mechanically.)")
        return "\n".join(lines)

    def _format_narrative_stance_bounds(self, narrative_stance: str, narrative_stance_bounds: dict | None) -> str:
        bounds = narrative_stance_bounds if isinstance(narrative_stance_bounds, dict) else {}
        if not bounds:
            return ""

        lines: list[str] = [f"[Initiative Bounds — {narrative_stance or 'balanced'}]"]

        max_new_beats = bounds.get("max_new_beats")
        if isinstance(max_new_beats, int) and max_new_beats >= 0:
            lines.append(f"At most {max_new_beats} new beat(s) may be introduced in a single response.")

        if bounds.get("allow_scene_jump") is False:
            lines.append("Do not perform abrupt scene jumps without clear causal setup.")

        emphasis = bounds.get("emphasis") if isinstance(bounds.get("emphasis"), list) else []
        if emphasis:
            lines.append("Prefer:")
            for item in emphasis:
                if item:
                    lines.append(f"- {item}")

        forbidden_moves = bounds.get("forbidden_moves") if isinstance(bounds.get("forbidden_moves"), list) else []
        if forbidden_moves:
            lines.append("Avoid:")
            for item in forbidden_moves:
                if item:
                    lines.append(f"- {item}")

        return "\n".join(lines)

    def _default_directive(self) -> dict:
        return {
            "book_author": {
                "current_arc": "안정적인 흐름을 유지",
                "narrative_goal": "현재 감정선을 자연스럽게 이어간다",
                "next_beats": [],
                "guardrails": ["감정선 급변 금지", "뜬금없는 대형 사건 금지"],
            },
            "director": {
                "scene_mandate": "현재 장면의 분위기를 유지하며 작은 파문만 허용",
                "required_outcomes": [],
                "forbidden_moves": ["갑작스러운 전투 발생", "캐릭터 성격 급변"],
                "pressure_level": "low",
            },
            "section_world": {
                "applies": False,
                "rules": [],
                "confidence": "low",
            },
        }

    # ------------------------------------------------------------------ #
    #  Prompt / Payload
    # ------------------------------------------------------------------ #

    def _read_system_prompt(self) -> str | None:
        try:
            return read_prompt(self.prompt_name)
        except Exception as exc:
            logger.warning("Supervisor prompt load failed: %s", exc)
            return None

    def _build_payload(
        self,
        system_prompt: str,
        context_messages: list[dict],
        wake_up_context: str,
        storylines_context: str = "",
    ) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": self._build_user_prompt(context_messages, wake_up_context, storylines_context),
                },
            ],
        }

    def _build_user_prompt(
        self,
        context_messages: list[dict],
        wake_up_context: str,
        storylines_context: str = "",
    ) -> str:
        context_text = self._format_context(context_messages)
        parts: list[str] = []

        parts.append("<Recent_Conversation>")
        parts.append(context_text if context_text else "(없음)")
        parts.append("</Recent_Conversation>")

        if wake_up_context and wake_up_context.strip():
            parts.append("")
            parts.append("<Wake_Up_Context>")
            parts.append(wake_up_context.strip())
            parts.append("</Wake_Up_Context>")

        if storylines_context and storylines_context.strip():
            parts.append("")
            parts.append("<Active_Storylines>")
            parts.append(storylines_context.strip())
            parts.append("</Active_Storylines>")

        return "\n".join(parts)

    def _format_context(self, context_messages: list[dict]) -> str:
        if not context_messages or not isinstance(context_messages, list):
            return ""

        recent_items = context_messages[-10:]
        lines: list[str] = []
        for item in recent_items:
            role = str(item.get("role", "unknown")).strip() or "unknown"
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            compact = " ".join(content.split())
            if len(compact) > 500:
                compact = compact[:497] + "..."
            lines.append(f"[{role}] {compact}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  LLM call
    # ------------------------------------------------------------------ #

    def _build_chat_url(self) -> str:
        normalized = self.endpoint.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return normalized + "/chat/completions"

    def _request_completion(self, payload: dict) -> str | None:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self._build_chat_url(),
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
            return self._extract_message_text(data)
        except Exception as exc:
            logger.warning("Supervisor LLM call failed: %s", exc)
            return None

    def _extract_message_text(self, data: dict) -> str | None:
        try:
            choices = data.get("choices") or []
            if not choices:
                return None
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if text:
                        text_parts.append(str(text))
                if text_parts:
                    return "\n".join(text_parts).strip()
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  JSON parsing & normalization
    # ------------------------------------------------------------------ #

    def _parse_response_json(self, response_text: str) -> dict | None:
        code_block_match = re.search(
            r"```json\s*(.*?)\s*```",
            response_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        candidates: list[str] = []
        if code_block_match:
            candidates.append(code_block_match.group(1).strip())
        candidates.append(response_text.strip())

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed

        # ── 기존 로직 실패 시: 잘린 JSON 수리 시도 ──
        for candidate in candidates:
            repaired = self._try_repair_truncated_json(candidate)
            if repaired is not None:
                logger.info("JSON repaired successfully (len=%d)", len(candidate))
                return repaired

        return None

    def _try_repair_truncated_json(self, text: str) -> dict | None:
        """잘린 JSON을 bracket counting으로 수리 시도."""
        try:
            cleaned = text.strip()
            # 후행 콤마 제거
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
            # smart quote 정규화
            cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"')
            cleaned = cleaned.replace("\u2018", "'").replace("\u2019", "'")

            if not cleaned.startswith("{"):
                idx = cleaned.find("{")
                if idx < 0:
                    return None
                cleaned = cleaned[idx:]

            in_string = False
            escaped = False
            brace_depth = 0
            bracket_depth = 0
            for ch in cleaned:
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth = max(0, brace_depth - 1)
                elif ch == "[":
                    bracket_depth += 1
                elif ch == "]":
                    bracket_depth = max(0, bracket_depth - 1)

            # 열린 문자열 닫기
            if in_string:
                cleaned += '"'
            # 부족한 bracket 추가
            cleaned += "]" * bracket_depth
            cleaned += "}" * brace_depth
            # 후행 콤마 재정리
            cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return None

    def _normalize_directive(self, parsed: dict) -> dict:
        """Phase 4-1: book_author + director 구조로 정규화.

        이전 형식(scene_read/drama_plan/...)이 들어오면 새 형식으로 마이그레이션한다.
        """
        try:
            defaults = self._default_directive()

            # ── 레거시 형식 → 새 형식 마이그레이션 ──
            if "book_author" not in parsed and "director" not in parsed:
                return self._migrate_legacy_directive(parsed)

            directive = dict(parsed)

            # ── book_author 정규화 ──
            sa = directive.get("book_author")
            if not isinstance(sa, dict):
                directive["book_author"] = dict(defaults["book_author"])
            else:
                sa.setdefault("current_arc", defaults["book_author"]["current_arc"])
                sa.setdefault("narrative_goal", defaults["book_author"]["narrative_goal"])
                if not isinstance(sa.get("next_beats"), list):
                    sa["next_beats"] = defaults["book_author"]["next_beats"]
                if not isinstance(sa.get("guardrails"), list):
                    sa["guardrails"] = defaults["book_author"]["guardrails"]

            # ── director 정규화 ──
            dr = directive.get("director")
            if not isinstance(dr, dict):
                directive["director"] = dict(defaults["director"])
            else:
                dr.setdefault("scene_mandate", defaults["director"]["scene_mandate"])
                if not isinstance(dr.get("required_outcomes"), list):
                    dr["required_outcomes"] = defaults["director"]["required_outcomes"]
                if not isinstance(dr.get("forbidden_moves"), list):
                    dr["forbidden_moves"] = defaults["director"]["forbidden_moves"]
                pl = dr.get("pressure_level")
                if pl not in ("low", "medium", "high", "critical"):
                    dr["pressure_level"] = defaults["director"]["pressure_level"]

            # ── section_world 정규화 (선택적) ──
            sw = directive.get("section_world")
            if isinstance(sw, dict):
                if not isinstance(sw.get("applies"), bool):
                    sw["applies"] = False
                if not isinstance(sw.get("rules"), list):
                    sw["rules"] = []
                conf = sw.get("confidence")
                if conf not in ("low", "medium", "high"):
                    sw["confidence"] = "low"
            else:
                directive["section_world"] = dict(defaults["section_world"])

            return directive
        except Exception as exc:
            logger.warning("Supervisor _normalize_directive failed: %s", exc)
            return self._default_directive()

    def _migrate_legacy_directive(self, parsed: dict) -> dict:
        """이전 scene_read/drama_plan/... 형식을 book_author+director로 변환."""
        defaults = self._default_directive()
        result = {"book_author": dict(defaults["book_author"]), "director": dict(defaults["director"]), "section_world": dict(defaults["section_world"])}

        # scene_read → director.scene_mandate 매핑
        scene_read = parsed.get("scene_read") or {}
        mood = scene_read.get("current_mood", "")
        tension = scene_read.get("tension_level", 2)
        try:
            tension = int(tension)
        except Exception:
            tension = 2
        tension = max(1, min(5, tension))

        # drama_plan → book_author 매핑
        drama = parsed.get("drama_plan") or {}
        goal = drama.get("goal", "")

        if mood:
            result["director"]["scene_mandate"] = mood
        if goal:
            result["book_author"]["narrative_goal"] = goal

        # must_include → director.required_outcomes
        mi = parsed.get("must_include")
        if isinstance(mi, list) and mi:
            result["director"]["required_outcomes"] = mi

        # avoid → director.forbidden_moves
        avoid = parsed.get("avoid")
        if isinstance(avoid, list) and avoid:
            result["director"]["forbidden_moves"] = avoid

        # writer_instruction → book_author.current_arc
        wi = parsed.get("writer_instruction")
        if wi and isinstance(wi, str):
            result["book_author"]["current_arc"] = wi[:200]

        # causal_hooks → book_author.next_beats
        hooks = parsed.get("causal_hooks")
        if isinstance(hooks, list) and hooks:
            result["book_author"]["next_beats"] = [
                str(h.get("source", ""))[:100] for h in hooks[:3] if isinstance(h, dict)
            ]

        # tension → pressure_level 매핑
        if tension <= 2:
            result["director"]["pressure_level"] = "low"
        elif tension <= 3:
            result["director"]["pressure_level"] = "medium"
        elif tension <= 4:
            result["director"]["pressure_level"] = "high"
        else:
            result["director"]["pressure_level"] = "critical"

        return result
