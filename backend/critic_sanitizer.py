import re


_CRITIC_CONTROL_SECTION_KEYWORDS = (
    "unresolved threads",
    "open threads",
    "pending threads",
    "continuity hooks",
    "director notes",
    "narrative guidance",
    "narrative control",
    "scene goals",
    "prompt guidance",
    "prompt template",
    "toggle expansion",
    "response template",
    "reasoning guideline",
    "reasoning guidelines",
    "archive label",
    "current input",
    "long-term memory archive",
    "system prompt",
    "feedback",
    "tags",
    "tag list",
    "behavior",
    "behaviour",
    "common behavior",
    "common behaviour",
    "behavior guide",
    "behaviour guide",
    "behavior rules",
    "behaviour rules",
    "instructions",
    "instruction",
    "rules",
    "rule",
    "response rules",
    "response format",
    "style guide",
    "writing guide",
    "persona",
    "pov",
    "author note",
    "system note",
    "meta note",
    "미해결 스레드",
    "열린 스레드",
    "연속성 훅",
    "장기 기억 아카이브",
    "아카이브 라벨",
    "피드백",
    "태그",
    "행동 지침",
    "행동 규칙",
    "지침",
    "규칙",
    "응답 규칙",
    "응답 형식",
    "시점",
)

_CRITIC_CONTROL_ROLE_PREFIXES = (
    "system",
    "developer",
    "prompt",
    "instruction",
    "instructions",
    "author note",
    "system note",
)

_CRITIC_INLINE_CONTROL_KEYWORDS = _CRITIC_CONTROL_SECTION_KEYWORDS + (
    "respond",
    "response",
    "roleplay",
    "format",
    "style",
)


def _normalize_control_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def looks_like_critic_section_header(line: str) -> bool:
    stripped = str(line or "").strip()
    if len(stripped) < 4:
        return False
    return (
        (stripped.startswith("━━") and stripped.endswith("━━"))
        or (stripped.startswith("[") and stripped.endswith("]") and len(stripped) <= 80)
        or (stripped.startswith("#") and len(stripped) <= 120)
        or (stripped.startswith("<") and stripped.endswith(">") and len(stripped) <= 80)
    )


def is_critic_control_section_header(line: str) -> bool:
    stripped = str(line or "").strip()
    if not looks_like_critic_section_header(stripped):
        return False
    core = _normalize_control_text(stripped.strip("━[]#<>/ "))
    return any(keyword in core for keyword in _CRITIC_CONTROL_SECTION_KEYWORDS)


def _is_role_prefixed_control_line(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped or len(stripped) > 220 or ":" not in stripped:
        return False

    prefix, remainder = stripped.split(":", 1)
    normalized_prefix = _normalize_control_text(prefix)
    if normalized_prefix not in _CRITIC_CONTROL_ROLE_PREFIXES:
        return False

    normalized_remainder = _normalize_control_text(remainder)
    return any(keyword in normalized_remainder for keyword in _CRITIC_INLINE_CONTROL_KEYWORDS)


def _is_critic_control_marker_line(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    if is_critic_control_section_header(stripped):
        return True
    if _is_role_prefixed_control_line(stripped):
        return True

    normalized = _normalize_control_text(stripped)
    if not any(keyword in normalized for keyword in _CRITIC_CONTROL_SECTION_KEYWORDS):
        return False
    if len(stripped) > 160:
        return False

    return (
        stripped.startswith(("#", "- ", "* ", "• ", "<"))
        or stripped.endswith(":")
        or ": " in stripped
    )


_PRIVATE_REASONING_NAME_HINTS = (
    "thought",
    "think",
    "reason",
    "analysis",
    "scratch",
    "reflect",
    "deliberat",
    "monologue",
    "rationale",
    "chainofthought",
    "cot",
)

_PRIVATE_REASONING_BODY_CUES = (
    re.compile(r"\b(?:i|we)\s+(?:need|should|must|will|can|want|have)\b", re.IGNORECASE),
    re.compile(r"\blet'?s\b", re.IGNORECASE),
    re.compile(r"\b(?:user|assistant|response|reply|instruction|instructions|system prompt|prompt|context|goal|strategy|policy)\b", re.IGNORECASE),
    re.compile(r"\b(?:analysis|analyze|reasoning|reason|thinking|thought process|chain of thought|scratchpad|reflection|deliberation)\b", re.IGNORECASE),
    re.compile(r"(?:사용자|응답|지침|프롬프트|문맥|전략|정책|추론|생각)"),
    re.compile(r"(?:ユーザー|応答|指示|プロンプト|文脈|戦略|方針|推論|思考)"),
)

_PRIVATE_REASONING_BLOCK_RE = re.compile(
    r"<([A-Za-z][A-Za-z0-9_:-]{1,40})(?:\s+[^>\n]{0,120})?>([\s\S]*?)</\1>",
    re.IGNORECASE,
)
_PRIVATE_REASONING_FENCE_RE = re.compile(
    r"```([A-Za-z0-9_:-]{1,40})\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)
_PRIVATE_REASONING_UNCLOSED_RE = re.compile(
    r"<([A-Za-z][A-Za-z0-9_:-]{1,40})(?:\s+[^>\n]{0,120})?>[\s\S]*$",
    re.IGNORECASE,
)
_PROMPT_TEMPLATE_FENCE_RE = re.compile(
    r"```([A-Za-z0-9_:-]{0,40})\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

_PROMPT_TEMPLATE_KEYWORDS = (
    "response template",
    "template guidelines",
    "current_input",
    "current input",
    "thought process",
    "response must follow",
    "content structure must follow",
    "approved",
)

_PROMPT_TEMPLATE_PLACEHOLDER_RE = re.compile(
    r"\{\{[^{}]{1,120}\}\}|\{(?:content|thought\s*process|checklist|response|chapter(?:\s+title)?|number|title)\}",
    re.IGNORECASE,
)

_PROMPT_TEMPLATE_LINE_RES = (
    re.compile(r"^```"),
    re.compile(r"^<\/?\s*thoughts\s*>$", re.IGNORECASE),
    re.compile(r"^\{(?:content|thought\s*process|checklist|response|chapter(?:\s+title)?|number|title)\}$", re.IGNORECASE),
    re.compile(r"^#+\s*(?:response template|template guidelines|approved|response|응답|応答)\b", re.IGNORECASE),
    re.compile(r"^(?:response must follow|content structure must follow)", re.IGNORECASE),
)

_ASSISTANT_PREFILL_STRONG_RES = (
    re.compile(r"\bcontinue\s+in-?scene\b", re.IGNORECASE),
    re.compile(r"\brespond\s+to\s+the\s+current\s+turn\b", re.IGNORECASE),
    re.compile(r"\blet\s+me\s+begin\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:will|shall)\s+(?:continue|craft|write|weave|begin|start)\b", re.IGNORECASE),
    re.compile(r"(?:현재\s*턴|지금부터).*(?:응답|이야기|서사|장면)"),
    re.compile(r"(?:이어가|시작하|만들어\s*보|써\s*보).*(?:이야기|서사|장면|응답|답변)"),
    re.compile(r"(?:続け|始め|書き).*(?:物語|シーン|応答)"),
)

_ASSISTANT_PREFILL_AUTHORIAL_RES = (
    re.compile(r"\b(?:i|we)\b", re.IGNORECASE),
    re.compile(r"(?:나|내가|우린|우리가|저는|제가)"),
    re.compile(r"(?:私|僕|俺|わたし)"),
)

_ASSISTANT_PREFILL_ADDRESSEE_RES = (
    re.compile(r"\b(?:you|user|client|reader)\b", re.IGNORECASE),
    re.compile(r"(?:너|당신|유저|사용자|의뢰인|독자)"),
    re.compile(r"(?:あなた|ユーザー|読者)"),
)

_ASSISTANT_PREFILL_META_RES = (
    re.compile(r"\b(?:scene|narrative|story|chapter|response|reply|roleplay|continuity|tone|summary|session|meta commentary)\b", re.IGNORECASE),
    re.compile(r"(?:장면|서사|이야기|챕터|응답|답변|역할극|연속성|톤|요약|세션|메타)"),
    re.compile(r"(?:シーン|物語|章|応答|返信|ロールプレイ|連続性|トーン|要約|セッション|メタ)"),
)

_ASSISTANT_PREFILL_ACTION_RES = (
    re.compile(r"\b(?:continue|craft|write|weave|begin|start|respond|reply|create|preserve)\b", re.IGNORECASE),
    re.compile(r"(?:이어가|작성하|써\s*내|만들|시작하|응답하|답변하|보존하|지키)"),
    re.compile(r"(?:続け|書き|描き|始め|応答|返答|作り|保ち)"),
)


def _normalize_reasoning_label(text: str) -> str:
    return re.sub(r"[\s_-]+", "", str(text or "").strip().lower())


def is_private_reasoning_envelope_name(tag_name: str) -> bool:
    normalized = _normalize_reasoning_label(tag_name)
    if not normalized:
        return False
    if normalized in {"filter", "__filter_complete__"} or "filtercomplete" in normalized:
        return True
    return any(hint in normalized for hint in _PRIVATE_REASONING_NAME_HINTS)


def looks_like_private_reasoning_body(text: str) -> bool:
    raw = str(text or "").strip()
    if len(raw) < 24:
        return False

    cue_count = sum(1 for cue in _PRIVATE_REASONING_BODY_CUES if cue.search(raw))
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    bullet_like_count = sum(1 for line in lines if re.match(r"(?:[-*•]\s+|\d+\.\s+)", line))
    if bullet_like_count >= 2:
        cue_count += 1

    return cue_count >= 2


def looks_like_prompt_template_scaffold(text: str) -> bool:
    raw = str(text or "").strip()
    if len(raw) < 16:
        return False

    cue_count = 0
    lowered = raw.lower()
    if any(keyword in lowered for keyword in _PROMPT_TEMPLATE_KEYWORDS):
        cue_count += 1
    if _PROMPT_TEMPLATE_PLACEHOLDER_RE.search(raw):
        cue_count += 1
    if re.search(r"^\s*#\s*(?:response|응답|応答|approved)\b", raw, re.IGNORECASE | re.MULTILINE):
        cue_count += 1
    if re.search(r"<\s*thoughts\s*>|<\/\s*thoughts\s*>", raw, re.IGNORECASE):
        cue_count += 1
    if raw.count("```") >= 2:
        cue_count += 1

    return cue_count >= 2


def _is_prompt_template_scaffold_line(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in _PROMPT_TEMPLATE_LINE_RES)


def looks_like_assistant_prefill_paragraph(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw or len(raw) > 520:
        return False

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines or len(lines) > 5:
        return False
    if any(len(line) > 180 for line in lines):
        return False

    if any(pattern.search(raw) for pattern in _ASSISTANT_PREFILL_STRONG_RES):
        return True

    authorial = any(pattern.search(raw) for pattern in _ASSISTANT_PREFILL_AUTHORIAL_RES)
    addressee = any(pattern.search(raw) for pattern in _ASSISTANT_PREFILL_ADDRESSEE_RES)
    meta = any(pattern.search(raw) for pattern in _ASSISTANT_PREFILL_META_RES)
    action = any(pattern.search(raw) for pattern in _ASSISTANT_PREFILL_ACTION_RES)
    multi_line = len(lines) >= 2

    return multi_line and authorial and action and (meta or addressee)


def strip_leading_assistant_prefill_paragraphs(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""

    steps = 0
    while clean and steps < 3:
        steps += 1
        parts = re.split(r"\n\s*\n", clean, maxsplit=1)
        head = parts[0].strip()
        if not looks_like_assistant_prefill_paragraph(head):
            break
        clean = parts[1].lstrip() if len(parts) > 1 else ""

    return clean


def strip_prompt_template_sections(text: str) -> str:
    clean = str(text or "")
    if not clean.strip():
        return ""

    clean = _PROMPT_TEMPLATE_FENCE_RE.sub(
        lambda match: ""
        if looks_like_prompt_template_scaffold(
            f"{match.group(1).strip()}\n{match.group(2)}" if match.group(1) else match.group(2)
        )
        else match.group(0),
        clean,
    )
    return clean


def strip_hidden_reasoning_sections(text: str) -> str:
    clean = str(text or "")
    if not clean.strip():
        return ""

    clean = clean.replace("\r\n", "\n")
    previous = None
    steps = 0
    while clean != previous and steps < 8:
        previous = clean
        steps += 1
        clean = _PRIVATE_REASONING_BLOCK_RE.sub(
            lambda match: ""
            if is_private_reasoning_envelope_name(match.group(1))
            or looks_like_private_reasoning_body(match.group(2))
            or looks_like_prompt_template_scaffold(match.group(2))
            else match.group(0),
            clean,
        )
        clean = _PRIVATE_REASONING_FENCE_RE.sub(
            lambda match: ""
            if is_private_reasoning_envelope_name(match.group(1))
            or looks_like_private_reasoning_body(match.group(2))
            or looks_like_prompt_template_scaffold(f"{match.group(1).strip()}\n{match.group(2)}" if match.group(1) else match.group(2))
            else match.group(0),
            clean,
        )
        clean = _PRIVATE_REASONING_UNCLOSED_RE.sub(
            lambda match: ""
            if is_private_reasoning_envelope_name(match.group(1))
            or looks_like_private_reasoning_body(re.sub(r"^<[^>\n]+>", "", match.group(0), count=1))
            or looks_like_prompt_template_scaffold(re.sub(r"^<[^>\n]+>", "", match.group(0), count=1))
            else match.group(0),
            clean,
        )
        clean = re.sub(r"<__filter_complete__\s*/?>", "", clean, flags=re.IGNORECASE)

    clean = strip_prompt_template_sections(clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def sanitize_assistant_chatlog_text(text: str) -> str:
    return strip_hidden_reasoning_sections(text)


def sanitize_text_for_critic(text: str) -> str:
    raw = strip_hidden_reasoning_sections(text)
    if not raw.strip():
        return ""
    raw = strip_leading_assistant_prefill_paragraphs(raw)
    if not raw.strip():
        return ""

    sanitized_lines: list[str] = []
    skipping_control_section = False
    for line in raw.splitlines():
        stripped = line.strip()

        if skipping_control_section:
            if not stripped:
                skipping_control_section = False
                continue
            if looks_like_critic_section_header(line):
                skipping_control_section = False
            else:
                continue

        if is_critic_control_section_header(line):
            skipping_control_section = True
            continue

        if _is_critic_control_marker_line(line):
            continue
        if _is_prompt_template_scaffold_line(line):
            continue

        sanitized_lines.append(line)

    cleaned = "\n".join(sanitized_lines).strip()
    if looks_like_prompt_template_scaffold(cleaned):
        return ""
    return cleaned


def sanitize_context_messages_for_critic(messages: list[dict] | None) -> list[dict]:
    sanitized: list[dict] = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        content = sanitize_text_for_critic(item.get("content", ""))
        if not content:
            continue
        copied = dict(item)
        copied["content"] = content
        sanitized.append(copied)
    return sanitized


def sanitize_inline_text_for_critic(text: str) -> str:
    raw = strip_hidden_reasoning_sections(text)
    if not raw.strip():
        return ""
    raw = strip_leading_assistant_prefill_paragraphs(raw)
    if not raw.strip():
        return ""

    kept_lines: list[str] = []
    for line in raw.splitlines():
        if is_critic_control_section_header(line):
            continue
        if _is_critic_control_marker_line(line):
            continue
        if _is_prompt_template_scaffold_line(line):
            continue
        stripped = line.strip()
        if stripped:
            kept_lines.append(stripped)

    cleaned = "\n".join(kept_lines).strip()
    if looks_like_prompt_template_scaffold(cleaned):
        return ""
    return cleaned


def _normalize_excerpt_for_match(text: str) -> str:
    collapsed = " ".join(str(text or "").strip().lower().split())
    return re.sub(r"[\"'`“”‘’]+", "", collapsed)


def excerpt_has_turn_grounding(excerpt: str, turn_content: str) -> bool:
    excerpt_norm = _normalize_excerpt_for_match(excerpt)
    turn_norm = _normalize_excerpt_for_match(turn_content)
    if len(excerpt_norm) < 4 or not turn_norm:
        return False
    return excerpt_norm in turn_norm


def sanitize_evidence_excerpt(excerpt: str, turn_content: str) -> str:
    cleaned = sanitize_text_for_critic(excerpt)
    if not cleaned:
        return ""
    if not excerpt_has_turn_grounding(cleaned, turn_content):
        return ""
    return cleaned[:500]