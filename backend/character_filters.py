import json
import re


_GENERIC_DESCRIPTOR_HUMAN_TOKENS = frozenset(
    {
        "woman",
        "man",
        "girl",
        "boy",
        "lady",
        "gentleman",
        "stranger",
        "figure",
        "person",
        "여자",
        "남자",
        "소녀",
        "소년",
        "여인",
        "사내",
        "남성",
        "여성",
        "숙녀",
        "아가씨",
        "노인",
        "아이",
    }
)
_DESCRIPTOR_STOPWORDS = frozenset({"a", "an", "the", "this", "that", "these", "those", "in", "on", "at", "of"})
_KOREAN_PARTICLE_SUFFIXES = (
    "으로",
    "에서",
    "에게",
    "한테",
    "와",
    "과",
    "랑",
    "의",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "도",
    "만",
)
_DESCRIPTOR_TOKEN_SPLIT_RE = re.compile(r"[\s\.,!?:;()\[\]{}/|\\\"'`~@#$%^&*+=<>\-\u2014\u2013]+")
_ANCHOR_EVENT_TYPES = frozenset({"relationship_shift", "personality_change"})


def _contains_korean(text: str) -> bool:
    return any("\uAC00" <= ch <= "\uD7A3" or "\u1100" <= ch <= "\u11FF" for ch in str(text or ""))


def normalize_character_descriptor_keyword(value: object) -> str:
    token = " ".join(str(value or "").split()).strip().casefold()
    if not token:
        return ""
    token = re.sub(r"^(?:a|an|the)\s+", "", token)
    token = re.sub(r"'s$", "", token)
    for suffix in _KOREAN_PARTICLE_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix):
            token = token[: -len(suffix)]
            break
    return token.strip()


def extract_character_descriptor_keywords(name: object) -> list[str]:
    text = " ".join(str(name or "").split()).strip()
    if not text:
        return []

    seen: set[str] = set()
    keywords: list[str] = []
    for raw_token in _DESCRIPTOR_TOKEN_SPLIT_RE.split(text.casefold()):
        token = normalize_character_descriptor_keyword(raw_token)
        if not token or token in _DESCRIPTOR_STOPWORDS:
            continue
        if _contains_korean(token):
            if len(token) < 1:
                continue
        elif len(token) < 3:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def looks_like_transient_descriptor_character_name(name: object) -> bool:
    keywords = extract_character_descriptor_keywords(name)
    if not keywords:
        return False
    has_generic_tail = any(token in _GENERIC_DESCRIPTOR_HUMAN_TOKENS for token in keywords)
    if not has_generic_tail:
        return False
    non_generic = [token for token in keywords if token not in _GENERIC_DESCRIPTOR_HUMAN_TOKENS]
    return bool(non_generic) or len(keywords) == 1


def _load_structured_payload(raw: object):
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw).strip()
    if not text or text in {"null", "{}", "[]"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _has_meaningful_payload(raw: object) -> bool:
    payload = _load_structured_payload(raw)
    if payload is None:
        return False
    if isinstance(payload, dict):
        return any(_has_meaningful_payload(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_has_meaningful_payload(value) for value in payload)
    return bool(str(payload).strip())


def has_character_continuity_anchor(
    *,
    appearance_json: object = None,
    personality_json: object = None,
    relationships_json: object = None,
    speech_style_json: object = None,
    events: object = None,
) -> bool:
    if _has_meaningful_payload(appearance_json):
        return True
    if _has_meaningful_payload(personality_json):
        return True
    if _has_meaningful_payload(relationships_json):
        return True
    if _has_meaningful_payload(speech_style_json):
        return True
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = str(event.get("type") or "").strip()
            if event_type in _ANCHOR_EVENT_TYPES:
                return True
    return False


def descriptor_recently_rementioned(
    name: object,
    *,
    recent_text: object,
    recent_keywords: set[str] | list[str] | tuple[str, ...],
) -> bool:
    raw_name = " ".join(str(name or "").split()).strip().casefold()
    raw_recent_text = " ".join(str(recent_text or "").split()).strip().casefold()
    if raw_name and raw_name in raw_recent_text:
        return True

    name_keywords = extract_character_descriptor_keywords(name)
    recent_keyword_set = {
        normalized
        for normalized in (
            normalize_character_descriptor_keyword(keyword)
            for keyword in (recent_keywords or [])
        )
        if normalized
    }
    generic_hit = bool(_GENERIC_DESCRIPTOR_HUMAN_TOKENS & recent_keyword_set)
    if not generic_hit and raw_recent_text:
        generic_hit = any(token in raw_recent_text for token in _GENERIC_DESCRIPTOR_HUMAN_TOKENS)
    if not generic_hit:
        return False

    non_generic = [token for token in name_keywords if token not in _GENERIC_DESCRIPTOR_HUMAN_TOKENS]
    if not non_generic:
        return True

    overlap = set(non_generic) & recent_keyword_set
    required_overlap = 1 if len(non_generic) == 1 else min(2, len(non_generic))
    return len(overlap) >= required_overlap