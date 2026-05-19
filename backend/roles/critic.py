import base64
import json
import hashlib
import logging
import re
from datetime import datetime
from time import perf_counter
from urllib.parse import urlencode

import httpx

from backend.archive.store import ArchiveStoreBridge
from backend.character_filters import has_character_continuity_anchor, looks_like_transient_descriptor_character_name
from backend.critic_sanitizer import sanitize_context_messages_for_critic, sanitize_evidence_excerpt, sanitize_inline_text_for_critic, sanitize_text_for_critic
from backend.direct_evidence import is_within_retention_window, normalize_archive_state, normalize_capture_verification, resolve_committed_gate
from backend.integrity.consistency_checker import ConsistencyChecker
from backend.palace.bridge import save_state_deltas, append_audit_log
from backend.database import SessionLocal
from backend.models import Memory, DirectEvidenceRecord, CharacterState, CharacterEvent, PendingThread
from backend.prompt_store import read_prompt
from backend.settings import settings

logger = logging.getLogger(__name__)

_PROMPT_CONTROL_TITLE_MARKERS = (
    "unresolved threads",
    "open threads",
    "pending threads",
    "continuity hooks",
    "director notes",
    "narrative guidance",
    "narrative control",
    "scene goals",
    "prompt guidance",
    "toggle expansion",
    "response template",
    "reasoning guideline",
    "archive label",
    "current input",
    "long-term memory archive",
    "system prompt",
    "미해결 스레드",
    "열린 스레드",
    "연속성 훅",
)

_EA1J_PIPELINE_VERSION = "ea1j.v1"
_EA1J_STAGE_EXTRACTOR = "evidence_extractor"
_EA1J_STAGE_REDUCER = "deterministic_reducer"
_EA1J_STAGE_COMPACTOR = "summary_compactor_background"
_EA1K_PREVIEW_VERSION = "ea1k.v1"
_EA1K_PREVIEW_MAX_RAW_ITEMS = 3
_EA1K_PREVIEW_MAX_DIRECT_ITEMS = 3
_EA1K_CORRECTION_SIGNAL_KEYWORDS = (
    "actually",
    "correction",
    "wrong",
    "not true",
    "정정",
    "틀렸",
    "아니",
    "사실은",
)
_EA1K_LONG_HORIZON_SIGNAL_KEYWORDS = (
    "before",
    "previously",
    "weeks",
    "months",
    "years",
    "지난",
    "오래",
    "며칠",
    "몇 주",
)

_OOC_PREFIX_PATTERN = re.compile(
    r"^\s*(?:/ooc\b|ooc\b\s*[:\-]|out\s+of\s+character\b|#{1,6}\s*(?:ooc|out\s+of\s+character)\b|\[\s*ooc\s*\]|\[\[\s*ooc\s*\]\]|\(\s*ooc\s*\)|\(\(\s*ooc\s*\)\)|오오씨\b)",
    flags=re.IGNORECASE,
)
_RISU_USER_PLACEHOLDER_PATTERN_TEXT = r"(?:\{\{\s*user\s*\}\}|<\s*user\s*>)"
_RISU_CHAR_PLACEHOLDER_PATTERN_TEXT = r"(?:\{\{\s*char(?:=[^}]*)?\s*\}\}|<\s*char(?:=[^>]*)?\s*>)"
_TURN_USER_PREFIX_PATTERN = re.compile(
    rf"^\s*(?:유저|user|{_RISU_USER_PLACEHOLDER_PATTERN_TEXT})\s*:\s*(.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_TURN_ASSISTANT_PREFIX_PATTERN = re.compile(
    rf"^\s*(?:assistant|어시스턴트|{_RISU_CHAR_PLACEHOLDER_PATTERN_TEXT})\s*:\s*(.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)
_PLAYER_ENTITY_TOKEN = "__PLAYER__"


_PLAYER_REFERENCE_LABELS = (
    _PLAYER_ENTITY_TOKEN,
    "{{user}}",
    "<user>",
    "player",
    "user",
    "participant",
    "conversation partner",
    "chat partner",
    "counterpart",
    "interlocutor",
    "플레이어",
    "유저",
    "사용자",
    "대화 상대",
    "상대방",
    "ユーザー",
    "プレイヤー",
    "参加者",
)


def _build_reference_label_pattern(labels: tuple[str, ...]) -> str:
    patterns: list[str] = []
    unique_labels = sorted(
        {str(label or "").strip() for label in labels if str(label or "").strip()},
        key=len,
        reverse=True,
    )
    for label in unique_labels:
        escaped = re.escape(label)
        if re.search(r"[A-Za-z0-9_]", label):
            patterns.append(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        else:
            patterns.append(escaped)
    return "|".join(patterns)


_PLAYER_REFERENCE_PATTERN_TEXT = _build_reference_label_pattern(_PLAYER_REFERENCE_LABELS)
_PLAYER_REFERENCE_NOUN_PATTERN = re.compile(
    rf"(?:{_PLAYER_REFERENCE_PATTERN_TEXT})(?:님|씨)?(?:\s*(?:은|는|이|가|을|를|의|에게|한테|께|와|과|랑|으로|로|도|만))?",
    flags=re.IGNORECASE,
)
_RISU_USER_PLACEHOLDER_FULL_PATTERN = re.compile(
    rf"^\s*{_RISU_USER_PLACEHOLDER_PATTERN_TEXT}\s*$",
    flags=re.IGNORECASE,
)
_RISU_CHAR_PLACEHOLDER_FULL_PATTERN = re.compile(
    rf"^\s*{_RISU_CHAR_PLACEHOLDER_PATTERN_TEXT}\s*$",
    flags=re.IGNORECASE,
)
_RISU_INLINE_PARTICIPANT_PATTERN = re.compile(
    rf"(?:{_RISU_USER_PLACEHOLDER_PATTERN_TEXT}|{_RISU_CHAR_PLACEHOLDER_PATTERN_TEXT})",
    flags=re.IGNORECASE,
)
_RELATIONSHIP_PAIR_SEPARATORS = ("->", "→", "=>", "/", "|", ",", "&", "-")
_ROLE_SPEAKER_PREFIX_PATTERN = re.compile(
    rf"^\s*(?:유저|user|assistant|어시스턴트|system|시스템|developer|개발자|{_RISU_USER_PLACEHOLDER_PATTERN_TEXT}|{_RISU_CHAR_PLACEHOLDER_PATTERN_TEXT})\s*:\s*",
    flags=re.IGNORECASE,
)
_PARTICIPANT_ENTITY_SLOT_KEYS = frozenset(
    {
        "subject",
        "object",
        "name",
        "owner",
        "target",
        "from",
        "to",
        "left_entity",
        "right_entity",
    }
)
_PARTICIPANT_DERIVED_RELATION_KEYS = frozenset({"pair", "pair_key"})
_PARTICIPANT_ARTIFACT_DROP = object()


def _looks_like_prompt_control_title(title: str) -> bool:
    normalized = " ".join(str(title or "").strip().lower().split())
    if not normalized:
        return False
    if normalized.startswith("━━") and normalized.endswith("━━"):
        return True
    return any(marker in normalized for marker in _PROMPT_CONTROL_TITLE_MARKERS)

# ── 한글/카타카나 로마자 정규화 (plugin.js normalizeEntityName과 동일 알고리즘) ──
_CHO  = ["g","kk","n","d","tt","r","m","b","pp","s","ss","","j","jj","ch","k","t","p","h"]
_JUNG = ["a","ae","ya","yae","eo","e","yeo","ye","o","wa","wae","oe","yo","u","wo","we","wi","yu","eu","ui","i"]
_JONG = ["","k","k","ks","n","nj","nh","t","l","lk","lm","lb","ls","lt","lp","lh","m","p","ps","t","t","ng","t","t","k","t","p","h"]


def _romanize_korean(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            idx = code - 0xAC00
            out.append(_CHO[idx // 588])
            out.append(_JUNG[(idx % 588) // 28])
            out.append(_JONG[idx % 28])
        else:
            out.append(ch.lower())
    return re.sub(r'[^a-z0-9]', '', ''.join(out))


def _normalize_entity_name(name: str) -> str:
    """엔티티 이름을 정규화 키로 변환. 한글이면 로마자화, 영어면 소문자+특수문자 제거."""
    s = (name or "").strip()
    if not s:
        return ""
    if re.search(r'[가-힣]', s):
        rom = _romanize_korean(s)
        return rom if rom else re.sub(r'[^a-z0-9가-힣]', '', s.lower())
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]', '', s.lower())


def _normalize_actor_reference_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = _normalize_entity_name(text)
    if normalized:
        return normalized
    return re.sub(r"[^a-z0-9가-힣\u3040-\u30ff\u4e00-\u9fff]", "", text.casefold())


_PLAYER_REFERENCE_KEYS = frozenset(
    _normalize_actor_reference_key(label)
    for label in _PLAYER_REFERENCE_LABELS
)

_NON_ENTITY_ROLE_KEYS = frozenset(
    _normalize_actor_reference_key(label)
    for label in (
        "assistant",
        "system",
        "developer",
        "prompt",
        "instruction",
        "bot",
        "agent",
        "어시스턴트",
        "시스템",
        "개발자",
        "프롬프트",
        "지시문",
        "봇",
        "아바타",
        "アシスタント",
        "システム",
        "開発者",
        "プロンプト",
        "ボット",
    )
)


def _detect_chat_provider_from_endpoint(endpoint: str) -> str:
    normalized = str(endpoint or "").strip().lower()
    if "aiplatform.googleapis.com" in normalized:
        return "vertex"
    if "generativelanguage.googleapis.com" in normalized:
        return "gemini"
    return "openai"


_KNOWN_CHAT_PROVIDERS = frozenset({"openai", "claude", "gemini", "vertex", "openrouter", "copilot", "custom"})


def _normalize_chat_provider(value: object) -> str:
    provider = str(value or "").strip().lower()
    return provider if provider in _KNOWN_CHAT_PROVIDERS else ""


def _normalize_google_generate_content_url(endpoint: str, model: str, *, is_vertex: bool) -> str:
    model_text = str(model or "").strip()
    base_url = str(endpoint or "").strip()
    if not base_url and not is_vertex:
        base_url = "https://generativelanguage.googleapis.com/v1beta"
    base_url = base_url.rstrip("/")

    if is_vertex:
        if ":streamGenerateContent" in base_url:
            return base_url.replace(":streamGenerateContent", ":generateContent")
        if re.search(r":generateContent$", base_url, re.IGNORECASE):
            return base_url
        if re.search(r"/models/[^/:]+$", base_url, re.IGNORECASE):
            return f"{base_url}:generateContent"
        return f"{base_url}/{model_text}:generateContent"

    if not re.search(r"/v[0-9][^/]*$", base_url, re.IGNORECASE) and not re.search(r"/v[0-9][^/]*/models/", base_url, re.IGNORECASE):
        base_url += "/v1beta"
    if re.search(r":generateContent$", base_url, re.IGNORECASE):
        return base_url
    if re.search(r"/models/[^/:]+$", base_url, re.IGNORECASE):
        return f"{base_url}:generateContent"
    if "/models/" in base_url:
        return base_url
    return f"{base_url}/models/{model_text}:generateContent"


def _extract_google_message_text(data: dict | None) -> str | None:
    candidates = (data or {}).get("candidates") or []
    if not candidates:
        return None
    parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
    if not isinstance(parts, list):
        return None
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("thought"):
            continue
        text = str(part.get("text") or "").strip()
        if text:
            chunks.append(text)
    combined = "\n\n".join(chunks).strip()
    return combined or None


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _get_vertex_access_token(client: httpx.Client, api_key: str) -> str:
    try:
        credentials = json.loads(str(api_key or "").strip())
    except Exception as exc:
        raise ValueError("Vertex AI Key must be a JSON service account credential.") from exc
    client_email = str((credentials or {}).get("client_email") or "").strip()
    private_key = str((credentials or {}).get("private_key") or "")
    if not client_email or not private_key:
        raise ValueError("Vertex AI credentials missing client_email or private_key")

    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except Exception as exc:
        raise RuntimeError("cryptography 패키지가 필요합니다: pip install cryptography") from exc

    now = int(datetime.utcnow().timestamp())
    header = {"alg": "RS256", "typ": "JWT"}
    claim_set = {
        "iss": client_email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": "https://oauth2.googleapis.com/token",
        "exp": now + 3600,
        "iat": now,
    }
    signing_input = f"{_base64url(json.dumps(header, separators=(',', ':')).encode('utf-8'))}.{_base64url(json.dumps(claim_set, separators=(',', ':')).encode('utf-8'))}"
    private_key_obj = serialization.load_pem_private_key(private_key.encode("utf-8"), password=None)
    signature = private_key_obj.sign(signing_input.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
    assertion = f"{signing_input}.{_base64url(signature)}"
    response = client.post(
        "https://oauth2.googleapis.com/token",
        content=urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        }),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    response.raise_for_status()
    payload = response.json()
    token = str((payload or {}).get("access_token") or "").strip()
    if not token:
        raise RuntimeError("No access token in Vertex AI token response")
    return token


def _normalize_actor_reference(value: object, *, allow_player_token: bool) -> str:
    text = sanitize_inline_text_for_critic(str(value or ""))
    if not text:
        return ""

    if _RISU_USER_PLACEHOLDER_FULL_PATTERN.match(text):
        return _PLAYER_ENTITY_TOKEN if allow_player_token else ""
    if _RISU_CHAR_PLACEHOLDER_FULL_PATTERN.match(text):
        return ""

    key = _normalize_actor_reference_key(text)
    if not key:
        return ""
    if key in _PLAYER_REFERENCE_KEYS:
        return _PLAYER_ENTITY_TOKEN if allow_player_token else ""
    if key in _NON_ENTITY_ROLE_KEYS:
        return ""
    return text


def _sanitize_participant_text(value: object) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""

    clean = _ROLE_SPEAKER_PREFIX_PATTERN.sub("", clean)
    clean = _RISU_INLINE_PARTICIPANT_PATTERN.sub(" ", clean)
    clean = _PLAYER_REFERENCE_NOUN_PATTERN.sub(" ", clean)
    clean = re.sub(r"\(\s*\)", "", clean)
    clean = re.sub(r"\[\s*\]", "", clean)
    clean = re.sub(r"\s{2,}", " ", clean)
    clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)
    return clean.strip(" \t\r\n,:;/-")


def _is_participant_or_role_label(value: object) -> bool:
    if _RISU_USER_PLACEHOLDER_FULL_PATTERN.match(str(value or "").strip()):
        return True
    if _RISU_CHAR_PLACEHOLDER_FULL_PATTERN.match(str(value or "").strip()):
        return True
    key = _normalize_actor_reference_key(value)
    return bool(key and (key in _PLAYER_REFERENCE_KEYS or key in _NON_ENTITY_ROLE_KEYS))


def _parse_relationship_pair_text(value: object) -> tuple[str | None, str | None]:
    text = str(value or "").strip()
    if not text:
        return None, None

    for separator in _RELATIONSHIP_PAIR_SEPARATORS:
        if separator not in text:
            continue
        parts = [part.strip() for part in text.split(separator) if str(part or "").strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
    return None, None


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein 편집 거리."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _find_canonical_character(db, sid: str, name: str) -> "CharacterState | None":
    """
    세션에서 name과 동일하거나 교차-스크립트 매칭되는 CharacterState를 찾는다.

    매칭 우선순위:
    1. 정확한 이름 일치 (character_name == name)
    2. 정규화 키 완전 일치 (romanized key 동일)
    3. 한글↔영문 교차-스크립트: Levenshtein ≤ 2 AND 양쪽 정규화 키가 4자 이상
    """
    # 1. 정확한 이름 일치
    exact = (
        db.query(CharacterState)
        .filter(CharacterState.chat_session_id == sid, CharacterState.character_name == name)
        .first()
    )
    if exact:
        return exact

    # 2 & 3. 정규화 키 기반 조회
    norm_key = _normalize_entity_name(name)
    if not norm_key or len(norm_key) < 2:
        return None

    name_has_korean = bool(re.search(r'[가-힣]', name))
    all_chars = (
        db.query(CharacterState)
        .filter(CharacterState.chat_session_id == sid)
        .all()
    )

    best: "CharacterState | None" = None
    best_dist = 999

    for row in all_chars:
        row_key = _normalize_entity_name(row.character_name)
        if not row_key or len(row_key) < 2:
            continue

        # 정규화 키 완전 일치 (어떤 스크립트든)
        if row_key == norm_key:
            return row

        # 교차-스크립트 fuzzy match: 한글↔영문 조합일 때만 허용
        row_has_korean = bool(re.search(r'[가-힣]', row.character_name))
        if name_has_korean == row_has_korean:
            continue  # 같은 스크립트면 skip (exact key 매치 실패 = 다른 인물)

        min_len = min(len(norm_key), len(row_key))
        if min_len < 4:
            continue  # 너무 짧은 이름은 fuzzy 제외

        dist = _levenshtein(norm_key, row_key)
        if dist <= 2 and dist < best_dist:
            best = row
            best_dist = dist

    if best:
        logger.info(
            "_find_canonical_character: cross-script match '%s' → '%s' (dist=%d)",
            name, best.character_name, best_dist,
        )
    return best


class Critic:
    def __init__(self) -> None:
        self.endpoint = ""
        self.api_key = ""
        self.model = ""
        self.provider = ""
        self.timeout = 30
        self.prompt_name = "critic_system"
        self.bridge = ArchiveStoreBridge()
        self.refresh_runtime_config()

    def _resolve_runtime_value(self, primary_attr: str, fallback_attr: str) -> str:
        primary = getattr(settings, primary_attr, "") or ""
        if isinstance(primary, str) and primary.strip():
            return primary
        fallback = getattr(settings, fallback_attr, "") or ""
        return fallback if isinstance(fallback, str) else ""

    def refresh_runtime_config(self) -> None:
        self.provider = _normalize_chat_provider(self._resolve_runtime_value("PROJECT_CRITIC_PROVIDER", "PROJECT_MAIN_PROVIDER"))
        self.endpoint = self._resolve_runtime_value("PROJECT_CRITIC_ENDPOINT", "PROJECT_MAIN_ENDPOINT")
        self.api_key = self._resolve_runtime_value("PROJECT_CRITIC_API_KEY", "PROJECT_MAIN_API_KEY")
        self.model = self._resolve_runtime_value("PROJECT_CRITIC_MODEL", "PROJECT_MAIN_MODEL")
        self.timeout = getattr(settings, "CRITIC_TIMEOUT", 30) or 30

    def _effective_provider(self) -> str:
        return self.provider or _detect_chat_provider_from_endpoint(self.endpoint)

    def _build_ea1k_deterministic_preview_pass(
        self,
        *,
        turn_index: int,
        turn_content: str,
        context: list[dict],
        chat_session_id: str | None,
    ) -> dict:
        preview = {
            "policy_version": _EA1K_PREVIEW_VERSION,
            "status": "pending",
            "recent_raw_preview": [],
            "recent_verified_direct_evidence_seed": [],
            "triage": {
                "priority": "low",
                "critic_focus_recommended": False,
                "reason_codes": [],
            },
            "compaction_trigger": {
                "should_trigger": False,
                "reason_codes": [],
                "recommended_layers": [],
            },
            "warnings": [],
        }

        try:
            raw_preview = self._ea1k_collect_recent_raw_preview_items(
                turn_content=turn_content,
                context=context,
                max_items=_EA1K_PREVIEW_MAX_RAW_ITEMS,
            )
            direct_seed = self._ea1k_collect_recent_verified_direct_evidence_seed(
                chat_session_id=chat_session_id,
                turn_index=turn_index,
                max_items=_EA1K_PREVIEW_MAX_DIRECT_ITEMS,
            )
            triage = self._ea1k_classify_preview_triage(
                turn_content=turn_content,
                raw_preview=raw_preview,
                direct_seed=direct_seed,
            )
            compaction_trigger = self._ea1k_classify_compaction_trigger(
                turn_index=turn_index,
                turn_content=turn_content,
                raw_preview=raw_preview,
                direct_seed=direct_seed,
            )

            preview.update({
                "status": "ok",
                "recent_raw_preview": raw_preview,
                "recent_verified_direct_evidence_seed": direct_seed,
                "triage": triage,
                "compaction_trigger": compaction_trigger,
            })
            return preview
        except Exception as exc:
            preview["status"] = "error"
            preview["warnings"] = ["preview_generation_failed"]
            preview["error"] = str(exc)[:160]
            return preview

    def _ea1k_collect_recent_raw_preview_items(
        self,
        *,
        turn_content: str,
        context: list[dict],
        max_items: int,
    ) -> list[dict]:
        candidates: list[tuple[str, str]] = []
        if isinstance(turn_content, str) and turn_content.strip():
            candidates.append(("latest_turn", turn_content.strip()))

        if isinstance(context, list):
            for item in reversed(context[-6:]):
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "unknown").strip().lower() or "unknown"
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                candidates.append((f"context_{role}", content))

        preview_items: list[dict] = []
        seen: set[str] = set()
        for source, text in candidates:
            parts = re.split(r"[\n\.!?;]+", text)
            for part in parts:
                normalized = sanitize_inline_text_for_critic(part)
                if not normalized:
                    continue
                compact = " ".join(normalized.split()).strip()
                if len(compact) < 10:
                    continue
                dedupe_key = compact.lower()
                if dedupe_key in seen:
                    continue
                preview_items.append({
                    "source": source,
                    "text": compact[:180],
                })
                seen.add(dedupe_key)
                if len(preview_items) >= max_items:
                    return preview_items

        return preview_items

    def _ea1k_collect_recent_verified_direct_evidence_seed(
        self,
        *,
        chat_session_id: str | None,
        turn_index: int,
        max_items: int,
    ) -> list[dict]:
        session_id = (chat_session_id or "").strip() or "default"
        db = SessionLocal()
        try:
            query = (
                db.query(DirectEvidenceRecord)
                .filter(DirectEvidenceRecord.chat_session_id == session_id)
                .filter(DirectEvidenceRecord.capture_verification == "verified")
                .filter(DirectEvidenceRecord.committed_gate.in_(["finalize", "manual"]))
                .filter(DirectEvidenceRecord.tombstoned.is_(False))
                .filter(DirectEvidenceRecord.superseded_by_id.is_(None))
            )

            if isinstance(turn_index, int) and turn_index >= 0:
                query = query.filter(DirectEvidenceRecord.source_turn_end <= turn_index)

            rows = (
                query.order_by(
                    DirectEvidenceRecord.source_turn_end.desc(),
                    DirectEvidenceRecord.id.desc(),
                )
                .limit(max_items * 4)
                .all()
            )

            result: list[dict] = []
            seen: set[str] = set()
            for row in rows:
                if not is_within_retention_window(
                    source_turn_end=row.source_turn_end,
                    latest_turn_index=turn_index,
                    archive_state=row.archive_state,
                    capture_verification=row.capture_verification,
                    repair_needed=row.repair_needed,
                    committed_gate=row.committed_gate,
                    tombstoned=row.tombstoned,
                    lineage=row.lineage_json,
                ):
                    continue

                text = " ".join(str(row.evidence_text or "").split()).strip()
                if len(text) < 8:
                    continue
                dedupe_key = text.lower()
                if dedupe_key in seen:
                    continue
                result.append({
                    "record_id": int(row.id),
                    "source_turn_start": int(row.source_turn_start),
                    "source_turn_end": int(row.source_turn_end),
                    "evidence_text": text[:180],
                })
                seen.add(dedupe_key)
                if len(result) >= max_items:
                    break
            return result
        finally:
            db.close()

    def _ea1k_classify_preview_triage(
        self,
        *,
        turn_content: str,
        raw_preview: list[dict],
        direct_seed: list[dict],
    ) -> dict:
        scan_text = "\n".join([
            str(turn_content or ""),
            *[str(item.get("text") or "") for item in raw_preview],
        ]).lower()

        reason_codes: list[str] = []
        correction_signal = any(keyword in scan_text for keyword in _EA1K_CORRECTION_SIGNAL_KEYWORDS)
        if correction_signal:
            reason_codes.append("explicit_correction_signal")

        if len(direct_seed) > 0:
            reason_codes.append("recent_verified_direct_evidence")
        if len(raw_preview) >= 2:
            reason_codes.append("multi_raw_signal")

        if correction_signal or len(direct_seed) >= 2:
            priority = "high"
        elif len(direct_seed) >= 1 or len(raw_preview) >= 2:
            priority = "medium"
        else:
            priority = "low"

        return {
            "priority": priority,
            "critic_focus_recommended": priority != "low",
            "reason_codes": reason_codes,
        }

    def _ea1k_classify_compaction_trigger(
        self,
        *,
        turn_index: int,
        turn_content: str,
        raw_preview: list[dict],
        direct_seed: list[dict],
    ) -> dict:
        scan_text = "\n".join([
            str(turn_content or ""),
            *[str(item.get("text") or "") for item in raw_preview],
        ]).lower()

        reason_codes: list[str] = []
        if len(direct_seed) >= 3:
            reason_codes.append("verified_seed_density_high")
        if isinstance(turn_index, int) and turn_index > 0 and turn_index % 10 == 0:
            reason_codes.append("episode_boundary_hint")
        if any(keyword in scan_text for keyword in _EA1K_LONG_HORIZON_SIGNAL_KEYWORDS):
            reason_codes.append("long_horizon_signal")

        should_trigger = bool(reason_codes)
        recommended_layers: list[str] = []
        if should_trigger:
            recommended_layers.append("episode_seed")
            if len(direct_seed) >= 3 or "long_horizon_signal" in reason_codes:
                recommended_layers.append("chapter_hint")

        return {
            "should_trigger": should_trigger,
            "reason_codes": reason_codes,
            "recommended_layers": recommended_layers,
        }

    def _format_ea1k_preview_seed_for_extract(self, preview_pass: dict) -> str:
        if not isinstance(preview_pass, dict):
            return ""
        if str(preview_pass.get("status") or "") != "ok":
            return ""

        triage = preview_pass.get("triage") or {}
        compaction = preview_pass.get("compaction_trigger") or {}
        raw_preview = list(preview_pass.get("recent_raw_preview") or [])[:_EA1K_PREVIEW_MAX_RAW_ITEMS]
        direct_seed = list(preview_pass.get("recent_verified_direct_evidence_seed") or [])[:_EA1K_PREVIEW_MAX_DIRECT_ITEMS]

        lines = [
            "<Deterministic_Evidence_Preview>",
            f"policy_version: {preview_pass.get('policy_version') or _EA1K_PREVIEW_VERSION}",
            f"triage_priority: {triage.get('priority') or 'low'}",
            f"triage_reason_codes: {', '.join(list(triage.get('reason_codes') or [])) or 'none'}",
            f"compaction_trigger: {'true' if bool(compaction.get('should_trigger')) else 'false'}",
            f"compaction_reason_codes: {', '.join(list(compaction.get('reason_codes') or [])) or 'none'}",
            f"compaction_recommended_layers: {', '.join(list(compaction.get('recommended_layers') or [])) or 'none'}",
            "recent_raw_preview:",
        ]
        for item in raw_preview:
            source = str(item.get("source") or "unknown")
            text = str(item.get("text") or "").strip()
            if text:
                lines.append(f"- [{source}] {text}")
        if len(lines) == 8:
            lines.append("- none")

        lines.append("recent_verified_direct_evidence_seed:")
        if direct_seed:
            for item in direct_seed:
                rid = item.get("record_id")
                start = item.get("source_turn_start")
                end = item.get("source_turn_end")
                text = str(item.get("evidence_text") or "").strip()
                lines.append(f"- [record:{rid} turn:{start}-{end}] {text}")
        else:
            lines.append("- none")
        lines.append("</Deterministic_Evidence_Preview>")

        return "\n".join(lines)

    def _write_ea1k_preview_audit(self, *, chat_session_id: str | None, turn_index: int, preview_pass: dict) -> None:
        try:
            append_audit_log(
                event_type="critic_preview_trace",
                chat_session_id=chat_session_id or "default",
                summary=f"EA-1k deterministic preview trace at turn {turn_index}",
                details=preview_pass,
                source="critic",
            )
        except Exception as exc:
            logger.warning("Critic.run preview audit write failed (turn=%d): %s", turn_index, exc)

    def _new_ea1j_pipeline_trace(self, turn_index: int) -> dict:
        return {
            "pipeline_version": _EA1J_PIPELINE_VERSION,
            "turn_index": int(turn_index),
            "all_in_single_critic_call": False,
            "split_contract": {
                "extractor_stage": _EA1J_STAGE_EXTRACTOR,
                "reducer_stage": _EA1J_STAGE_REDUCER,
                "compactor_stage": _EA1J_STAGE_COMPACTOR,
            },
            "stages": {
                _EA1J_STAGE_EXTRACTOR: {
                    "status": "pending",
                    "owner": "critic._extract",
                },
                _EA1J_STAGE_REDUCER: {
                    "status": "pending",
                    "owner": "critic.run + backend deterministic writes",
                    "components": {},
                },
                _EA1J_STAGE_COMPACTOR: {
                    "status": "deferred_background",
                    "owner": "complete-turn background summary/maintenance path",
                    "reason": "dense summary and ledger compaction is intentionally decoupled from Critic extraction",
                },
            },
            "warnings": [],
        }

    def _mark_ea1j_reducer_component(
        self,
        pipeline_trace: dict,
        component_name: str,
        status: str,
        *,
        reason: str | None = None,
        details: dict | None = None,
    ) -> None:
        stages = pipeline_trace.get("stages") or {}
        reducer_stage = stages.get(_EA1J_STAGE_REDUCER) or {}
        components = reducer_stage.setdefault("components", {})

        payload = {"status": status}
        if reason:
            payload["reason"] = reason
        if isinstance(details, dict) and details:
            payload.update(details)

        components[component_name] = payload

        if status == "failed":
            warnings = pipeline_trace.setdefault("warnings", [])
            warnings.append(f"{component_name}:{reason or 'failed'}")

    def _finalize_ea1j_pipeline_trace(self, pipeline_trace: dict) -> dict:
        stages = pipeline_trace.get("stages") or {}
        extractor_stage = stages.get(_EA1J_STAGE_EXTRACTOR) or {}
        reducer_stage = stages.get(_EA1J_STAGE_REDUCER) or {}
        reducer_components = reducer_stage.get("components") or {}

        ok_count = 0
        failed_count = 0
        skipped_count = 0
        for value in reducer_components.values():
            status = str((value or {}).get("status") or "")
            if status == "ok":
                ok_count += 1
            elif status == "failed":
                failed_count += 1
            else:
                skipped_count += 1

        reducer_stage["component_counts"] = {
            "total": len(reducer_components),
            "ok": ok_count,
            "failed": failed_count,
            "skipped": skipped_count,
        }

        if failed_count > 0 and ok_count > 0:
            reducer_stage["status"] = "degraded"
        elif failed_count > 0:
            reducer_stage["status"] = "failed"
        elif ok_count > 0:
            reducer_stage["status"] = "ok"
        else:
            reducer_stage["status"] = "skipped"

        extractor_status = str(extractor_stage.get("status") or "pending")
        reducer_status = str(reducer_stage.get("status") or "pending")

        if extractor_status == "failed" or reducer_status in {"failed", "degraded"}:
            pipeline_trace["status"] = "degraded"
        elif extractor_status == "skipped" and reducer_status == "skipped":
            pipeline_trace["status"] = "skipped"
        elif extractor_status == "ok":
            pipeline_trace["status"] = "ok"
        else:
            pipeline_trace["status"] = "partial"

        return pipeline_trace

    def _write_ea1j_pipeline_audit(self, *, chat_session_id: str | None, turn_index: int, pipeline_trace: dict) -> None:
        try:
            append_audit_log(
                event_type="critic_pipeline_trace",
                chat_session_id=chat_session_id or "default",
                summary=f"EA-1j split pipeline trace at turn {turn_index}",
                details=pipeline_trace,
                source="critic",
            )
        except Exception as exc:
            logger.warning("Critic.run pipeline audit write failed (turn=%d): %s", turn_index, exc)

    def _looks_like_ooc_turn(self, text: str) -> bool:
        content = str(text or "").strip()
        if not content:
            return False
        if _OOC_PREFIX_PATTERN.match(content):
            return True
        return False

    def _should_skip_ingest_for_ooc(self, turn_content: str, context: list[dict]) -> bool:
        if self._looks_like_ooc_turn(turn_content):
            return True

        match = _TURN_USER_PREFIX_PATTERN.search(str(turn_content or ""))
        if match and self._looks_like_ooc_turn(match.group(1) or ""):
            return True

        match = _TURN_ASSISTANT_PREFIX_PATTERN.search(str(turn_content or ""))
        if match and self._looks_like_ooc_turn(match.group(1) or ""):
            return True

        if not isinstance(context, list):
            return False
        for item in reversed(context[-3:]):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role != "user":
                continue
            if self._looks_like_ooc_turn(item.get("content") or ""):
                return True
        return False

    def run(
        self,
        turn_index: int,
        turn_content: str,
        context: list[dict],
        chat_session_id: str | None = None,
        output_language_override: dict | None = None,
    ) -> None:
        logger.info("Critic.run started (turn=%d, session=%s)", turn_index, chat_session_id or "default")
        pipeline_trace = self._new_ea1j_pipeline_trace(turn_index)
        extractor_stage = (pipeline_trace.get("stages") or {}).get(_EA1J_STAGE_EXTRACTOR) or {}
        try:
            sanitized_turn_content = sanitize_text_for_critic(turn_content)
            sanitized_context = sanitize_context_messages_for_critic(context)
            if not sanitized_turn_content or not str(sanitized_turn_content).strip():
                extractor_stage["status"] = "skipped"
                extractor_stage["reason"] = "empty_turn_content"
                reducer_stage = (pipeline_trace.get("stages") or {}).get(_EA1J_STAGE_REDUCER) or {}
                reducer_stage["status"] = "skipped"
                reducer_stage["reason"] = "extractor_input_missing"
                logger.info("Critic.run skipped: empty turn_content (turn=%d)", turn_index)
                return

            if self._should_skip_ingest_for_ooc(sanitized_turn_content, sanitized_context):
                extractor_stage["status"] = "skipped"
                extractor_stage["reason"] = "ooc_turn_guard"
                reducer_stage = (pipeline_trace.get("stages") or {}).get(_EA1J_STAGE_REDUCER) or {}
                reducer_stage["status"] = "skipped"
                reducer_stage["reason"] = "ooc_turn_guard"
                logger.info("Critic.run skipped by OOC guard (turn=%d)", turn_index)
                try:
                    append_audit_log(
                        event_type="critic_ingest_trace",
                        chat_session_id=chat_session_id or "default",
                        summary=f"Critic skipped ingest by OOC guard at turn {turn_index}",
                        details={
                            "turn_index": turn_index,
                            "surface": "ooc_turn_guard",
                            "trace": {"applied": True},
                        },
                        source="critic",
                    )
                except Exception:
                    pass
                return

            preview_pass = self._build_ea1k_deterministic_preview_pass(
                turn_index=turn_index,
                turn_content=sanitized_turn_content,
                context=sanitized_context,
                chat_session_id=chat_session_id,
            )
            triage = preview_pass.get("triage") or {}
            compaction = preview_pass.get("compaction_trigger") or {}
            pipeline_trace["preview_pass"] = {
                "policy_version": preview_pass.get("policy_version") or _EA1K_PREVIEW_VERSION,
                "status": preview_pass.get("status") or "pending",
                "raw_item_count": len(preview_pass.get("recent_raw_preview") or []),
                "direct_seed_count": len(preview_pass.get("recent_verified_direct_evidence_seed") or []),
                "triage_priority": triage.get("priority") or "low",
                "compaction_trigger": bool(compaction.get("should_trigger")),
                "compaction_recommended_layers": list(compaction.get("recommended_layers") or []),
                "warnings": list(preview_pass.get("warnings") or []),
            }
            self._write_ea1k_preview_audit(
                chat_session_id=chat_session_id,
                turn_index=turn_index,
                preview_pass=preview_pass,
            )

            preview_seed_block = self._format_ea1k_preview_seed_for_extract(preview_pass)
            extract_turn_content = sanitized_turn_content
            if preview_seed_block:
                extract_turn_content = f"{sanitized_turn_content}\n\n{preview_seed_block}"

            extractor_stage["preview_policy_version"] = _EA1K_PREVIEW_VERSION
            extractor_stage["preview_seed_applied"] = bool(preview_seed_block)
            extractor_stage["preview_triage_priority"] = triage.get("priority") or "low"
            extractor_stage["preview_compaction_trigger"] = bool(compaction.get("should_trigger"))

            result = self._extract(
                turn_content=extract_turn_content,
                context=sanitized_context,
                output_language_override=output_language_override,
            )
            if result is None:
                extractor_stage["status"] = "failed"
                extractor_stage["reason"] = "no_json_extracted"
                reducer_stage = (pipeline_trace.get("stages") or {}).get(_EA1J_STAGE_REDUCER) or {}
                reducer_stage["status"] = "skipped"
                reducer_stage["reason"] = "extractor_output_missing"
                logger.info("Critic.run skipped: no JSON extracted (turn=%d)", turn_index)
                return

            result, participant_trace = self._sanitize_result_for_participant(result)

            extractor_stage["status"] = "ok"
            extractor_stage["output_counts"] = {
                "evidence_excerpts": len(result.get("evidence_excerpts") or []),
                "kg_triples": len(result.get("kg_triples") or []),
                "state_deltas": 1 if isinstance(result.get("state_deltas"), dict) and result.get("state_deltas") else 0,
                "character_deltas": len(result.get("character_deltas") or []),
                "pending_threads": len(result.get("pending_threads") or []),
                "prune_targets": len(result.get("prune_targets") or []),
            }

            logger.info(
                "Critic.run extracted JSON (turn=%d, importance=%s, emotional=%.2f, narrative=%.2f, evidence=%d)",
                turn_index,
                result.get("importance_score", 5),
                result.get("emotional_intensity") or 0.0,
                result.get("narrative_significance") or 0.0,
                len(result.get("evidence_excerpts") or []),
            )

            archive_hint = result.get("archive_hint")
            if not isinstance(archive_hint, dict):
                legacy_library_hint = result.get("library_hint")
                legacy_palace_hint = result.get("palace_hint")
                if isinstance(legacy_library_hint, dict):
                    archive_hint = legacy_library_hint
                elif isinstance(legacy_palace_hint, dict):
                    archive_hint = legacy_palace_hint
                else:
                    archive_hint = {}
            wing = None
            room = None
            if isinstance(archive_hint, dict):
                wing = archive_hint.get("wing")
                room = archive_hint.get("room")

            try:
                safe_result, memory_trace = self._sanitize_memory_summary_for_critic(result)
                self.bridge.save_memory(
                    turn_index=turn_index,
                    summary_json=safe_result,
                    wing=wing,
                    room=room,
                    chat_session_id=chat_session_id,
                )
                self._mark_ea1j_reducer_component(
                    pipeline_trace,
                    "memory_summary_write",
                    "ok",
                    details={
                        "sanitized_fields": int(memory_trace.get("sanitized_fields", 0) or 0),
                        "dropped_fields": int(memory_trace.get("dropped_fields", 0) or 0),
                        "dropped_evidence_excerpts": int(memory_trace.get("dropped_evidence_excerpts", 0) or 0),
                    },
                )
                logger.info("Critic.run memory saved (turn=%d)", turn_index)
                if (
                    int(memory_trace.get("sanitized_fields", 0) or 0) > 0
                    or int(memory_trace.get("dropped_fields", 0) or 0) > 0
                    or int(memory_trace.get("dropped_evidence_excerpts", 0) or 0) > 0
                ):
                    append_audit_log(
                        event_type="critic_ingest_trace",
                        chat_session_id=chat_session_id or "default",
                        summary=f"Critic memory summary guard sanitized items at turn {turn_index}",
                        details={
                            "turn_index": turn_index,
                            "surface": "memory_summary",
                            "trace": memory_trace,
                        },
                        source="critic",
                    )
            except Exception as exc:
                self._mark_ea1j_reducer_component(
                    pipeline_trace,
                    "memory_summary_write",
                    "failed",
                    reason="bridge_save_memory_failed",
                    details={"error": str(exc)[:160]},
                )
                logger.warning("Critic.run memory save failed (turn=%d): %s", turn_index, exc)

            evidence_excerpts = result.get("evidence_excerpts")
            if isinstance(evidence_excerpts, list) and evidence_excerpts:
                try:
                    ev_result = self._save_direct_evidence_records(
                        turn_index=turn_index,
                        turn_content=sanitized_turn_content,
                        evidence_excerpts=evidence_excerpts,
                        chat_session_id=chat_session_id,
                    )
                    logger.info(
                        "Critic.run direct evidence saved (turn=%d, inserted=%d, skipped=%d, trace=%s)",
                        turn_index,
                        ev_result.get("inserted", 0),
                        ev_result.get("skipped", 0),
                        ev_result.get("trace") or {},
                    )
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "direct_evidence_write",
                        "ok",
                        details={
                            "inserted": int(ev_result.get("inserted", 0) or 0),
                            "skipped": int(ev_result.get("skipped", 0) or 0),
                        },
                    )
                    if ev_result.get("skipped", 0) > 0:
                        append_audit_log(
                            event_type="critic_ingest_trace",
                            chat_session_id=chat_session_id or "default",
                            summary=f"Critic direct evidence guard skipped items at turn {turn_index}",
                            details={
                                "turn_index": turn_index,
                                "surface": "direct_evidence",
                                "trace": ev_result.get("trace") or {},
                            },
                            source="critic",
                        )
                except Exception as exc:
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "direct_evidence_write",
                        "failed",
                        reason="direct_evidence_save_failed",
                        details={"error": str(exc)[:160]},
                    )
                    logger.warning("Critic.run direct evidence save failed (turn=%d): %s", turn_index, exc)
            else:
                self._mark_ea1j_reducer_component(
                    pipeline_trace,
                    "direct_evidence_write",
                    "skipped",
                    reason="no_evidence_excerpts",
                )

            kg_triples = result.get("kg_triples")
            if isinstance(kg_triples, list) and kg_triples:
                try:
                    safe_kg_triples, kg_trace = self._sanitize_kg_triples_for_critic(kg_triples)
                    if safe_kg_triples:
                        self.bridge.save_kg_triples(safe_kg_triples, source_turn=turn_index, chat_session_id=chat_session_id)
                    logger.info(
                        "Critic.run kg saved (turn=%d, count=%d, trace=%s)",
                        turn_index,
                        len(safe_kg_triples),
                        kg_trace,
                    )
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "kg_triple_write",
                        "ok",
                        details={
                            "accepted": int(kg_trace.get("accepted", 0) or 0),
                            "skipped": int(kg_trace.get("skipped", 0) or 0),
                        },
                    )
                    if kg_trace.get("skipped", 0) > 0:
                        append_audit_log(
                            event_type="critic_ingest_trace",
                            chat_session_id=chat_session_id or "default",
                            summary=f"Critic kg guard skipped items at turn {turn_index}",
                            details={
                                "turn_index": turn_index,
                                "surface": "kg_triples",
                                "trace": kg_trace,
                            },
                            source="critic",
                        )
                except Exception as exc:
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "kg_triple_write",
                        "failed",
                        reason="kg_save_failed",
                        details={"error": str(exc)[:160]},
                    )
                    logger.warning("Critic.run kg save failed (turn=%d): %s", turn_index, exc)
            else:
                self._mark_ea1j_reducer_component(
                    pipeline_trace,
                    "kg_triple_write",
                    "skipped",
                    reason="no_kg_triples",
                )

            # Phase 2-2: state_deltas → active_states
            state_deltas = result.get("state_deltas")
            if isinstance(state_deltas, dict) and state_deltas:
                try:
                    sd_result = save_state_deltas(
                        chat_session_id=chat_session_id,
                        turn_index=turn_index,
                        state_deltas=state_deltas,
                    )
                    logger.info(
                        "Critic.run state_deltas processed (turn=%d, saved=%s, skipped=%s, errors=%s)",
                        turn_index,
                        sd_result.get("saved", []),
                        sd_result.get("skipped", []),
                        sd_result.get("errors", []),
                    )
                    canonical_upsert = sd_result.get("canonical_state_upsert") or {}
                    canonical_status = str(canonical_upsert.get("status") or "off")
                    canonical_component_status = "ok"
                    canonical_component_reason = None
                    if canonical_status in {"off", "empty"}:
                        canonical_component_status = "skipped"
                        canonical_component_reason = "canonical_upsert_not_applied"
                    elif canonical_status in {"error", "degraded"}:
                        canonical_component_status = "failed"
                        canonical_component_reason = "canonical_upsert_failed"

                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "canonical_state_upsert",
                        canonical_component_status,
                        reason=canonical_component_reason,
                        details={
                            "policy_version": canonical_upsert.get("policy_version"),
                            "status": canonical_status,
                            "upserted_count": int(canonical_upsert.get("upserted_count", 0) or 0),
                            "fallback_count": int(canonical_upsert.get("fallback_count", 0) or 0),
                            "empty_fallback_count": int(canonical_upsert.get("empty_fallback_count", 0) or 0),
                        },
                    )
                    if sd_result.get("errors"):
                        self._mark_ea1j_reducer_component(
                            pipeline_trace,
                            "active_state_reduce",
                            "failed",
                            reason="state_deltas_save_errors",
                            details={
                                "errors": list(sd_result.get("errors") or [])[:5],
                                "canonical_state_upsert_status": canonical_status,
                                "canonical_state_upsert_policy_version": canonical_upsert.get("policy_version"),
                            },
                        )
                    else:
                        self._mark_ea1j_reducer_component(
                            pipeline_trace,
                            "active_state_reduce",
                            "ok",
                            details={
                                "saved": list(sd_result.get("saved") or []),
                                "skipped": list(sd_result.get("skipped") or []),
                                "canonical_state_upsert_status": canonical_status,
                                "canonical_state_upsert_policy_version": canonical_upsert.get("policy_version"),
                            },
                        )
                except Exception as exc:
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "active_state_reduce",
                        "failed",
                        reason="state_deltas_save_failed",
                        details={"error": str(exc)[:160]},
                    )
                    logger.warning("Critic.run state_deltas save failed (turn=%d): %s", turn_index, exc)
            else:
                self._mark_ea1j_reducer_component(
                    pipeline_trace,
                    "active_state_reduce",
                    "skipped",
                    reason="no_state_deltas",
                )
                logger.info("Critic.run no state_deltas in output (turn=%d)", turn_index)

            # Phase E-2: character_deltas → character_states + character_events
            character_deltas = result.get("character_deltas")
            if isinstance(character_deltas, list) and character_deltas:
                try:
                    cd_result = self._process_character_deltas(
                        chat_session_id=chat_session_id,
                        turn_index=turn_index,
                        character_deltas=character_deltas,
                    )
                    logger.info(
                        "Critic.run character_deltas processed (turn=%d, upserted=%d, events=%d)",
                        turn_index,
                        cd_result.get("upserted", 0),
                        cd_result.get("events_created", 0),
                    )
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "character_delta_upsert",
                        "ok",
                        details={
                            "upserted": int(cd_result.get("upserted", 0) or 0),
                            "events_created": int(cd_result.get("events_created", 0) or 0),
                        },
                    )
                except Exception as exc:
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "character_delta_upsert",
                        "failed",
                        reason="character_deltas_save_failed",
                        details={"error": str(exc)[:160]},
                    )
                    logger.warning("Critic.run character_deltas save failed (turn=%d): %s", turn_index, exc)
            else:
                self._mark_ea1j_reducer_component(
                    pipeline_trace,
                    "character_delta_upsert",
                    "skipped",
                    reason="no_character_deltas",
                )
                logger.info("Critic.run no character_deltas in output (turn=%d)", turn_index)

            # H-5b: pending_threads → pending_threads 테이블
            pending_threads = result.get("pending_threads")
            if isinstance(pending_threads, list) and pending_threads:
                try:
                    ch_result = self._process_pending_threads(
                        chat_session_id=chat_session_id,
                        turn_index=turn_index,
                        hook_candidates=pending_threads,
                    )
                    logger.info(
                        "Critic.run pending_threads processed (turn=%d, inserted=%d, updated=%d, skipped=%d, trace=%s)",
                        turn_index,
                        ch_result.get("inserted", 0),
                        ch_result.get("updated", 0),
                        ch_result.get("skipped", 0),
                        ch_result.get("trace") or {},
                    )
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "pending_thread_upsert",
                        "ok",
                        details={
                            "inserted": int(ch_result.get("inserted", 0) or 0),
                            "updated": int(ch_result.get("updated", 0) or 0),
                            "skipped": int(ch_result.get("skipped", 0) or 0),
                        },
                    )
                    if ch_result.get("skipped", 0) > 0:
                        append_audit_log(
                            event_type="critic_ingest_trace",
                            chat_session_id=chat_session_id or "default",
                            summary=f"Critic pending thread guard skipped items at turn {turn_index}",
                            details={
                                "turn_index": turn_index,
                                "surface": "pending_threads",
                                "trace": ch_result.get("trace") or {},
                            },
                            source="critic",
                        )
                except Exception as exc:
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "pending_thread_upsert",
                        "failed",
                        reason="pending_threads_save_failed",
                        details={"error": str(exc)[:160]},
                    )
                    logger.warning("Critic.run pending_threads save failed (turn=%d): %s", turn_index, exc)
            else:
                self._mark_ea1j_reducer_component(
                    pipeline_trace,
                    "pending_thread_upsert",
                    "skipped",
                    reason="no_pending_threads",
                )
                logger.info("Critic.run no pending_threads in output (turn=%d)", turn_index)

            prune_targets = result.get("prune_targets")
            if isinstance(prune_targets, list) and prune_targets:
                logger.info(
                    "Critic.run prune_targets received (turn=%d, targets=%s)",
                    turn_index,
                    prune_targets,
                )
                # Sprint 4-C-2: soft prune 실행
                if getattr(settings, "PRUNE_POLICY", "soft") == "soft":
                    self._apply_soft_prune(prune_targets, chat_session_id, turn_index)
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "soft_prune",
                        "ok",
                        details={"target_count": len(prune_targets)},
                    )
                else:
                    self._mark_ea1j_reducer_component(
                        pipeline_trace,
                        "soft_prune",
                        "skipped",
                        reason="prune_policy_disabled",
                    )
            else:
                self._mark_ea1j_reducer_component(
                    pipeline_trace,
                    "soft_prune",
                    "skipped",
                    reason="no_prune_targets",
                )
        except Exception as exc:
            pipeline_trace.setdefault("warnings", []).append(f"run_exception:{str(exc)[:160]}")
            logger.warning("Critic.run failed quietly: %s", exc)
        finally:
            finalized_trace = self._finalize_ea1j_pipeline_trace(pipeline_trace)
            self._write_ea1j_pipeline_audit(
                chat_session_id=chat_session_id,
                turn_index=turn_index,
                pipeline_trace=finalized_trace,
            )

    def _save_direct_evidence_records(
        self,
        turn_index: int,
        turn_content: str,
        evidence_excerpts: list[str],
        chat_session_id: str | None = None,
    ) -> dict:
        session_id = (chat_session_id or "").strip() or "default"
        normalized_turn = str(turn_content or "").strip()
        inserted = 0
        skipped = 0
        started_at = perf_counter()
        trace = {
            "policy_version": "lc1a.v1",
            "attempted": len(evidence_excerpts or []),
            "sanitized_empty_or_ungrounded": 0,
            "duplicate_source_hash": 0,
            "write_chars": 0,
            "elapsed_ms": 0.0,
        }

        db = SessionLocal()
        try:
            for excerpt_index, excerpt in enumerate(evidence_excerpts):
                text = sanitize_evidence_excerpt(excerpt, normalized_turn)
                if not text:
                    trace["sanitized_empty_or_ungrounded"] += 1
                    skipped += 1
                    continue

                digest = hashlib.sha1(
                    f"{session_id}|{turn_index}|{excerpt_index}|{normalized_turn}|{text}".encode("utf-8")
                ).hexdigest()

                existing = (
                    db.query(DirectEvidenceRecord)
                    .filter(DirectEvidenceRecord.chat_session_id == session_id)
                    .filter(DirectEvidenceRecord.source_hash == digest)
                    .first()
                )
                if existing:
                    trace["duplicate_source_hash"] += 1
                    skipped += 1
                    continue

                row = DirectEvidenceRecord(
                    chat_session_id=session_id,
                    evidence_kind="turn_excerpt",
                    evidence_text=text,
                    source_turn_start=turn_index,
                    source_turn_end=turn_index,
                    turn_anchor=turn_index,
                    source_message_ids_json=json.dumps([f"turn:{turn_index}"], ensure_ascii=False),
                    source_hash=digest,
                    archive_state=normalize_archive_state("pending_capture"),
                    capture_stage="critic_extract",
                    capture_verification=normalize_capture_verification("pending"),
                    committed_gate=resolve_committed_gate("pending"),
                    lineage_json=json.dumps({
                        "source": "critic.evidence_excerpts",
                        "excerpt_index": excerpt_index,
                    }, ensure_ascii=False),
                )
                db.add(row)
                inserted += 1
                trace["write_chars"] += len(text)

            db.commit()
            trace["elapsed_ms"] = round((perf_counter() - started_at) * 1000.0, 3)
            return {"inserted": inserted, "skipped": skipped, "trace": trace}
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _sanitize_participant_artifact_value(
        self,
        value: object,
        *,
        field_name: str | None = None,
        trace: dict | None = None,
    ) -> object:
        if isinstance(value, dict):
            sanitized: dict = {}
            for key, item in value.items():
                key_text = str(key or "")
                scrubbed = self._sanitize_participant_artifact_value(
                    item,
                    field_name=key_text,
                    trace=trace,
                )
                if scrubbed is _PARTICIPANT_ARTIFACT_DROP:
                    if trace is not None:
                        trace["dropped_items"] += 1
                    continue
                if isinstance(scrubbed, str):
                    scrubbed = scrubbed.strip()
                    if not scrubbed:
                        if trace is not None:
                            trace["dropped_items"] += 1
                        continue
                if scrubbed == {} or scrubbed == []:
                    if trace is not None:
                        trace["dropped_items"] += 1
                    continue
                sanitized[key] = scrubbed
            return sanitized

        if isinstance(value, list):
            sanitized_list: list = []
            for item in value:
                scrubbed = self._sanitize_participant_artifact_value(
                    item,
                    field_name=field_name,
                    trace=trace,
                )
                if scrubbed is _PARTICIPANT_ARTIFACT_DROP:
                    if trace is not None:
                        trace["dropped_items"] += 1
                    continue
                if isinstance(scrubbed, str):
                    scrubbed = scrubbed.strip()
                    if not scrubbed:
                        if trace is not None:
                            trace["dropped_items"] += 1
                        continue
                if scrubbed == {} or scrubbed == []:
                    if trace is not None:
                        trace["dropped_items"] += 1
                    continue
                sanitized_list.append(scrubbed)
            return sanitized_list

        if isinstance(value, str):
            raw_text = str(value or "")
            baseline = raw_text.strip()

            if field_name in _PARTICIPANT_DERIVED_RELATION_KEYS:
                if baseline and trace is not None:
                    trace["sanitized_fields"] += 1
                return _PARTICIPANT_ARTIFACT_DROP

            if field_name in _PARTICIPANT_ENTITY_SLOT_KEYS:
                actor = _normalize_actor_reference(raw_text, allow_player_token=False)
                if trace is not None and actor != baseline:
                    trace["sanitized_fields"] += 1
                return actor or _PARTICIPANT_ARTIFACT_DROP

            cleaned = _sanitize_participant_text(raw_text)
            if trace is not None and cleaned != baseline:
                trace["sanitized_fields"] += 1
            return cleaned or _PARTICIPANT_ARTIFACT_DROP

        return value

    def _sanitize_state_deltas_for_participant(
        self,
        state_deltas: object,
        *,
        trace: dict | None = None,
    ) -> dict:
        if not isinstance(state_deltas, dict):
            return {}

        sanitized: dict = {}

        scene_state = self._sanitize_participant_artifact_value(
            state_deltas.get("scene_state"),
            field_name="scene_state",
            trace=trace,
        )
        if isinstance(scene_state, dict) and scene_state:
            sanitized["scene_state"] = scene_state

        relationship_changes: list[dict] = []
        for raw_item in state_deltas.get("relationship_changes") or []:
            if not isinstance(raw_item, dict):
                continue
            identity = raw_item.get("identity") if isinstance(raw_item.get("identity"), dict) else {}
            pair_from, pair_to = _parse_relationship_pair_text(
                raw_item.get("pair") or identity.get("pair") or raw_item.get("pair_key") or identity.get("pair_key")
            )
            from_name = _normalize_actor_reference(
                raw_item.get("from")
                or raw_item.get("source")
                or identity.get("left_entity")
                or identity.get("source")
                or pair_from,
                allow_player_token=False,
            )
            to_name = _normalize_actor_reference(
                raw_item.get("to")
                or raw_item.get("target")
                or identity.get("right_entity")
                or identity.get("target")
                or pair_to,
                allow_player_token=False,
            )
            if not from_name or not to_name:
                if trace is not None:
                    trace["dropped_items"] += 1
                continue

            cleaned = self._sanitize_participant_artifact_value(raw_item, trace=trace)
            if not isinstance(cleaned, dict):
                cleaned = {}
            cleaned["from"] = from_name
            cleaned["to"] = to_name
            cleaned.pop("pair", None)
            cleaned.pop("pair_key", None)

            identity_clean = cleaned.get("identity") if isinstance(cleaned.get("identity"), dict) else {}
            identity_clean["left_entity"] = from_name
            identity_clean["right_entity"] = to_name
            identity_clean.pop("pair", None)
            identity_clean.pop("pair_key", None)
            cleaned["identity"] = identity_clean
            relationship_changes.append(cleaned)

        if relationship_changes:
            sanitized["relationship_changes"] = relationship_changes

        unresolved_threads = self._sanitize_participant_artifact_value(
            state_deltas.get("unresolved_threads"),
            field_name="unresolved_threads",
            trace=trace,
        )
        if isinstance(unresolved_threads, dict) and unresolved_threads:
            sanitized["unresolved_threads"] = unresolved_threads

        return sanitized

    def _sanitize_pending_thread_candidate_for_participant(
        self,
        raw_candidate: object,
        *,
        trace: dict | None = None,
    ) -> dict | None:
        if not isinstance(raw_candidate, dict):
            return None

        cleaned = self._sanitize_participant_artifact_value(raw_candidate, trace=trace)
        if not isinstance(cleaned, dict):
            cleaned = {}

        thread_type = str(raw_candidate.get("thread_type") or cleaned.get("thread_type") or "").strip()
        title = _sanitize_participant_text(raw_candidate.get("title") or cleaned.get("title") or "")
        if not thread_type or not title:
            if trace is not None:
                trace["dropped_items"] += 1
            return None

        cleaned["thread_type"] = thread_type
        cleaned["title"] = title

        owner = _normalize_actor_reference(raw_candidate.get("owner") or cleaned.get("owner"), allow_player_token=False)
        target = _normalize_actor_reference(raw_candidate.get("target") or cleaned.get("target"), allow_player_token=False)
        if owner:
            cleaned["owner"] = owner
        else:
            cleaned.pop("owner", None)
        if target:
            cleaned["target"] = target
        else:
            cleaned.pop("target", None)

        details = _sanitize_participant_text(
            raw_candidate.get("details") or cleaned.get("details") or cleaned.get("details_json") or ""
        )
        if details:
            cleaned["details"] = details
        else:
            cleaned.pop("details", None)
            cleaned.pop("details_json", None)

        return cleaned

    def _sanitize_character_delta_for_participant(
        self,
        raw_delta: object,
        *,
        trace: dict | None = None,
    ) -> dict | None:
        if not isinstance(raw_delta, dict):
            return None

        name = _normalize_actor_reference(raw_delta.get("name"), allow_player_token=False)
        if not name:
            if trace is not None:
                trace["dropped_items"] += 1
            return None

        relationships = self._sanitize_character_relationships(raw_delta.get("relationships"))
        events = raw_delta.get("events") if isinstance(raw_delta.get("events"), list) else []
        if (
            looks_like_transient_descriptor_character_name(name)
            and not has_character_continuity_anchor(
                appearance_json=raw_delta.get("appearance"),
                personality_json=raw_delta.get("personality"),
                relationships_json=relationships,
                speech_style_json=raw_delta.get("speech_style"),
                events=events,
            )
        ):
            if trace is not None:
                trace["dropped_items"] += 1
            return None

        cleaned = self._sanitize_participant_artifact_value(raw_delta, trace=trace)
        if not isinstance(cleaned, dict):
            cleaned = {}
        cleaned["name"] = name

        if relationships:
            cleaned["relationships"] = relationships
        else:
            cleaned.pop("relationships", None)

        events = cleaned.get("events") if isinstance(cleaned.get("events"), list) else []
        if events:
            cleaned["events"] = [evt for evt in events if isinstance(evt, dict) and evt]
        else:
            cleaned.pop("events", None)

        return cleaned

    def _sanitize_result_for_participant(self, result: dict) -> tuple[dict, dict]:
        if not isinstance(result, dict):
            return {}, {"policy_version": "participant_scrub.v1", "sanitized_fields": 0, "dropped_items": 0}

        trace = {
            "policy_version": "participant_scrub.v1",
            "sanitized_fields": 0,
            "dropped_items": 0,
        }
        sanitized = dict(result)

        raw_turn_summary = sanitized.get("turn_summary")
        if isinstance(raw_turn_summary, str):
            cleaned_summary = _sanitize_participant_text(raw_turn_summary)
            if cleaned_summary != str(raw_turn_summary or "").strip():
                trace["sanitized_fields"] += 1
            sanitized["turn_summary"] = cleaned_summary

        raw_excerpts = sanitized.get("evidence_excerpts")
        if isinstance(raw_excerpts, list):
            cleaned_excerpts: list[str] = []
            for item in raw_excerpts:
                if not isinstance(item, str):
                    trace["dropped_items"] += 1
                    continue
                cleaned_item = _sanitize_participant_text(item)
                if not cleaned_item:
                    trace["dropped_items"] += 1
                    continue
                if cleaned_item != str(item or "").strip():
                    trace["sanitized_fields"] += 1
                cleaned_excerpts.append(cleaned_item[:500])
            sanitized["evidence_excerpts"] = cleaned_excerpts

        raw_kg_triples = sanitized.get("kg_triples")
        if isinstance(raw_kg_triples, list):
            cleaned_kg_triples: list = []
            for item in raw_kg_triples:
                if not isinstance(item, dict):
                    cleaned_kg_triples.append(item)
                    continue
                if _is_participant_or_role_label(item.get("subject")) or _is_participant_or_role_label(item.get("object")):
                    trace["dropped_items"] += 1
                    continue
                cleaned_kg_triples.append(item)
            sanitized["kg_triples"] = cleaned_kg_triples

        raw_state_deltas = sanitized.get("state_deltas")
        if isinstance(raw_state_deltas, dict):
            sanitized["state_deltas"] = self._sanitize_state_deltas_for_participant(
                raw_state_deltas,
                trace=trace,
            )

        raw_character_deltas = sanitized.get("character_deltas")
        if isinstance(raw_character_deltas, list):
            cleaned_character_deltas: list[dict] = []
            for item in raw_character_deltas:
                cleaned_item = self._sanitize_character_delta_for_participant(item, trace=trace)
                if cleaned_item:
                    cleaned_character_deltas.append(cleaned_item)
            sanitized["character_deltas"] = cleaned_character_deltas

        raw_pending_threads = sanitized.get("pending_threads")
        if isinstance(raw_pending_threads, list):
            cleaned_pending_threads: list[dict] = []
            for item in raw_pending_threads:
                cleaned_item = self._sanitize_pending_thread_candidate_for_participant(item, trace=trace)
                if cleaned_item:
                    cleaned_pending_threads.append(cleaned_item)
            sanitized["pending_threads"] = cleaned_pending_threads

        processed_keys = {
            "turn_summary",
            "evidence_excerpts",
            "kg_triples",
            "state_deltas",
            "character_deltas",
            "pending_threads",
        }
        for key in list(sanitized.keys()):
            if key in processed_keys:
                continue
            if key in {"importance_score", "emotional_intensity", "narrative_significance", "archive_hint", "library_hint", "palace_hint"}:
                continue
            scrubbed = self._sanitize_participant_artifact_value(
                sanitized.get(key),
                field_name=str(key),
                trace=trace,
            )
            if scrubbed is _PARTICIPANT_ARTIFACT_DROP or scrubbed in ("", None, {}, []):
                sanitized.pop(key, None)
                continue
            sanitized[key] = scrubbed

        return sanitized, trace

    def _sanitize_memory_summary_for_critic(self, summary_json: dict) -> tuple[dict, dict]:
        """RG-1j: memory summary payload에서 control/meta 텍스트를 제거한다."""
        if not isinstance(summary_json, dict):
            return {}, {"dropped_fields": 0, "sanitized_fields": 0, "dropped_evidence_excerpts": 0}

        trace = {
            "dropped_fields": 0,
            "sanitized_fields": 0,
            "dropped_evidence_excerpts": 0,
        }
        sanitized = dict(summary_json)

        raw_turn_summary = sanitized.get("turn_summary")
        if isinstance(raw_turn_summary, str):
            safe_turn_summary = sanitize_text_for_critic(raw_turn_summary)
            if safe_turn_summary != raw_turn_summary:
                trace["sanitized_fields"] += 1
            if safe_turn_summary.strip():
                sanitized["turn_summary"] = safe_turn_summary
            else:
                sanitized.pop("turn_summary", None)
                trace["dropped_fields"] += 1

        raw_excerpts = sanitized.get("evidence_excerpts")
        if isinstance(raw_excerpts, list):
            safe_excerpts: list[str] = []
            dropped = 0
            for item in raw_excerpts:
                if not isinstance(item, str):
                    dropped += 1
                    continue
                safe_item = sanitize_text_for_critic(item)
                if not safe_item:
                    dropped += 1
                    continue
                safe_excerpts.append(safe_item[:500])
            if safe_excerpts:
                if len(safe_excerpts) != len(raw_excerpts):
                    trace["sanitized_fields"] += 1
                sanitized["evidence_excerpts"] = safe_excerpts
            else:
                if raw_excerpts:
                    trace["dropped_fields"] += 1
                sanitized["evidence_excerpts"] = []
            trace["dropped_evidence_excerpts"] = dropped

        if not str(sanitized.get("turn_summary") or "").strip():
            fallback_summary = sanitize_inline_text_for_critic(
                str(summary_json.get("turn_summary") or "")
            )
            if fallback_summary:
                sanitized["turn_summary"] = fallback_summary[:500]
                trace["sanitized_fields"] += 1
            else:
                sanitized["turn_summary"] = "(sanitized-empty)"
                trace["dropped_fields"] += 1

        return sanitized, trace

    def _extract(
        self,
        turn_content: str,
        context: list[dict],
        output_language_override: dict | None = None,
    ) -> dict | None:
        try:
            self.refresh_runtime_config()
            if not turn_content or not str(turn_content).strip():
                return None
            if not isinstance(context, list):
                context = []
            system_prompt = self._read_system_prompt()
            if not system_prompt:
                return None

            if not self.endpoint or not self.api_key or not self.model:
                logger.warning("Critic settings missing; extraction skipped.")
                return None

            detected_output_language = self._detect_output_language(turn_content, context)
            fallback_output_language = self._normalize_output_language_override(output_language_override)
            if detected_output_language.get("source") == "assistant_turn":
                output_language = detected_output_language
            else:
                output_language = fallback_output_language or detected_output_language
            payload = self._build_payload(system_prompt, turn_content, context, output_language)
            raw_text = self._request_completion(payload)
            if not raw_text:
                return None

            parsed = self._parse_response_json(raw_text)
            if parsed is None:
                logger.warning("Critic JSON parse failed.")
                return None

            return self._normalize_result(parsed)
        except Exception as exc:
            logger.warning("Critic extraction failed: %s", exc, exc_info=True)
            return None

    def _normalize_output_language_override(self, output_language_override: dict | None) -> dict | None:
        if not isinstance(output_language_override, dict):
            return None

        code = str(output_language_override.get("code") or "").strip().lower()
        if not code:
            return None

        label = str(output_language_override.get("label") or "").strip() or code
        source = str(output_language_override.get("source") or "client_override").strip() or "client_override"
        return {
            "code": code,
            "label": label,
            "source": source,
        }

    def _read_system_prompt(self) -> str | None:
        try:
            return read_prompt(self.prompt_name)
        except Exception as exc:
            logger.warning("Critic prompt load failed: %s", exc)
            return None

    def _build_payload(
        self,
        system_prompt: str,
        turn_content: str,
        context: list[dict],
        output_language: dict | None = None,
    ) -> dict:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": self._build_user_prompt(turn_content, context, output_language),
                },
            ],
        }

    def _build_google_payload(self, payload: dict) -> dict:
        messages = payload.get("messages") or []
        system_prompt = "\n\n".join(
            str(item.get("content") or "")
            for item in messages
            if item.get("role") == "system"
        ).strip()
        user_content = "\n\n".join(
            str(item.get("content") or "")
            for item in messages
            if item.get("role") != "system"
        ).strip()
        body = {
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        }
        if system_prompt:
            body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        return body

    def _build_user_prompt(
        self,
        turn_content: str,
        context: list[dict],
        output_language: dict | None = None,
    ) -> str:
        # 너무 긴 컨텐츠는 토큰 오버플로 방지를 위해 절단
        safe_content = str(turn_content or "")
        if len(safe_content) > 4000:
            safe_content = safe_content[:4000] + "...(절단됨)"
        context_text = self._format_context(sanitize_context_messages_for_critic(context))
        language_block = self._format_output_language_hint(output_language)
        return (
            f"{language_block}\n\n"
            f"{self._format_participant_contract()}\n\n"
            "<Latest_Turn>\n"
            f"{safe_content}\n"
            "</Latest_Turn>\n\n"
            "<Context>\n"
            f"{context_text}\n"
            "</Context>"
        )

    def _format_participant_contract(self) -> str:
        return (
            "<Participant_Contract>\n"
            "Do not persist the real chat participant as an entity, relationship target, owner, or role label.\n"
            "Never use user/player/participant/assistant/system/developer labels as character names or structured artifact subjects/objects.\n"
            "Treat RisuAI placeholders like {{user}} and {{char}} as runtime speaker markers, never as persisted entity labels.\n"
            "Preserve continuity through scene state, world changes, unresolved consequences, and character-internal updates phrased without naming the participant.\n"
            "</Participant_Contract>"
        )

    def _detect_output_language(self, turn_content: str, context: list[dict]) -> dict:
        """
                assistant output을 최우선으로, 없으면 Latest_Turn / 최근 user context를 바탕으로 출력 언어 힌트를 만든다.

        반환 예시:
        {
          "code": "ko",
          "label": "Korean",
          "source": "latest_turn",
        }
        """
        assistant_sample = self._extract_labeled_turn_segment(turn_content, "assistant")
        if assistant_sample:
            hint = self._detect_language_from_text(assistant_sample)
            hint["source"] = "assistant_turn"
            return hint

        candidates: list[str] = []
        if isinstance(turn_content, str) and turn_content.strip():
            candidates.append(turn_content.strip())

        if isinstance(context, list):
            for item in reversed(context):
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip().lower()
                content = item.get("content")
                if role != "user" or not isinstance(content, str) or not content.strip():
                    continue
                candidates.append(content.strip())
                if len(candidates) >= 3:
                    break

        sample = "\n".join(candidates)[:6000]
        return self._detect_language_from_text(sample)

    def _extract_labeled_turn_segment(self, turn_content: str, target_role: str) -> str:
        text = str(turn_content or "")
        if not text.strip():
            return ""

        target = str(target_role or "").strip().lower()
        if target not in {"user", "assistant"}:
            return ""

        user_labels = {"user", "{{user}}", "유저", "사용자", "ユーザー"}
        assistant_labels = {"assistant", "{{char}}", "{{assistant}}", "char", "bot", "model", "ai", "어시스턴트", "アシスタント"}

        current_role = ""
        buffer: list[str] = []
        segments: list[tuple[str, str]] = []

        for raw_line in text.splitlines():
            line = str(raw_line or "")
            match = re.match(r"^\s*([^:\n]{1,20})\s*:\s*(.*)$", line)
            if match:
                label = match.group(1).strip().lower()
                if label in user_labels or label in assistant_labels:
                    if current_role and buffer:
                        segment_text = "\n".join(buffer).strip()
                        if segment_text:
                            segments.append((current_role, segment_text))
                    current_role = "assistant" if label in assistant_labels else "user"
                    tail = match.group(2).strip()
                    buffer = [tail] if tail else []
                    continue

            if current_role:
                buffer.append(line)

        if current_role and buffer:
            segment_text = "\n".join(buffer).strip()
            if segment_text:
                segments.append((current_role, segment_text))

        selected = [segment for role, segment in segments if role == target]
        if not selected:
            return ""
        return "\n\n".join(selected[-2:]).strip()

    def _detect_language_from_text(self, text: str) -> dict:
        t = str(text or "")
        if not t.strip():
            return {"code": "auto", "label": "Same as source text", "source": "empty"}

        counts = {
            "ko": len(re.findall(r"[가-힣]", t)),
            "ja": len(re.findall(r"[\u3040-\u30ff]", t)),
            "zh": len(re.findall(r"[\u4e00-\u9fff]", t)),
            "latin": len(re.findall(r"[A-Za-z]", t)),
            "cyrillic": len(re.findall(r"[\u0400-\u04FF]", t)),
            "arabic": len(re.findall(r"[\u0600-\u06FF]", t)),
            "devanagari": len(re.findall(r"[\u0900-\u097F]", t)),
        }

        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top_key, top_count = ranked[0]
        if top_count <= 0:
            return {"code": "auto", "label": "Same as source text", "source": "no_script_signal"}

        if top_key == "latin":
            return {
                "code": "auto_latin",
                "label": "Same language as source text (Latin script)",
                "source": "latest_turn",
            }
        if top_key == "cyrillic":
            return {
                "code": "auto_cyrillic",
                "label": "Same language as source text (Cyrillic script)",
                "source": "latest_turn",
            }
        if top_key == "arabic":
            return {
                "code": "auto_arabic",
                "label": "Same language as source text (Arabic script)",
                "source": "latest_turn",
            }
        if top_key == "devanagari":
            return {
                "code": "auto_devanagari",
                "label": "Same language as source text (Devanagari script)",
                "source": "latest_turn",
            }

        label_map = {
            "ko": "Korean",
            "ja": "Japanese",
            "zh": "Chinese",
        }
        return {
            "code": top_key,
            "label": label_map.get(top_key, "Same as source text"),
            "source": "latest_turn",
        }

    def _format_output_language_hint(self, output_language: dict | None) -> str:
        info = output_language or {"code": "auto", "label": "Same as source text", "source": "fallback"}
        code = str(info.get("code") or "auto")
        label = str(info.get("label") or "Same as source text")
        source = str(info.get("source") or "unknown")
        return (
            "<Output_Language>\n"
            f"detected_language_code: {code}\n"
            f"detected_language_label: {label}\n"
            f"signal_source: {source}\n"
            "Hard rule: All natural-language values in the JSON output must follow the detected language.\n"
            "- Keep JSON keys/snake_case schema unchanged.\n"
            "- evidence_excerpts must remain faithful to original utterances.\n"
            "- If code starts with auto_*, infer the exact language from Latest_Turn and stay consistent.\n"
            "</Output_Language>"
        )

    def _format_context(self, context: list[dict]) -> str:
        if not context or not isinstance(context, list):
            return ""

        recent_items = context[-6:]
        lines: list[str] = []
        for item in recent_items:
            role = str(item.get("role", "unknown")).strip() or "unknown"
            content = sanitize_text_for_critic(item.get("content", ""))
            if not content:
                continue
            compact = " ".join(content.split())
            if len(compact) > 300:
                compact = compact[:297] + "..."
            lines.append(f"[{role}] {compact}")

        return "\n".join(lines)

    def _build_chat_url(self) -> str:
        provider = self._effective_provider()
        if provider == "vertex":
            return _normalize_google_generate_content_url(self.endpoint, self.model, is_vertex=True)
        if provider == "gemini":
            return _normalize_google_generate_content_url(self.endpoint, self.model, is_vertex=False)
        normalized = self.endpoint.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return normalized + "/chat/completions"

    def _request_completion(self, payload: dict) -> str | None:
        try:
            provider = self._effective_provider()
            with httpx.Client(timeout=self.timeout) as client:
                request_payload = payload
                headers = {"Content-Type": "application/json"}
                if provider == "vertex":
                    request_payload = self._build_google_payload(payload)
                    headers["Authorization"] = f"Bearer {_get_vertex_access_token(client, self.api_key)}"
                elif provider == "gemini":
                    request_payload = self._build_google_payload(payload)
                    headers["x-goog-api-key"] = self.api_key
                else:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                response = client.post(
                    self._build_chat_url(),
                    headers=headers,
                    json=request_payload,
                )
                response.raise_for_status()
                data = response.json()
            if provider in {"vertex", "gemini"}:
                return _extract_google_message_text(data)
            return self._extract_message_text(data)
        except Exception as exc:
            logger.warning("Critic LLM call failed: %s", exc, exc_info=True)
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

    def _process_pending_threads(
        self,
        chat_session_id: str | None,
        turn_index: int,
        hook_candidates: list,
    ) -> dict:
        """
        H-5b: critic이 추출한 pending_threads 후보를 DB에 저장한다.
        - title(소문자)이 같은 open/paused hook이 이미 있으면 last_seen_turn만 갱신한다.
        - 신규 hook은 status='open'으로 insert한다.
        - MO의 기존 session-scoped row 패턴에 맞춰 구현.
        """
        VALID_THREAD_TYPES = {"promise", "unresolved_goal", "open_question", "risk", "emotional_debt"}
        MIN_CONFIDENCE = 0.3

        sid = chat_session_id or "default"
        inserted = 0
        updated = 0
        skipped = 0
        trace = {
            "invalid_shape": 0,
            "missing_required": 0,
            "prompt_control_title": 0,
            "invalid_thread_type": 0,
            "low_confidence": 0,
        }

        db = SessionLocal()
        try:
            for raw in hook_candidates:
                raw = self._sanitize_pending_thread_candidate_for_participant(raw)
                if not isinstance(raw, dict):
                    trace["invalid_shape"] += 1
                    skipped += 1
                    continue

                thread_type = str(raw.get("thread_type") or "").strip().lower()
                raw_title = str(raw.get("title") or "")
                title = sanitize_inline_text_for_critic(raw_title)
                confidence = raw.get("confidence")

                if _looks_like_prompt_control_title(raw_title) or _looks_like_prompt_control_title(title):
                    logger.info("pending_threads: prompt-control title '%s' — skipped", raw_title or title)
                    trace["prompt_control_title"] += 1
                    skipped += 1
                    continue

                # 필수값 검증
                if not thread_type or not title:
                    trace["missing_required"] += 1
                    skipped += 1
                    continue
                if thread_type not in VALID_THREAD_TYPES:
                    logger.warning("pending_threads: unknown thread_type '%s' — skipped", thread_type)
                    trace["invalid_thread_type"] += 1
                    skipped += 1
                    continue
                try:
                    confidence_val = float(confidence) if confidence is not None else None
                    if confidence_val is not None and confidence_val < MIN_CONFIDENCE:
                        logger.info("pending_threads: low confidence %.2f for '%s' — skipped", confidence_val, title)
                        trace["low_confidence"] += 1
                        skipped += 1
                        continue
                except (TypeError, ValueError):
                    confidence_val = None

                owner = _normalize_actor_reference(raw.get("owner"), allow_player_token=False) or None
                target = _normalize_actor_reference(raw.get("target"), allow_player_token=False) or None
                details = _sanitize_participant_text(
                    sanitize_inline_text_for_critic(raw.get("details") or "")
                ) or None

                # title 기준 중복 체크 (소문자 비교, open/paused 상태만)
                title_lower = title.lower()
                existing = (
                    db.query(PendingThread)
                    .filter(
                        PendingThread.chat_session_id == sid,
                        PendingThread.status.in_(["open", "paused"]),
                    )
                    .all()
                )
                matched = next(
                    (h for h in existing if (h.title or "").lower() == title_lower),
                    None,
                )

                if matched:
                    # 같은 hook — last_seen_turn만 갱신
                    matched.last_seen_turn = turn_index
                    # confidence가 더 높으면 갱신
                    if confidence_val is not None and (matched.confidence is None or confidence_val > matched.confidence):
                        matched.confidence = confidence_val
                    updated += 1
                else:
                    # 신규 hook insert
                    new_hook = PendingThread(
                        chat_session_id=sid,
                        thread_type=thread_type,
                        title=title,
                        status="open",
                        owner=owner,
                        target=target,
                        source_turn=turn_index,
                        last_seen_turn=turn_index,
                        confidence=confidence_val,
                        details_json=details,
                    )
                    db.add(new_hook)
                    inserted += 1

            db.commit()
        finally:
            db.close()

        return {"inserted": inserted, "updated": updated, "skipped": skipped, "trace": trace}

    def _apply_soft_prune(self, prune_targets: list, chat_session_id: str | None, turn_index: int) -> None:
        """Sprint 4-C-2: prune_targets의 키워드로 기억을 찾아 importance를 2 하향한다."""
        try:
            sid = chat_session_id or "default"
            db = SessionLocal()
            try:
                pruned = []
                for target in prune_targets:
                    keyword = str(target).strip().lower() if target else ""
                    if not keyword:
                        continue
                    rows = (
                        db.query(Memory)
                        .filter(Memory.chat_session_id == sid)
                        .filter(Memory.importance > 1)
                        .all()
                    )
                    for mem in rows:
                        summary_text = (mem.summary_json or "").lower()
                        if keyword in summary_text:
                            old_imp = mem.importance or 5
                            new_imp = max(1.0, old_imp - 2.0)
                            mem.importance = new_imp
                            pruned.append({"id": mem.id, "old": old_imp, "new": new_imp, "keyword": keyword})
                db.commit()

                if pruned:
                    logger.info("Critic soft_prune applied: %d memories adjusted (turn=%d)", len(pruned), turn_index)
                    try:
                        append_audit_log(
                            event_type="soft_prune",
                            chat_session_id=sid,
                            summary=f"Soft prune: {len(pruned)} memories, turn {turn_index}",
                            details={"pruned": pruned, "targets": prune_targets},
                            source="critic",
                        )
                    except Exception:
                        pass
            finally:
                db.close()
        except Exception as exc:
            logger.warning("Critic _apply_soft_prune failed (non-fatal): %s", exc)

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

    def _normalize_result(self, parsed: dict) -> dict:
        try:
            normalized = dict(parsed)
            normalized.setdefault("turn_summary", "")

            if not isinstance(normalized.get("archive_hint"), dict):
                legacy_library_hint = normalized.get("library_hint")
                legacy_palace_hint = normalized.get("palace_hint")
                if isinstance(legacy_library_hint, dict):
                    normalized["archive_hint"] = legacy_library_hint
                elif isinstance(legacy_palace_hint, dict):
                    normalized["archive_hint"] = legacy_palace_hint

            importance = normalized.get("importance_score", 5)
            try:
                importance = int(importance)
            except Exception:
                importance = 5
            normalized["importance_score"] = max(1, min(10, importance))

            kg_triples = normalized.get("kg_triples")
            if not isinstance(kg_triples, list):
                normalized["kg_triples"] = []

            prune_targets = normalized.get("prune_targets")
            if not isinstance(prune_targets, list):
                normalized["prune_targets"] = []

            # Phase 5-1: 새 critic 출력 필드 — fail-safe 파싱
            try:
                normalized["evidence_excerpts"] = self._normalize_evidence_excerpts(
                    normalized.get("evidence_excerpts")
                )
                normalized["emotional_intensity"] = self._normalize_float_signal(
                    normalized.get("emotional_intensity")
                )
                normalized["narrative_significance"] = self._normalize_float_signal(
                    normalized.get("narrative_significance")
                )
            except Exception as exc:
                logger.warning("Critic Phase 5-1 field normalization failed (non-fatal): %s", exc)
                normalized.setdefault("evidence_excerpts", [])
                normalized.setdefault("emotional_intensity", None)
                normalized.setdefault("narrative_significance", None)

            return normalized
        except Exception as exc:
            logger.warning("Critic _normalize_result failed: %s", exc)
            return {"turn_summary": "", "importance_score": 5, "kg_triples": [], "prune_targets": [], "evidence_excerpts": [], "emotional_intensity": None, "narrative_significance": None}

    # ------------------------------------------------------------------ #
    #  Phase 5-1: 새 필드 정규화 헬퍼
    # ------------------------------------------------------------------ #

    def _normalize_evidence_excerpts(self, raw: object) -> list[str]:
        """evidence_excerpts: 문자열 배열로 정규화. 비정상이면 빈 배열."""
        if not isinstance(raw, list):
            return []
        result: list[str] = []
        for item in raw[:5]:  # 최대 5개 (안전 상한)
            if isinstance(item, str) and item.strip():
                result.append(item.strip()[:500])  # 개별 발췌 길이 제한
        return result

    def _normalize_float_signal(self, raw: object) -> float | None:
        """0.0~1.0 범위 float로 정규화. 비정상이면 None."""
        if raw is None:
            return None
        try:
            val = float(raw)
        except (ValueError, TypeError):
            return None
        return max(0.0, min(1.0, val))

    def _sanitize_kg_triples_for_critic(self, raw_triples: list) -> tuple[list[dict], dict]:
        """Critic kg_triples를 최소 안전 규칙으로 정리하고 skip trace를 반환한다."""
        trace = {
            "accepted": 0,
            "skipped": 0,
            "invalid_shape": 0,
            "missing_required": 0,
            "control_or_sanitized_empty": 0,
        }
        sanitized: list[dict] = []

        for raw in raw_triples:
            if not isinstance(raw, dict):
                trace["skipped"] += 1
                trace["invalid_shape"] += 1
                continue

            raw_subject = raw.get("subject") or ""
            raw_predicate = raw.get("predicate") or ""
            raw_object = raw.get("object") or ""

            subject = _normalize_actor_reference(raw_subject, allow_player_token=False)
            predicate = str(raw_predicate).strip()
            object_value = _normalize_actor_reference(raw_object, allow_player_token=False)

            if not str(raw_subject).strip() or not predicate or not str(raw_object).strip():
                trace["skipped"] += 1
                trace["missing_required"] += 1
                continue

            if not subject or not object_value:
                trace["skipped"] += 1
                trace["control_or_sanitized_empty"] += 1
                continue

            sanitized.append({
                "subject": subject,
                "predicate": predicate,
                "object": object_value,
                "valid_from": raw.get("valid_from"),
                "valid_to": raw.get("valid_to"),
            })
            trace["accepted"] += 1

        return sanitized, trace

    def _sanitize_character_relationships(self, raw_relationships: object) -> list[dict]:
        if not isinstance(raw_relationships, list):
            return []

        sanitized: list[dict] = []
        for relation in raw_relationships[:20]:
            if not isinstance(relation, dict):
                continue
            target = _normalize_actor_reference(relation.get("target"), allow_player_token=False)
            if not target:
                continue
            cleaned_value = self._sanitize_participant_artifact_value(relation)
            cleaned = dict(cleaned_value) if isinstance(cleaned_value, dict) else dict(relation)
            cleaned["target"] = target
            sanitized.append(cleaned)
        return sanitized

    # ------------------------------------------------------------------ #
    #  Phase E-2: Character Delta → DB
    # ------------------------------------------------------------------ #

    def _process_character_deltas(
        self,
        chat_session_id: str | None,
        turn_index: int,
        character_deltas: list,
    ) -> dict:
        """character_deltas 배열을 파싱하여 character_states UPSERT + character_events INSERT."""
        sid = chat_session_id or "default"
        upserted = 0
        events_created = 0

        db = SessionLocal()
        try:
            consistency_checker = ConsistencyChecker()
            for delta in character_deltas:
                delta = self._sanitize_character_delta_for_participant(delta)
                if not isinstance(delta, dict):
                    continue
                name = _normalize_actor_reference(delta.get("name"), allow_player_token=False)
                if not name:
                    continue

                # ── CharacterState UPSERT (중복 방지: 교차-스크립트 이름 정규화 매칭) ──
                existing = _find_canonical_character(db, sid, name)

                appearance = self._normalize_character_axis_payload(delta.get("appearance"))
                personality = self._normalize_character_axis_payload(delta.get("personality"))
                status = self._normalize_character_axis_payload(delta.get("status"))
                relationships = self._sanitize_character_relationships(delta.get("relationships"))
                changed_axes: set[str] = set()
                accepted_events = 0

                gate_result = consistency_checker.check_before_save(
                    db,
                    chat_session_id=sid,
                    new_character_state={
                        "name": name,
                        "relationships": relationships,
                    },
                )
                consistency_checker.record_audit(
                    db,
                    chat_session_id=sid,
                    surface="character_state",
                    result=gate_result,
                    source="critic",
                    payload={
                        "character_name": name,
                        "turn_index": turn_index,
                    },
                )
                if gate_result.get("decision") in {"reject", "flag_for_review"}:
                    continue

                if existing:
                    # carry-forward default: 빈/동일 payload는 덮어쓰지 않고 유지한다.
                    if isinstance(appearance, dict) and appearance:
                        merged, changed = self._merge_json_field(existing.appearance_json, appearance)
                        if changed:
                            existing.appearance_json = json.dumps(merged, ensure_ascii=False)
                            changed_axes.add("appearance")
                    if isinstance(personality, dict) and personality:
                        merged, changed = self._merge_json_field(existing.personality_json, personality)
                        if changed:
                            existing.personality_json = json.dumps(merged, ensure_ascii=False)
                            changed_axes.add("personality")
                    if isinstance(status, dict) and status:
                        merged, changed = self._merge_json_field(existing.status_json, status)
                        if changed:
                            existing.status_json = json.dumps(merged, ensure_ascii=False)
                            changed_axes.add("status")
                    if isinstance(relationships, list) and relationships:
                        merged_relationships, changed = self._merge_relationships(
                            existing.relationships_json, relationships
                        )
                        if changed:
                            existing.relationships_json = merged_relationships
                            changed_axes.add("relationships")
                else:
                    new_state = CharacterState(
                        chat_session_id=sid,
                        character_name=name,
                        appearance_json=json.dumps(appearance, ensure_ascii=False) if isinstance(appearance, dict) and appearance else None,
                        personality_json=json.dumps(personality, ensure_ascii=False) if isinstance(personality, dict) and personality else None,
                        status_json=json.dumps(status, ensure_ascii=False) if isinstance(status, dict) and status else None,
                        relationships_json=json.dumps(relationships, ensure_ascii=False) if isinstance(relationships, list) and relationships else None,
                        turn_index=turn_index,
                    )
                    db.add(new_state)
                    upserted += 1
                    if appearance:
                        changed_axes.add("appearance")
                    if personality:
                        changed_axes.add("personality")
                    if status:
                        changed_axes.add("status")
                    if relationships:
                        changed_axes.add("relationships")

                # ── CharacterEvent INSERT ──
                events = delta.get("events")
                if isinstance(events, list):
                    for evt in events[:10]:  # 안전 상한
                        if not isinstance(evt, dict):
                            continue
                        evt_type = (evt.get("type") or "").strip()
                        if evt_type not in ("relationship_shift", "personality_change", "status_change", "appearance_change"):
                            continue
                        detail = _sanitize_participant_text(evt.get("detail") or "")
                        if not self._should_record_character_event(
                            existing=existing,
                            event_type=evt_type,
                            detail=detail,
                            changed_axes=changed_axes,
                        ):
                            continue
                        db.add(CharacterEvent(
                            chat_session_id=sid,
                            character_name=name,
                            turn_index=turn_index,
                            event_type=evt_type,
                            details_json=json.dumps({"detail": detail}, ensure_ascii=False) if detail else None,
                        ))
                        events_created += 1
                        accepted_events += 1

                if existing and (changed_axes or accepted_events):
                    existing.turn_index = turn_index
                    upserted += 1

            db.commit()
        finally:
            db.close()

        return {"upserted": upserted, "events_created": events_created}

    def _has_character_axis_value(self, value: object) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (dict, list, tuple, set)):
            return bool(value)
        return True

    def _normalize_character_axis_payload(self, value: object):
        if isinstance(value, dict):
            cleaned: dict = {}
            for key, item in value.items():
                normalized = self._normalize_character_axis_payload(item)
                if self._has_character_axis_value(normalized):
                    cleaned[key] = normalized
            return cleaned
        if isinstance(value, list):
            cleaned_list = [
                normalized
                for normalized in (self._normalize_character_axis_payload(item) for item in value)
                if self._has_character_axis_value(normalized)
            ]
            return cleaned_list
        if isinstance(value, str):
            text = _sanitize_participant_text(value)
            return text if text else None
        return value

    def _stored_character_axis_is_missing(self, existing_json: str | None, expected_type) -> bool:
        if not existing_json:
            return True
        try:
            parsed = json.loads(existing_json)
        except Exception:
            return True
        return not isinstance(parsed, expected_type) or not bool(parsed)

    def _character_event_axis_name(self, event_type: str) -> str | None:
        return {
            "appearance_change": "appearance",
            "personality_change": "personality",
            "status_change": "status",
            "relationship_shift": "relationships",
        }.get((event_type or "").strip())

    def _should_record_character_event(
        self,
        *,
        existing: CharacterState | None,
        event_type: str,
        detail: str,
        changed_axes: set[str],
    ) -> bool:
        if not detail:
            return False
        axis_name = self._character_event_axis_name(event_type)
        if axis_name and axis_name in changed_axes:
            return True
        if existing is None:
            return True
        if axis_name == "appearance":
            return self._stored_character_axis_is_missing(existing.appearance_json, dict)
        if axis_name == "personality":
            return self._stored_character_axis_is_missing(existing.personality_json, dict)
        if axis_name == "status":
            return self._stored_character_axis_is_missing(existing.status_json, dict)
        if axis_name == "relationships":
            return self._stored_character_axis_is_missing(existing.relationships_json, list)
        return False

    def _merge_json_field(self, existing_json: str | None, new_data: dict) -> tuple[dict, bool]:
        """기존 JSON 문자열과 새 dict를 병합하되, blank/identical payload는 carry-forward한다."""
        cleaned_new_data = self._normalize_character_axis_payload(new_data)
        if not isinstance(cleaned_new_data, dict) or not cleaned_new_data:
            try:
                existing = json.loads(existing_json) if existing_json else {}
            except Exception:
                existing = {}
            return existing if isinstance(existing, dict) else {}, False

        try:
            existing = {}
            if existing_json:
                parsed = json.loads(existing_json)
                if isinstance(parsed, dict):
                    existing = parsed
        except Exception:
            existing = {}

        merged = dict(existing)
        changed = False
        for key, value in cleaned_new_data.items():
            if merged.get(key) != value:
                merged[key] = value
                changed = True
        return merged, changed

    def _merge_relationships(self, existing_json: str | None, new_rels: list) -> tuple[str, bool]:
        """관계 배열 병합. target 기준으로 기존 관계를 업데이트하거나 새로 추가."""
        try:
            existing_list = []
            if existing_json:
                parsed = json.loads(existing_json)
                if isinstance(parsed, list):
                    existing_list = parsed

            # target → index 매핑
            by_target = {}
            for i, rel in enumerate(existing_list):
                if isinstance(rel, dict) and rel.get("target"):
                    by_target[rel["target"]] = i

            changed = False
            for new_rel in new_rels:
                if not isinstance(new_rel, dict):
                    continue
                target = (new_rel.get("target") or "").strip()
                if not target:
                    continue
                if target in by_target:
                    # 기존 관계 업데이트
                    idx = by_target[target]
                    merged_rel = dict(existing_list[idx])
                    merged_rel.update(new_rel)
                    if merged_rel != existing_list[idx]:
                        existing_list[idx] = merged_rel
                        changed = True
                else:
                    existing_list.append(new_rel)
                    by_target[target] = len(existing_list) - 1
                    changed = True

            return json.dumps(existing_list, ensure_ascii=False), changed
        except Exception:
            return json.dumps(new_rels, ensure_ascii=False), bool(new_rels)