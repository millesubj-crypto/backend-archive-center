import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx
import numpy as np
from sqlalchemy.orm import Session

from backend.archive.hybrid_policy import CURRENT_HYBRID_RETRIEVAL_POLICY
from backend.database import SessionLocal
from backend.embedding_provenance import (
    STEP13_RETRIEVAL_FALLBACK_POLICY_VERSION,
    assess_memory_embedding_state,
)
from backend import to_kst
from backend.models import ChatLog, KGTriple, Memory, DEFAULT_SESSION_ID
from backend.settings import settings

logger = logging.getLogger(__name__)

_STEP13_RETRIEVAL_DEGRADED_SIMILARITY_WEIGHT = 0.35

_PLAYER_ENTITY_TOKEN = "__PLAYER__"
_RISU_USER_PLACEHOLDER_PATTERN = re.compile(
    r"^\s*(?:\{\{\s*user\s*\}\}|<\s*user\s*>)\s*$",
    flags=re.IGNORECASE,
)
_RISU_CHAR_PLACEHOLDER_PATTERN = re.compile(
    r"^\s*(?:\{\{\s*char(?:=[^}]*)?\s*\}\}|<\s*char(?:=[^>]*)?\s*>)\s*$",
    flags=re.IGNORECASE,
)
_HY1_POLICY = CURRENT_HYBRID_RETRIEVAL_POLICY
_BUNDLE_ROOT_H18_5 = Path(__file__).resolve().parents[2]
_STEP18_5_CHROMA_LIVE_CUTOVER_CONTRACT_VERSION = "h18_5.limited_live.v1"
_STEP18_5_CHROMA_LIVE_MERGE_POLICY_VERSION = "h18_5.live_merge.v1"
_STEP18_5_CHROMA_LIVE_ALLOWED_MODES = frozenset({"off", "limited_memory", "primary_memory"})
_STEP18_5_CHROMA_LIVE_SUPPORT_SURFACE_MODE = "sqlite_hydrated_memory_support"
_STEP18_5_CHROMA_LIVE_PRECEDENCE_CEILING = "below_current_user_input_and_verified_direct_evidence"


def _resolve_chroma_shadow_persist_dir_h18_5() -> Path:
    raw = " ".join(
        str(getattr(settings, "CHROMA_SHADOW_PERSIST_DIRECTORY", ".chroma_shadow") or ".chroma_shadow").split()
    ).strip()
    target = Path(raw or ".chroma_shadow").expanduser()
    if not target.is_absolute():
        target = (_BUNDLE_ROOT_H18_5 / target).resolve()
    return target


def _resolve_chroma_shadow_collection_name_h18_5() -> str:
    raw = " ".join(
        str(getattr(settings, "CHROMA_SHADOW_COLLECTION_NAME", "archive_center_shadow") or "archive_center_shadow").split()
    ).strip()
    return raw or "archive_center_shadow"


def _normalize_chroma_live_cutover_mode_h18_5(value: object) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in _STEP18_5_CHROMA_LIVE_ALLOWED_MODES else "off"


def _normalize_chroma_live_candidate_limit_h18_5(value: object, *, fallback: int) -> int:
    fallback_limit = max(1, min(50, fallback if isinstance(fallback, int) and fallback > 0 else 5))
    try:
        limit = int(value)
    except Exception:
        limit = fallback_limit
    if limit <= 0:
        limit = fallback_limit
    return max(1, min(50, limit))


def _contains_cjk_hy1(value: str) -> bool:
    return bool(re.search(r"[가-힣\u3040-\u30ff\u4e00-\u9fff]", str(value or "")))


def _safe_load_summary_json_hy1(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _extract_terms_hy1(value: object, *, limit: int = 16) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def push_token(token: str) -> None:
        normalized = str(token or "").strip().casefold()
        if not normalized:
            return
        if normalized in _HY1_POLICY.stopwords:
            return
        if len(normalized) < 2 and not _contains_cjk_hy1(normalized):
            return
        if normalized in seen:
            return
        seen.add(normalized)
        terms.append(normalized)

    def walk(inner: object) -> None:
        if len(terms) >= limit or inner is None:
            return
        if isinstance(inner, dict):
            for item in inner.values():
                walk(item)
                if len(terms) >= limit:
                    return
            return
        if isinstance(inner, (list, tuple, set)):
            for item in inner:
                walk(item)
                if len(terms) >= limit:
                    return
            return
        for token in _HY1_POLICY.term_pattern.findall(str(inner or "")):
            push_token(token)
            if len(terms) >= limit:
                return

    walk(value)
    return terms


def _match_terms_hy1(query_terms: list[str], candidate_terms: list[str], *, limit: int = 4) -> list[str]:
    candidate_set = {str(term or "").strip().casefold() for term in candidate_terms if str(term or "").strip()}
    matched: list[str] = []
    for term in query_terms:
        normalized = str(term or "").strip().casefold()
        if normalized and normalized in candidate_set and normalized not in matched:
            matched.append(normalized)
        if len(matched) >= limit:
            break
    return matched


def _is_generic_archive_slot_hy1(value: object) -> bool:
    normalized = str(value or "").strip().casefold()
    if not normalized:
        return True
    return normalized in _HY1_POLICY.generic_archive_slots


def _build_hybrid_signal_hy1(
    *,
    query: str,
    summary_json: dict,
    archive_hint: dict,
    archive_wing: str | None,
    archive_room: str | None,
) -> dict:
    query_terms = _extract_terms_hy1(query, limit=8)
    if not query_terms:
        return {
            "keyword_overlap_score": 0.0,
            "keyword_overlap_boost": 0.0,
            "keyword_overlap_terms": [],
            "hybrid_baseline_policy_version": _HY1_POLICY.hybrid_baseline_version,
            "speaker_bias_score": 0.0,
            "location_bias_score": 0.0,
            "storyline_bias_score": 0.0,
            "soft_bias_score": 0.0,
            "speaker_bias_terms": [],
            "location_bias_terms": [],
            "storyline_bias_terms": [],
            "soft_bias_policy_version": _HY1_POLICY.soft_bias_version,
        }

    keyword_candidates = _extract_terms_hy1(
        [
            summary_json.get("turn_summary"),
            summary_json.get("summary"),
            summary_json.get("text"),
            summary_json.get("key_entities"),
            summary_json.get("key_events"),
            summary_json.get("relationship_changes_json"),
            summary_json.get("open_loops_json"),
            archive_hint,
            None if _is_generic_archive_slot_hy1(archive_wing) else archive_wing,
            None if _is_generic_archive_slot_hy1(archive_room) else archive_room,
        ],
        limit=32,
    )
    keyword_overlap_terms = _match_terms_hy1(query_terms, keyword_candidates)
    keyword_overlap_score = round(len(keyword_overlap_terms) / max(1, len(query_terms)), 4)
    keyword_overlap_boost = round(
        min(_HY1_POLICY.keyword_boost_cap, keyword_overlap_score * _HY1_POLICY.keyword_boost_cap),
        4,
    )

    speaker_candidates = _extract_terms_hy1(
        [
            summary_json.get("speaker"),
            summary_json.get("speakers"),
            summary_json.get("character_names"),
            summary_json.get("key_entities"),
            archive_hint.get("speaker") if isinstance(archive_hint, dict) else None,
            archive_hint.get("speakers") if isinstance(archive_hint, dict) else None,
            archive_hint.get("entities") if isinstance(archive_hint, dict) else None,
        ],
        limit=16,
    )
    location_candidates = _extract_terms_hy1(
        [
            summary_json.get("location"),
            summary_json.get("locations"),
            summary_json.get("scene_location"),
            summary_json.get("place"),
            archive_hint.get("location") if isinstance(archive_hint, dict) else None,
            archive_hint.get("place") if isinstance(archive_hint, dict) else None,
            archive_hint.get("room") if isinstance(archive_hint, dict) else None,
            None if _is_generic_archive_slot_hy1(archive_room) else archive_room,
        ],
        limit=16,
    )
    storyline_candidates = _extract_terms_hy1(
        [
            summary_json.get("storyline"),
            summary_json.get("storylines"),
            summary_json.get("storyline_name"),
            summary_json.get("current_arc"),
            summary_json.get("arc_name"),
            summary_json.get("chapter_title"),
            archive_hint.get("storyline") if isinstance(archive_hint, dict) else None,
            archive_hint.get("wing") if isinstance(archive_hint, dict) else None,
            None if _is_generic_archive_slot_hy1(archive_wing) else archive_wing,
        ],
        limit=16,
    )

    speaker_bias_terms = _match_terms_hy1(query_terms, speaker_candidates)
    location_bias_terms = _match_terms_hy1(query_terms, location_candidates)
    storyline_bias_terms = _match_terms_hy1(query_terms, storyline_candidates)

    speaker_bias_score = _HY1_POLICY.speaker_bias_weight if speaker_bias_terms else 0.0
    location_bias_score = _HY1_POLICY.location_bias_weight if location_bias_terms else 0.0
    storyline_bias_score = _HY1_POLICY.storyline_bias_weight if storyline_bias_terms else 0.0
    soft_bias_score = round(
        min(_HY1_POLICY.soft_bias_cap, speaker_bias_score + location_bias_score + storyline_bias_score),
        4,
    )

    return {
        "keyword_overlap_score": keyword_overlap_score,
        "keyword_overlap_boost": keyword_overlap_boost,
        "keyword_overlap_terms": keyword_overlap_terms,
        "hybrid_baseline_policy_version": _HY1_POLICY.hybrid_baseline_version,
        "speaker_bias_score": round(speaker_bias_score, 4),
        "location_bias_score": round(location_bias_score, 4),
        "storyline_bias_score": round(storyline_bias_score, 4),
        "soft_bias_score": soft_bias_score,
        "speaker_bias_terms": speaker_bias_terms,
        "location_bias_terms": location_bias_terms,
        "storyline_bias_terms": storyline_bias_terms,
        "soft_bias_policy_version": _HY1_POLICY.soft_bias_version,
    }


def _coerce_float_hy1(value: object) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _tail_budget_trace_hy1(
    candidate: dict,
    cutoff_item: dict,
    *,
    original_rank: int,
) -> dict | None:
    policy = _HY1_POLICY.tail_budget_policy
    candidate_final_score = _coerce_float_hy1(candidate.get("final_score"))
    cutoff_final_score = _coerce_float_hy1(cutoff_item.get("final_score"))
    score_gap = round(max(0.0, cutoff_final_score - candidate_final_score), 4)
    if score_gap > policy.max_score_gap:
        return None

    candidate_soft_bias = _coerce_float_hy1(candidate.get("soft_bias_score"))
    cutoff_soft_bias = _coerce_float_hy1(cutoff_item.get("soft_bias_score"))
    soft_bias_rescue = candidate_soft_bias > 0.0 and candidate_soft_bias > cutoff_soft_bias

    candidate_keyword_terms = candidate.get("keyword_overlap_terms") or []
    cutoff_keyword_terms = cutoff_item.get("keyword_overlap_terms") or []
    candidate_keyword_score = _coerce_float_hy1(candidate.get("keyword_overlap_score"))
    cutoff_keyword_score = _coerce_float_hy1(cutoff_item.get("keyword_overlap_score"))
    keyword_rescue = (
        candidate_keyword_score > cutoff_keyword_score
        and len(candidate_keyword_terms) > len(cutoff_keyword_terms)
    )

    if not soft_bias_rescue and not keyword_rescue:
        return None

    reason_parts: list[str] = []
    if soft_bias_rescue:
        reason_parts.append("structured_soft_bias_rescue")
    if keyword_rescue:
        reason_parts.append("keyword_overlap_rescue")

    return {
        "tail_budget_policy_version": policy.version,
        "tail_budget_original_rank": int(original_rank),
        "tail_budget_promoted": True,
        "tail_budget_reason": "+".join(reason_parts),
        "tail_budget_score_gap": score_gap,
        "_priority": (
            1 if soft_bias_rescue else 0,
            1 if keyword_rescue else 0,
            round(candidate_soft_bias - cutoff_soft_bias, 4),
            len(candidate_keyword_terms) - len(cutoff_keyword_terms),
            round(candidate_keyword_score - cutoff_keyword_score, 4),
            -score_gap,
            -int(original_rank),
        ),
    }


def _chroma_live_merge_trace_h18_5(
    candidate: dict,
    cutoff_item: dict,
    *,
    candidate_rank: int,
) -> dict | None:
    policy = _HY1_POLICY.tail_budget_policy
    candidate_final_score = _coerce_float_hy1(candidate.get("final_score"))
    cutoff_final_score = _coerce_float_hy1(cutoff_item.get("final_score"))
    score_gap = round(max(0.0, cutoff_final_score - candidate_final_score), 4)
    if score_gap > policy.max_score_gap:
        return None

    return {
        "chroma_live_merge_policy_version": _STEP18_5_CHROMA_LIVE_MERGE_POLICY_VERSION,
        "chroma_live_merged": True,
        "chroma_live_merge_rank": int(candidate_rank),
        "chroma_live_merge_reason": "hydrated_candidate_within_score_gap",
        "chroma_live_merge_score_gap": score_gap,
        "_priority": (
            -int(candidate_rank),
            -score_gap,
            candidate_final_score,
        ),
    }


def _apply_tail_budget_rescue_hy1(memory_results: list[dict], n_results: int) -> list[dict]:
    policy = _HY1_POLICY.tail_budget_policy
    if policy.max_promoted_items < 1 or n_results < 1:
        return list(memory_results[: max(0, n_results)])

    selected_with_rank = [
        (index + 1, item)
        for index, item in enumerate(memory_results[:n_results])
    ]
    tail_with_rank = [
        (index + 1, item)
        for index, item in enumerate(memory_results[n_results:], start=n_results)
    ]

    promotions_used = 0
    while promotions_used < policy.max_promoted_items and selected_with_rank and tail_with_rank:
        _cutoff_rank, cutoff_item = selected_with_rank[-1]
        eligible: list[tuple[tuple, int, dict, dict]] = []
        for original_rank, candidate in tail_with_rank:
            trace = _tail_budget_trace_hy1(candidate, cutoff_item, original_rank=original_rank)
            if trace:
                eligible.append((trace["_priority"], original_rank, candidate, trace))

        if not eligible:
            break

        _priority, promoted_rank, promoted_candidate, trace = max(eligible, key=lambda entry: entry[0])
        promoted_item = dict(promoted_candidate)
        for key, value in trace.items():
            if not key.startswith("_"):
                promoted_item[key] = value

        demoted_entry = selected_with_rank.pop()
        selected_with_rank.append((promoted_rank, promoted_item))
        selected_with_rank.sort(key=lambda entry: _coerce_float_hy1(entry[1].get("final_score")), reverse=True)
        tail_with_rank = [entry for entry in tail_with_rank if entry[0] != promoted_rank]
        tail_with_rank.append(demoted_entry)
        tail_with_rank.sort(key=lambda entry: entry[0])
        promotions_used += 1

    return [item for _rank, item in selected_with_rank]


def _apply_chroma_live_merge_h18_5(
    memory_results: list[dict],
    selected_results: list[dict],
    *,
    live_contract: dict,
) -> list[dict]:
    if str(live_contract.get("state") or "") != "chroma_engaged":
        return list(selected_results)

    candidate_rank_by_row_id = live_contract.get("candidate_rank_by_row_id") or {}
    if not isinstance(candidate_rank_by_row_id, dict) or not candidate_rank_by_row_id:
        return list(selected_results)

    promotion_budget = max(0, int(_HY1_POLICY.tail_budget_policy.max_promoted_items or 0))
    if promotion_budget < 1 or not selected_results:
        return list(selected_results)

    selected = [dict(item) for item in selected_results]
    selected_ids = {
        item.get("id")
        for item in selected
        if isinstance(item.get("id"), int)
    }

    promotions_used = 0
    while promotions_used < promotion_budget and selected:
        cutoff_item = selected[-1]
        eligible: list[tuple[tuple, dict, dict]] = []
        for candidate in memory_results:
            row_id = candidate.get("id")
            if not isinstance(row_id, int) or row_id in selected_ids:
                continue
            candidate_rank = candidate_rank_by_row_id.get(row_id)
            if not isinstance(candidate_rank, int) or candidate_rank <= 0:
                continue
            trace = _chroma_live_merge_trace_h18_5(candidate, cutoff_item, candidate_rank=candidate_rank)
            if trace:
                eligible.append((trace["_priority"], candidate, trace))

        if not eligible:
            break

        _priority, promoted_candidate, trace = max(eligible, key=lambda entry: entry[0])
        promoted_item = dict(promoted_candidate)
        for key, value in trace.items():
            if not key.startswith("_"):
                promoted_item[key] = value

        demoted_item = selected.pop()
        demoted_id = demoted_item.get("id") if isinstance(demoted_item.get("id"), int) else None
        if demoted_id is not None:
            selected_ids.discard(demoted_id)

        selected.append(promoted_item)
        promoted_id = promoted_item.get("id") if isinstance(promoted_item.get("id"), int) else None
        if promoted_id is not None:
            selected_ids.add(promoted_id)
        selected.sort(key=lambda item: _coerce_float_hy1(item.get("final_score")), reverse=True)
        promotions_used += 1

    return selected


def _normalize_kg_actor_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"[^0-9a-z가-힣\u3040-\u30ff\u4e00-\u9fff]+", "", text.casefold())


_PLAYER_REFERENCE_KEYS = frozenset(
    _normalize_kg_actor_key(label)
    for label in (
        _PLAYER_ENTITY_TOKEN,
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
)

_NON_ENTITY_ROLE_KEYS = frozenset(
    _normalize_kg_actor_key(label)
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


def _normalize_kg_entity_label(value: object, *, allow_player_token: bool = True) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if _RISU_USER_PLACEHOLDER_PATTERN.match(text):
        return _PLAYER_ENTITY_TOKEN if allow_player_token else ""
    if _RISU_CHAR_PLACEHOLDER_PATTERN.match(text):
        return ""
    key = _normalize_kg_actor_key(text)
    if not key:
        return ""
    if key in _PLAYER_REFERENCE_KEYS:
        return _PLAYER_ENTITY_TOKEN if allow_player_token else ""
    if key in _NON_ENTITY_ROLE_KEYS:
        return ""
    return text[:255]


def _normalize_embedding_provider(value: str | None) -> str:
    provider = str(value or "openai").strip().lower()
    return provider if provider in {"openai", "gemini", "vertex", "voyageai", "custom"} else "openai"


def _resolve_embedding_endpoint(provider: str, endpoint: str, model: str) -> str:
    endpoint_text = str(endpoint or "").strip().rstrip("/")
    model_text = str(model or "").strip()
    if provider == "openai":
        base = endpoint_text or "https://api.openai.com"
        return base if base.endswith("/v1/embeddings") else base + "/v1/embeddings"
    if provider == "voyageai":
        base = endpoint_text or "https://api.voyageai.com"
        return base if base.endswith("/v1/embeddings") else base + "/v1/embeddings"
    if provider == "gemini":
        base = endpoint_text or "https://generativelanguage.googleapis.com/v1beta"
        if re.search(r"generativelanguage\.googleapis\.com", base, re.IGNORECASE):
            base = base.rstrip("/")
            if not re.search(r"/v[0-9][^/]*$", base, re.IGNORECASE) and not re.search(r"/v[0-9][^/]*/models/", base, re.IGNORECASE):
                base += "/v1beta"
            if re.search(r":embedContent$", base, re.IGNORECASE):
                return base
            if re.search(r"/models/[^/:]+$", base, re.IGNORECASE):
                return f"{base}:embedContent"
            if "/models/" in base:
                return base
            return f"{base}/models/{model_text}:embedContent"
        if re.search(r":[a-zA-Z]+$", base):
            return base
        return f"{base}/models/{model_text}:embedContent"
    return endpoint_text


def _extract_embedding_vector(data: dict | None) -> list[float] | None:
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("embedding"), list):
        return data["embedding"]
    if isinstance(data.get("data"), list) and data["data"]:
        first = data["data"][0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list):
            return first["embedding"]
    if isinstance(data.get("embeddings"), list) and data["embeddings"]:
        first = data["embeddings"][0]
        if isinstance(first, list):
            return first
        if isinstance(first, dict):
            values = first.get("values") or first.get("embedding")
            if isinstance(values, list):
                return values
    predictions = data.get("predictions")
    if isinstance(predictions, list) and predictions:
        first = predictions[0]
        if isinstance(first, dict):
            values = ((first.get("embeddings") or {}).get("values") if isinstance(first.get("embeddings"), dict) else None)
            if isinstance(values, list):
                return values
    return None


def _base64url(data: bytes) -> str:
    import base64
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


def _safe_session_id(chat_session_id: str | None) -> str:
    """chat_session_id가 비어 있거나 None이면 안전한 fallback 값을 반환한다."""
    if chat_session_id and isinstance(chat_session_id, str) and chat_session_id.strip():
        return chat_session_id.strip()
    return DEFAULT_SESSION_ID


class ArchiveBridge:
    # ------------------------------------------------------------------ #
    #  내부 헬퍼
    # ------------------------------------------------------------------ #

    def _get_embedding(self, text: str) -> list[float] | None:
        """Project Embedding LLM API를 호출해 벡터를 반환한다.
        설정 누락 또는 호출 실패 시 None을 반환하며, 예외를 밖으로 던지지 않는다.
        """
        provider = _normalize_embedding_provider(getattr(settings, "PROJECT_EMBEDDING_PROVIDER", "openai"))
        endpoint = settings.PROJECT_EMBEDDING_ENDPOINT
        api_key = settings.PROJECT_EMBEDDING_API_KEY or settings.PROJECT_MAIN_API_KEY
        model = settings.PROJECT_EMBEDDING_MODEL

        if not api_key or not model:
            logger.warning("Embedding 설정이 비어 있어 건너뜁니다.")
            return None

        try:
            url = _resolve_embedding_endpoint(provider, endpoint or settings.PROJECT_MAIN_ENDPOINT, model)
            if not url:
                logger.warning("Embedding endpoint가 비어 있어 건너뜁니다. provider=%s", provider)
                return None

            with httpx.Client(timeout=settings.EMBEDDING_TIMEOUT) as client:
                headers = {"Content-Type": "application/json"}
                payload = {"model": model, "input": text}
                if provider in {"openai", "custom"}:
                    headers["Authorization"] = f"Bearer {api_key}"
                elif provider == "voyageai":
                    headers["Authorization"] = f"Bearer {api_key}"
                    payload = {"model": model, "input": [text]}
                elif provider == "gemini":
                    headers["x-goog-api-key"] = api_key
                    payload = {
                        "model": f"models/{model}",
                        "content": {"parts": [{"text": text}]},
                    }
                elif provider == "vertex":
                    access_token = _get_vertex_access_token(client, api_key)
                    headers["Authorization"] = f"Bearer {access_token}"
                    payload = {
                        "instances": [{"content": text}],
                        "parameters": {"autoTruncate": True},
                    }

                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            embedding = _extract_embedding_vector(data)
            if isinstance(embedding, list):
                return embedding

            logger.warning("Embedding 응답에서 벡터를 찾지 못했습니다. 키: %s", list(data.keys()))
            return None

        except Exception as exc:
            logger.warning("Embedding 호출 실패: %s", exc)
            return None

    def _cosine(self, vec1, vec2) -> float:
        """두 벡터의 코사인 유사도를 반환한다. 계산 불가 시 0.0."""
        try:
            a = np.array(vec1, dtype=float)
            b = np.array(vec2, dtype=float)
            if a.size == 0 or b.size == 0:
                return 0.0
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            return 0.0

    def _extract_archive_hint(self, summary_json: dict) -> dict:
        archive_hint = summary_json.get("archive_hint")
        if isinstance(archive_hint, dict):
            return archive_hint

        library_hint = summary_json.get("library_hint")
        if isinstance(library_hint, dict):
            return library_hint

        palace_hint = summary_json.get("palace_hint")
        if isinstance(palace_hint, dict):
            return palace_hint

        return {}

    # ------------------------------------------------------------------ #
    #  저장
    # ------------------------------------------------------------------ #

    def save_memory(
        self,
        turn_index: int,
        summary_json: dict,
        wing: str | None = None,
        room: str | None = None,
        chat_session_id: str | None = None,
    ) -> None:
        """memories 테이블에 요약 JSON + embedding을 저장한다."""
        session_id = _safe_session_id(chat_session_id)

        try:
            importance = float(summary_json.get("importance_score", 5))
        except (ValueError, TypeError):
            importance = 5.0

        archive_hint = self._extract_archive_hint(summary_json)
        archive_wing = wing or archive_hint.get("wing") or "wing_general"
        archive_room = room or archive_hint.get("room") or "room_events"

        embed_text = (
            summary_json.get("turn_summary")
            or json.dumps(summary_json, ensure_ascii=False)
        )
        embedding_vec = self._get_embedding(embed_text)
        embedding_str = json.dumps(embedding_vec) if embedding_vec is not None else None
        current_embedding_model = (settings.PROJECT_EMBEDDING_MODEL or "").strip() or None

        db: Session = SessionLocal()
        try:
            row = Memory(
                chat_session_id=session_id,
                turn_index=turn_index,
                summary_json=json.dumps(summary_json, ensure_ascii=False),
                embedding=embedding_str,
                embedding_model=current_embedding_model,
                importance=importance,
            )
            row.archive_wing = archive_wing
            row.archive_room = archive_room
            db.add(row)
            db.commit()
            self._upsert_chroma_shadow_document_h18_5(
                session_id=session_id,
                source_row_id=row.id,
                embedding_vec=embedding_vec,
                document_text=embed_text,
            )
            logger.info("ArchiveBridge.save_memory complete (turn=%d, session=%s, wing=%s)", turn_index, session_id, archive_wing)
        except Exception as exc:
            db.rollback()
            logger.error("ArchiveBridge.save_memory failed: %s", exc)
        finally:
            db.close()

    def save_kg_triples(self, triples: list[dict], source_turn: int, chat_session_id: str | None = None) -> None:
        """kg_triples 테이블에 저장한다. triples가 비어 있으면 조용히 종료."""
        if not triples:
            return

        session_id = _safe_session_id(chat_session_id)

        db: Session = SessionLocal()
        try:
            for triple in triples:
                subject = _normalize_kg_entity_label(triple.get("subject"), allow_player_token=True)
                predicate = str(triple.get("predicate", "") or "").strip()[:255]
                obj = _normalize_kg_entity_label(triple.get("object"), allow_player_token=True)
                if not subject or not predicate or not obj:
                    continue
                row = KGTriple(
                    chat_session_id=session_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    valid_from=triple.get("valid_from"),
                    valid_to=triple.get("valid_to"),
                    source_turn=source_turn,
                )
                db.add(row)
            db.commit()
            logger.info("ArchiveBridge.save_kg_triples complete (turn=%d, session=%s, count=%d)", source_turn, session_id, len(triples))
        except Exception as exc:
            db.rollback()
            logger.error("ArchiveBridge.save_kg_triples failed: %s", exc)
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    #  조회
    # ------------------------------------------------------------------ #

    def get_verbatim_by_turn(self, turn_index: int, chat_session_id: str | None = None) -> dict | None:
        """chat_logs에서 해당 turn의 원문을 반환한다. 없으면 None."""
        session_id = _safe_session_id(chat_session_id)

        db: Session = SessionLocal()
        try:
            rows = (
                db.query(ChatLog)
                .filter(ChatLog.chat_session_id == session_id)
                .filter(ChatLog.turn_index == turn_index)
                .order_by(ChatLog.id)
                .all()
            )
            if not rows:
                return None
            return {
                "turn_index": turn_index,
                "messages": [{"role": row.role, "content": row.content} for row in rows],
            }
        except Exception as exc:
            logger.error("ArchiveBridge.get_verbatim_by_turn failed: %s", exc)
            return None
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    #  Scoring 헬퍼
    # ------------------------------------------------------------------ #

    def _compute_recency_score(self, created_at, now: datetime) -> float:
        """Exponential decay 기반 recency score (0 ~ 1). half-life 이후 0.5."""
        if created_at is None:
            return 0.5

        try:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)

            # tz-naive 비교 통일
            if created_at.tzinfo is not None:
                created_at = created_at.replace(tzinfo=None)
            if now.tzinfo is not None:
                now = now.replace(tzinfo=None)

            age_hours = max(0.0, (now - created_at).total_seconds() / 3600.0)
            half_life = settings.RECENCY_HALF_LIFE_HOURS
            floor = settings.RECENCY_FLOOR

            score = 2.0 ** (-age_hours / half_life) if half_life > 0 else 1.0
            return max(floor, min(1.0, score))
        except Exception:
            return 0.5

    @staticmethod
    def _normalize_importance(importance) -> float:
        """importance (1-10 scale) → 0-1 정규화."""
        if importance is None:
            return 0.5
        try:
            return max(0.0, min(1.0, (float(importance) - 1.0) / 9.0))
        except (ValueError, TypeError):
            return 0.5

    @staticmethod
    def _parse_embedding_vector(value: object) -> list[float] | None:
        if value is None or value == "":
            return None
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
        except Exception:
            return None
        return parsed if isinstance(parsed, list) else None

    def _build_memory_retrieval_state(self, row: Memory, query_vec: list[float] | None) -> dict:
        row_vec = self._parse_embedding_vector(getattr(row, "embedding", None))
        provenance = assess_memory_embedding_state(
            embedding_value=row_vec,
            embedding_model=getattr(row, "embedding_model", None),
            current_model=settings.PROJECT_EMBEDDING_MODEL,
        )

        raw_similarity = self._cosine(query_vec, row_vec) if query_vec is not None and row_vec is not None else 0.0
        status = provenance.get("status")

        if row_vec is not None and query_vec is not None and status in {"current_model_match", "project_model_unset"}:
            retrieval_mode = "embedding_current"
            similarity = raw_similarity
        elif row_vec is not None and query_vec is not None and status in {"model_mismatch", "missing_embedding_model"}:
            retrieval_mode = "hybrid_degrade"
            similarity = raw_similarity * _STEP13_RETRIEVAL_DEGRADED_SIMILARITY_WEIGHT
        else:
            retrieval_mode = "importance_only"
            similarity = 0.0

        return {
            "raw_similarity": round(raw_similarity, 4),
            "similarity": round(similarity, 4),
            "retrieval_mode": retrieval_mode,
            "embedding_provenance_status": provenance.get("status"),
            "needs_reembed": provenance.get("needs_reembed"),
            "provenance_policy_version": provenance.get("provenance_policy_version"),
            "needs_reembed_policy_version": provenance.get("needs_reembed_policy_version"),
            "current_project_embedding_model": provenance.get("current_model"),
            "stored_embedding_model": provenance.get("stored_model"),
        }

    # ------------------------------------------------------------------ #
    #  검색
    # ------------------------------------------------------------------ #

    SIMILARITY_THRESHOLD = 0.65

    def _open_chroma_live_collection_h18_5(self):
        persist_dir = _resolve_chroma_shadow_persist_dir_h18_5()
        writable_probe = persist_dir if persist_dir.exists() else persist_dir.parent
        if not os.access(writable_probe, os.W_OK):
            raise PermissionError("persist_directory_not_writable")

        persist_dir.mkdir(parents=True, exist_ok=True)

        import chromadb as _chromadb  # type: ignore

        try:
            from chromadb.config import Settings as _ChromaSettings  # type: ignore

            client = _chromadb.PersistentClient(
                path=str(persist_dir),
                settings=_ChromaSettings(anonymized_telemetry=False),
            )
        except Exception:
            client = _chromadb.PersistentClient(path=str(persist_dir))

        return client.get_or_create_collection(name=_resolve_chroma_shadow_collection_name_h18_5())

    def _query_chroma_live_candidates_h18_5(
        self,
        *,
        session_id: str,
        query_vec: list[float] | None,
        mode: str,
        candidate_limit: int,
    ) -> dict:
        contract = {
            "contract_version": _STEP18_5_CHROMA_LIVE_CUTOVER_CONTRACT_VERSION,
            "mode": mode,
            "candidate_limit": candidate_limit,
            "candidate_count": 0,
            "hydrated_count": 0,
            "candidate_row_ids": [],
            "candidate_distance_by_row_id": {},
            "candidate_rank_by_row_id": {},
            "collection_name": _resolve_chroma_shadow_collection_name_h18_5(),
            "state": "off",
            "reason": None,
        }

        if mode not in {"limited_memory", "primary_memory"}:
            return contract
        if not bool(getattr(settings, "CHROMA_SHADOW_ENABLED", False)):
            contract["state"] = "shadow_disabled"
            contract["reason"] = "shadow_runtime_disabled"
            return contract
        if not str(getattr(settings, "PROJECT_EMBEDDING_MODEL", "") or "").strip():
            contract["state"] = "blocked"
            contract["reason"] = "embedding_model_missing"
            return contract
        if not query_vec:
            contract["state"] = "degraded"
            contract["reason"] = "query_embedding_unavailable"
            return contract

        try:
            collection = self._open_chroma_live_collection_h18_5()
        except ImportError:
            contract["state"] = "blocked"
            contract["reason"] = "chromadb_dependency_unavailable"
            return contract
        except PermissionError:
            contract["state"] = "blocked"
            contract["reason"] = "persist_directory_not_writable"
            return contract
        except Exception as exc:
            contract["state"] = "degraded"
            contract["reason"] = f"collection_open_failed:{exc.__class__.__name__}"
            return contract

        try:
            payload = collection.query(
                query_embeddings=[list(query_vec)],
                n_results=candidate_limit,
                where={
                    "$and": [
                        {"chat_session_id": session_id},
                        {"tier": "memory"},
                    ]
                },
                include=["metadatas", "distances"],
            )
        except Exception as exc:
            contract["state"] = "degraded"
            contract["reason"] = f"query_failed:{exc.__class__.__name__}"
            return contract

        metadatas_bucket = list((payload or {}).get("metadatas") or [])
        distances_bucket = list((payload or {}).get("distances") or [])
        metadatas = list(metadatas_bucket[0] if metadatas_bucket else [])
        distances = list(distances_bucket[0] if distances_bucket else [])

        candidate_row_ids: list[int] = []
        candidate_distance_by_row_id: dict[int, float] = {}
        for index, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("tier") or "").strip().lower() != "memory":
                continue
            if str(metadata.get("source_table") or "").strip().lower() != "memories":
                continue
            try:
                source_row_id = int(metadata.get("source_row_id"))
            except Exception:
                continue
            if source_row_id <= 0 or source_row_id in candidate_row_ids:
                continue
            candidate_row_ids.append(source_row_id)
            if index < len(distances):
                try:
                    candidate_distance_by_row_id[source_row_id] = round(float(distances[index]), 4)
                except Exception:
                    pass

        contract["candidate_row_ids"] = candidate_row_ids
        contract["candidate_count"] = len(candidate_row_ids)
        contract["candidate_distance_by_row_id"] = candidate_distance_by_row_id
        if not candidate_row_ids:
            contract["state"] = "empty"
            contract["reason"] = "no_memory_candidates"
            return contract

        contract["state"] = "candidate_ready"
        return contract

    def _build_chroma_live_item_fields_h18_5(
        self,
        *,
        live_contract: dict,
        row_id: int | None,
    ) -> dict:
        candidate_rank_by_row_id = live_contract.get("candidate_rank_by_row_id") or {}
        candidate_distance_by_row_id = live_contract.get("candidate_distance_by_row_id") or {}
        candidate_rank = candidate_rank_by_row_id.get(row_id) if isinstance(row_id, int) else None
        candidate_distance = candidate_distance_by_row_id.get(row_id) if isinstance(row_id, int) else None

        fields = {
            "chroma_live_contract_version": live_contract.get("contract_version"),
            "chroma_live_state": live_contract.get("state"),
            "chroma_live_mode": live_contract.get("mode"),
            "chroma_live_candidate_limit": live_contract.get("candidate_limit"),
            "chroma_live_candidate_count": live_contract.get("candidate_count"),
            "chroma_live_hydrated_count": live_contract.get("hydrated_count"),
            "chroma_live_hydrated": candidate_rank is not None,
            "chroma_live_reason": live_contract.get("reason"),
            "chroma_live_truth_authority": "sqlite",
            "chroma_live_supporting_only": True,
            "chroma_live_support_surface_mode": _STEP18_5_CHROMA_LIVE_SUPPORT_SURFACE_MODE,
            "chroma_live_cannot_override_current_user_input": True,
            "chroma_live_cannot_override_verified_direct_evidence": True,
            "chroma_live_precedence_ceiling": _STEP18_5_CHROMA_LIVE_PRECEDENCE_CEILING,
        }
        if isinstance(row_id, int):
            fields["chroma_live_hydration_source_table"] = "memories"
            fields["chroma_live_hydration_source_row_id"] = row_id
        if isinstance(candidate_rank, int):
            fields["chroma_live_candidate_rank"] = candidate_rank
        if candidate_distance is not None:
            try:
                fields["chroma_live_distance"] = round(float(candidate_distance), 4)
            except Exception:
                pass
        if isinstance(live_contract.get("merge_candidate_limit"), int):
            fields["chroma_live_merge_candidate_limit"] = live_contract.get("merge_candidate_limit")
        return fields

    def search_memories(
        self,
        query: str,
        wing: str | None = None,
        n_results: int = 5,
        chat_session_id: str | None = None,
    ) -> list[dict]:
        """1??? memory retrieval. Chroma is primary candidate source; SQLite hydrates and verifies."""
        if not query or not str(query).strip():
            return []
        query = str(query).strip()
        n_results = max(1, min(50, n_results)) if isinstance(n_results, int) and n_results > 0 else 5
        session_id = _safe_session_id(chat_session_id)

        query_vec = self._get_embedding(query)
        live_mode = _normalize_chroma_live_cutover_mode_h18_5(getattr(settings, "CHROMA_LIVE_CUTOVER_MODE", "primary_memory"))
        live_candidate_limit = min(
            n_results,
            _normalize_chroma_live_candidate_limit_h18_5(
                getattr(settings, "CHROMA_LIVE_CANDIDATE_LIMIT", n_results),
                fallback=n_results,
            ),
        )
        live_contract = self._query_chroma_live_candidates_h18_5(
            session_id=session_id,
            query_vec=query_vec,
            mode=live_mode,
            candidate_limit=live_candidate_limit,
        )

        # --- Primary Memory path: fetch only Chroma candidate rows ---
        if live_mode == "primary_memory" and live_contract.get("state") == "candidate_ready":
            candidate_ids = list(live_contract.get("candidate_row_ids") or [])
            if candidate_ids:
                db: Session = SessionLocal()
                try:
                    db_query = (
                        db.query(Memory)
                        .filter(Memory.chat_session_id == session_id)
                        .filter(Memory.id.in_(candidate_ids))
                    )
                    if wing:
                        db_query = db_query.filter(Memory.archive_wing == wing)
                    candidate_rows = db_query.all()
                except Exception as exc:
                    logger.error("ArchiveBridge.search_memories primary hydration failed: %s", exc)
                    candidate_rows = []
                finally:
                    db.close()

                # Hydrate ranking by candidate order
                rows_by_id = {row.id: row for row in candidate_rows if isinstance(getattr(row, "id", None), int)}
                hydrated_ids = [rid for rid in candidate_ids if rid in rows_by_id]
                live_contract["candidate_rank_by_row_id"] = {
                    rid: idx + 1 for idx, rid in enumerate(hydrated_ids)
                }
                live_contract["hydrated_count"] = len(hydrated_ids)

                if hydrated_ids:
                    live_contract["state"] = "chroma_engaged"
                    memory_results = self._build_scored_memory_results(
                        rows=candidate_rows,
                        query=query,
                        query_vec=query_vec,
                        live_contract=live_contract,
                        n_results=n_results,
                    )
                    if memory_results:
                        return memory_results
                    # Empty scored results: fall through to fallback
                else:
                    live_contract["state"] = "sqlite_fallback"
                    live_contract["reason"] = "sqlite_hydration_empty"
            # If no candidates or hydration empty, fall through to full SQLite fallback

        # --- Legacy limited_memory path or primary fallback: fetch all SQLite rows ---
        db: Session = SessionLocal()
        try:
            db_query = db.query(Memory).filter(Memory.chat_session_id == session_id)
            if wing:
                db_query = db_query.filter(Memory.archive_wing == wing)
            rows = db_query.all()
        except Exception as exc:
            logger.error("ArchiveBridge.search_memories DB query failed: %s", exc)
            return []
        finally:
            db.close()

        rows_by_id = {
            row.id: row
            for row in rows
            if isinstance(getattr(row, "id", None), int)
        }
        if live_contract.get("state") == "candidate_ready":
            candidate_rank_by_row_id: dict[int, int] = {}
            hydrated_row_ids: list[int] = []
            for source_row_id in list(live_contract.get("candidate_row_ids") or []):
                if source_row_id in rows_by_id and source_row_id not in candidate_rank_by_row_id:
                    hydrated_row_ids.append(source_row_id)
                    candidate_rank_by_row_id[source_row_id] = len(hydrated_row_ids)

            live_contract["candidate_rank_by_row_id"] = candidate_rank_by_row_id
            live_contract["hydrated_count"] = len(hydrated_row_ids)
            live_contract["merge_candidate_limit"] = min(
                len(hydrated_row_ids),
                max(0, int(_HY1_POLICY.tail_budget_policy.max_promoted_items or 0)),
            )
            if hydrated_row_ids:
                live_contract["state"] = "chroma_engaged"
            else:
                live_contract["state"] = "sqlite_fallback"
                live_contract["reason"] = "sqlite_hydration_empty"

        memory_results = self._build_scored_memory_results(
            rows=rows,
            query=query,
            query_vec=query_vec,
            live_contract=live_contract,
            n_results=n_results,
            apply_merge=True,
        )

        max_similarity = memory_results[0]["similarity"] if memory_results else 0.0
        needs_fallback = len(memory_results) == 0 or max_similarity < self.SIMILARITY_THRESHOLD

        if needs_fallback:
            fallback_results = self._fallback_verbatim(
                query,
                limit=n_results,
                chat_session_id=session_id,
                live_contract=live_contract,
            )
            return memory_results + fallback_results

        return memory_results

    def _build_scored_memory_results(
        self,
        rows: list,
        query: str,
        query_vec: list[float] | None,
        live_contract: dict,
        n_results: int,
        apply_merge: bool = False,
    ) -> list[dict]:
        """Build, score, and return memory results from SQLite rows."""
        memory_results: list[dict] = []
        for row in rows:
            retrieval_state = self._build_memory_retrieval_state(row, query_vec)
            summary_json = _safe_load_summary_json_hy1(row.summary_json)
            archive_hint = self._extract_archive_hint(summary_json)
            hybrid_signal = _build_hybrid_signal_hy1(
                query=query,
                summary_json=summary_json,
                archive_hint=archive_hint,
                archive_wing=row.archive_wing,
                archive_room=row.archive_room,
            )
            memory_results.append(
                {
                    "id": row.id,
                    "chat_session_id": row.chat_session_id,
                    "turn_index": row.turn_index,
                    "summary_json": row.summary_json,
                    "importance": row.importance,
                    "archive_wing": row.archive_wing,
                    "archive_room": row.archive_room,
                    "library_wing": row.archive_wing,
                    "library_room": row.archive_room,
                    "similarity": retrieval_state["similarity"],
                    "raw_similarity": retrieval_state["raw_similarity"],
                    "retrieval_mode": retrieval_state["retrieval_mode"],
                    "retrieval_fallback_policy_version": STEP13_RETRIEVAL_FALLBACK_POLICY_VERSION,
                    "embedding_provenance_status": retrieval_state["embedding_provenance_status"],
                    "needs_reembed": retrieval_state["needs_reembed"],
                    "provenance_policy_version": retrieval_state["provenance_policy_version"],
                    "needs_reembed_policy_version": retrieval_state["needs_reembed_policy_version"],
                    "current_project_embedding_model": retrieval_state["current_project_embedding_model"],
                    "stored_embedding_model": retrieval_state["stored_embedding_model"],
                    "embedding_model": getattr(row, "embedding_model", None),
                    "created_at": to_kst(row.created_at),
                    "source": "memory",
                    **hybrid_signal,
                    **self._build_chroma_live_item_fields_h18_5(live_contract=live_contract, row_id=row.id),
                }
            )

        now = datetime.now()
        w_sim = settings.SCORING_SIMILARITY_WEIGHT
        w_imp = settings.SCORING_IMPORTANCE_WEIGHT
        w_rec = settings.SCORING_RECENCY_WEIGHT

        for item in memory_results:
            imp_norm = self._normalize_importance(item["importance"])
            rec_score = self._compute_recency_score(item.get("created_at"), now)
            item["importance_norm"] = round(imp_norm, 4)
            item["recency_score"] = round(rec_score, 4)
            item["semantic_rank_score"] = round(
                w_sim * item["similarity"] + w_imp * imp_norm + w_rec * rec_score, 4
            )
            item["hybrid_baseline_score"] = round(
                float(item.get("semantic_rank_score") or 0.0) + float(item.get("keyword_overlap_boost") or 0.0),
                4,
            )
            item["final_score"] = round(
                float(item.get("hybrid_baseline_score") or 0.0) + float(item.get("soft_bias_score") or 0.0),
                4,
            )
            item.pop("keyword_overlap_boost", None)

        memory_results.sort(key=lambda item: item["final_score"], reverse=True)
        top_memories = _apply_tail_budget_rescue_hy1(memory_results, n_results)
        if apply_merge:
            top_memories = _apply_chroma_live_merge_h18_5(memory_results, top_memories, live_contract=live_contract)
        return top_memories

    def _fallback_verbatim(
        self,
        query: str,
        limit: int = 5,
        chat_session_id: str | None = None,
        live_contract: dict | None = None,
    ) -> list[dict]:
        """chat_logs에서 keyword LIKE 기반 fallback 검색. 최근순 정렬. session scope."""
        if not query or not str(query).strip():
            return []

        session_id = _safe_session_id(chat_session_id)

        db: Session = SessionLocal()
        try:
            keywords = [keyword.strip() for keyword in str(query).split() if keyword.strip()]
            if not keywords:
                return []

            from sqlalchemy import or_

            conditions = [ChatLog.content.like(f"%{keyword}%") for keyword in keywords]
            rows = (
                db.query(ChatLog)
                .filter(ChatLog.chat_session_id == session_id)
                .filter(or_(*conditions))
                .order_by(ChatLog.turn_index.desc(), ChatLog.id.desc())
                .limit(limit)
                .all()
            )

            results: list[dict] = []
            for row in rows:
                item = {
                    "id": row.id,
                    "chat_session_id": row.chat_session_id,
                    "turn_index": row.turn_index,
                    "role": row.role,
                    "content": row.content,
                    "source": "chat_log",
                }
                if isinstance(live_contract, dict):
                    item.update(self._build_chroma_live_item_fields_h18_5(live_contract=live_contract, row_id=None))
                results.append(item)
            return results
        except Exception as exc:
            logger.error("ArchiveBridge._fallback_verbatim failed: %s", exc)
            return []
        finally:
            db.close()

    # ------------------------------------------------------------------ #
    #  삭제 (rollback용)
    # ------------------------------------------------------------------ #

    def delete_after_turn(self, from_turn_index: int, chat_session_id: str | None = None) -> None:
        """from_turn_index ???????? session ??????????? (rollback??. SQLite + Chroma shadow ??????."""
        if from_turn_index is None or not isinstance(from_turn_index, int) or from_turn_index < 0:
            logger.warning("ArchiveBridge.delete_after_turn invalid turn_index=%s, skipped", from_turn_index)
            return

        session_id = _safe_session_id(chat_session_id)

        # Collect candidate Memory row IDs before SQLite delete so we can prune Chroma
        candidate_row_ids: list[int] = []
        db: Session = SessionLocal()
        try:
            candidate_rows = (
                db.query(Memory.id)
                .filter(Memory.chat_session_id == session_id)
                .filter(Memory.turn_index >= from_turn_index)
                .all()
            )
            candidate_row_ids = [int(row.id) for row in candidate_rows if getattr(row, "id", None) is not None]
        except Exception as exc:
            logger.warning("ArchiveBridge.delete_after_turn candidate ID collection failed (non-fatal): %s", exc)
        finally:
            db.close()

        db = SessionLocal()
        try:
            deleted_logs = (
                db.query(ChatLog)
                .filter(ChatLog.chat_session_id == session_id)
                .filter(ChatLog.turn_index >= from_turn_index)
                .delete()
            )
            deleted_mems = (
                db.query(Memory)
                .filter(Memory.chat_session_id == session_id)
                .filter(Memory.turn_index >= from_turn_index)
                .delete()
            )
            deleted_kg = (
                db.query(KGTriple)
                .filter(KGTriple.chat_session_id == session_id)
                .filter(KGTriple.source_turn >= from_turn_index)
                .delete()
            )
            db.commit()
            logger.info(
                "ArchiveBridge.delete_after_turn(%d, session=%s) complete ??logs:%d, memories:%d, kg:%d",
                from_turn_index,
                session_id,
                deleted_logs,
                deleted_mems,
                deleted_kg,
            )
        except Exception as exc:
            db.rollback()
            logger.error("ArchiveBridge.delete_after_turn failed: %s", exc)
            return
        finally:
            db.close()

        # Prune Chroma shadow documents for deleted Memory IDs
        if candidate_row_ids:
            self._prune_chroma_shadow_documents(
                session_id=session_id,
                source_row_ids=candidate_row_ids,
                tier="memory",
            )

    def _upsert_chroma_shadow_document_h18_5(
        self,
        *,
        session_id: str,
        source_row_id: int,
        embedding_vec: list[float] | None,
        document_text: str,
    ) -> None:
        """Upsert a Chroma shadow document for a saved Memory row. Fail-open."""
        if not embedding_vec:
            logger.debug(
                "Chroma shadow upsert skipped: no embedding (session=%s, row=%s)", session_id, source_row_id
            )
            return
        try:
            collection = self._open_chroma_live_collection_h18_5()
        except Exception as exc:
            logger.debug("Chroma shadow upsert skipped: collection unavailable (%s)", exc)
            return

        chroma_doc_id = f"mem_{source_row_id}"
        try:
            collection.upsert(
                ids=[chroma_doc_id],
                embeddings=[list(embedding_vec)],
                metadatas=[{
                    "chat_session_id": session_id,
                    "tier": "memory",
                    "source_table": "memories",
                    "source_row_id": source_row_id,
                }],
                documents=[str(document_text or "")],
            )
            logger.debug(
                "Chroma shadow upsert done (session=%s, row=%s, doc_id=%s)",
                session_id,
                source_row_id,
                chroma_doc_id,
            )
        except Exception as exc:
            logger.warning("Chroma shadow upsert failed (non-fatal): %s", exc)

    def _prune_chroma_shadow_documents(
        self,
        *,
        session_id: str,
        source_row_ids: list[int],
        tier: str = "memory",
    ) -> None:
        """Remove Chroma shadow documents whose metadata matches the given source_row_ids. Fail-open."""
        if not source_row_ids:
            return
        try:
            collection = self._open_chroma_live_collection_h18_5()
        except Exception as exc:
            logger.debug("Chroma shadow prune skipped: collection unavailable (%s)", exc)
            return

        try:
            for rid in source_row_ids:
                try:
                    collection.delete(
                        where={
                            "$and": [
                                {"chat_session_id": session_id},
                                {"source_row_id": rid},
                                {"tier": tier},
                            ]
                        }
                    )
                except Exception as inner:
                    logger.debug("Chroma shadow delete for row %s failed (non-fatal): %s", rid, inner)
            logger.info(
                "Chroma shadow prune done (session=%s, tier=%s, attempted=%d)",
                session_id,
                tier,
                len(source_row_ids),
            )
        except Exception as exc:
            logger.warning("Chroma shadow prune failed (non-fatal): %s", exc)

