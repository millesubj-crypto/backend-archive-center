import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel


_COPILOT_CODE_VERSION = "1.85.0"
_COPILOT_CHAT_VERSION = "0.22.0"
_COPILOT_USER_AGENT = f"GitHubCopilotChat/{_COPILOT_CHAT_VERSION}"
_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
_VERTEX_TOKEN_CACHE: dict[str, tuple[str, float]] = {}


class ProxyPluginMainRequest(BaseModel):
    messages: list
    model: str = ""
    max_tokens: int = 1024
    temperature: float = 0.7
    provider: str | None = None
    max_completion_tokens: int | None = None
    reasoning_preset: str | None = None
    reasoning_effort: str | None = None
    reasoning_budget_tokens: int | None = None
    budget_tokens: int | None = None
    glm_thinking_type: str | None = None
    endpoint: str = ""
    api_key: str = ""
    timeout_ms: int | None = None


@dataclass
class ProxyPluginMainResult:
    content: dict
    status_code: int = 200


def _proxy_is_glm_like(model: str, endpoint: str, provider: str) -> bool:
    model_lc = str(model or "").strip().lower()
    endpoint_lc = str(endpoint or "").strip().lower()
    provider_lc = str(provider or "").strip().lower()
    return bool(
        re.match(r"^glm[-\d.]", model_lc)
        or re.search(r"(?:open\.)?bigmodel\.cn|zhipu", endpoint_lc)
        or (provider_lc == "custom" and model_lc.startswith("glm"))
    )


def _proxy_resolve_reasoning_family(provider: str, reasoning_preset: str, model: str, endpoint: str) -> str:
    preset_lc = str(reasoning_preset or "auto").strip().lower()
    if preset_lc in {"gpt", "gemini", "claude", "glm"}:
        return preset_lc
    provider_lc = str(provider or "openai").strip().lower()
    if _proxy_is_glm_like(model, endpoint, provider_lc):
        return "glm"
    if provider_lc == "claude":
        return "claude"
    if provider_lc in {"gemini", "vertex"}:
        return "gemini"
    return "gpt"


def _proxy_resolve_gemini_thinking_mode(model: str) -> str:
    model_lc = str(model or "").strip().lower()
    if "gemini-2.5" in model_lc:
        return "budget"
    if re.search(r"gemini-(?:3(?:\D|$)|[4-9](?:\D|$)|\d{2,}(?:\D|$))", model_lc):
        return "level"
    return "budget"


def _proxy_openai_base_url(provider: str, endpoint: str) -> str:
    provider_lc = str(provider or "openai").strip().lower()
    endpoint_text = str(endpoint or "").strip()
    if endpoint_text:
        return endpoint_text.rstrip("/")
    if provider_lc == "openrouter":
        return "https://openrouter.ai/api"
    if provider_lc == "copilot":
        return "https://api.githubcopilot.com"
    return "https://api.openai.com"


def _proxy_normalize_gemini_endpoint(endpoint: str, model: str, action: str = "generateContent") -> str:
    normalized_action = "embedContent" if action == "embedContent" else "generateContent"
    model_text = str(model or "").strip()
    base_url = str(endpoint or "").strip() or "https://generativelanguage.googleapis.com/v1beta"
    base_url = base_url.rstrip("/")
    if not re.search(r"generativelanguage\.googleapis\.com", base_url, re.IGNORECASE):
        if re.search(r":[a-zA-Z]+$", base_url):
            return base_url
        return f"{base_url}/models/{model_text}:{normalized_action}"
    if not re.search(r"/v[0-9][^/]*$", base_url, re.IGNORECASE) and not re.search(r"/v[0-9][^/]*/models/", base_url, re.IGNORECASE):
        base_url += "/v1beta"
    if re.search(rf":{normalized_action}$", base_url, re.IGNORECASE):
        return base_url
    if re.search(r"/models/[^/:]+$", base_url, re.IGNORECASE):
        return f"{base_url}:{normalized_action}"
    if "/models/" in base_url:
        return base_url
    return f"{base_url}/models/{model_text}:{normalized_action}"


def _proxy_extract_gemini_text(candidate: dict | None) -> str:
    parts = (((candidate or {}).get("content") or {}).get("parts") or [])
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("thought"):
            continue
        text = str(part.get("text") or "").strip()
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def _proxy_normalize_chat_response(content: str, model: str, usage: dict | None = None, finish_reason: str = "stop") -> dict:
    return {
        "model": model,
        "choices": [{
            "message": {"role": "assistant", "content": content},
        }],
    }


async def _proxy_fetch_json(client: httpx.AsyncClient, url: str, *, headers: dict, body: dict) -> tuple[httpx.Response, dict | None]:
    response = await client.post(url, json=body, headers=headers)
    data = None
    try:
        data = response.json()
    except Exception:
        data = None
    return response, data


async def _proxy_get_copilot_token(client: httpx.AsyncClient, api_key: str) -> str:
    source_token = re.sub(r"[^\x20-\x7E]", "", str(api_key or "")).strip()
    if not source_token:
        return ""
    response = await client.get(
        _COPILOT_TOKEN_URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {source_token}",
            "Origin": "vscode-file://vscode-app",
            "Editor-Version": f"vscode/{_COPILOT_CODE_VERSION}",
            "Editor-Plugin-Version": f"copilot-chat/{_COPILOT_CHAT_VERSION}",
            "Copilot-Integration-Id": "vscode-chat",
            "User-Agent": _COPILOT_USER_AGENT,
        },
    )
    if not response.is_success:
        return source_token
    try:
        payload = response.json()
    except Exception:
        return source_token
    token = str((payload or {}).get("token") or "").strip()
    return token or source_token


def _proxy_base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


async def _proxy_get_vertex_access_token(client: httpx.AsyncClient, api_key: str) -> str:
    cache_key = str(api_key or "").strip()
    cached = _VERTEX_TOKEN_CACHE.get(cache_key)
    now = datetime.now(timezone.utc).timestamp()
    if cached and now < (cached[1] - 60):
        return cached[0]

    try:
        credentials = json.loads(cache_key)
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
        raise RuntimeError("cryptography package is required for Vertex AI service account auth") from exc

    issued_at = int(now)
    header = {"alg": "RS256", "typ": "JWT"}
    claim_set = {
        "iss": client_email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": "https://oauth2.googleapis.com/token",
        "exp": issued_at + 3600,
        "iat": issued_at,
    }
    signing_input = f"{_proxy_base64url(json.dumps(header, separators=(',', ':')).encode('utf-8'))}.{_proxy_base64url(json.dumps(claim_set, separators=(',', ':')).encode('utf-8'))}"
    private_key_obj = serialization.load_pem_private_key(private_key.encode("utf-8"), password=None)
    signature = private_key_obj.sign(signing_input.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
    assertion = f"{signing_input}.{_proxy_base64url(signature)}"
    response = await client.post(
        "https://oauth2.googleapis.com/token",
        content=urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        }),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if not response.is_success:
        raise RuntimeError(f"Failed to get Vertex AI access token: {response.text[:300]}")
    payload = response.json()
    token = str((payload or {}).get("access_token") or "").strip()
    if not token:
        raise RuntimeError("No access token in Vertex AI token response")
    expiry = now + max(300, int(payload.get("expires_in") or 3600))
    _VERTEX_TOKEN_CACHE[cache_key] = (token, expiry)
    return token


def _proxy_plugin_main_is_unsupported_parameter_error(detail: str) -> bool:
    text = str(detail or "").lower()
    if not text:
        return False
    return any(token in text for token in (
        "unsupported parameter",
        "unknown parameter",
        "unrecognized parameter",
        "extra fields not permitted",
        "additional properties are not allowed",
    )) or ("invalid_request_error" in text and "parameter" in text)


async def _proxy_call_openai_like(client: httpx.AsyncClient, req: ProxyPluginMainRequest, endpoint: str, api_key: str, model: str) -> dict:
    provider = str(req.provider or "openai").strip().lower()
    base_url = _proxy_openai_base_url(provider, endpoint)
    is_glm = _proxy_is_glm_like(model, endpoint, provider)
    endpoint_suffix = "/chat/completions" if provider == "copilot" or is_glm else "/v1/chat/completions"
    url = base_url if base_url.endswith(endpoint_suffix) else f"{base_url.rstrip('/')}{endpoint_suffix}"
    auth_token = api_key
    if provider == "copilot":
        auth_token = await _proxy_get_copilot_token(client, api_key)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://risuai.xyz"
        headers["X-Title"] = "Archive Center"
    elif provider == "copilot":
        headers.update({
            "Editor-Version": f"vscode/{_COPILOT_CODE_VERSION}",
            "Editor-version": f"vscode/{_COPILOT_CODE_VERSION}",
            "Editor-Plugin-Version": f"copilot-chat/{_COPILOT_CHAT_VERSION}",
            "Editor-plugin-version": f"copilot-chat/{_COPILOT_CHAT_VERSION}",
            "Copilot-Integration-Id": "vscode-chat",
            "User-Agent": _COPILOT_USER_AGENT,
            "X-Github-Api-Version": "2025-10-01",
            "X-Initiator": "user",
        })

    reasoning_family = _proxy_resolve_reasoning_family(provider, req.reasoning_preset or "auto", model, endpoint)
    requested_tokens = max(1, int(req.max_tokens or 1024))
    configured_max = max(0, int(req.max_completion_tokens or 0))
    body = {
        "model": model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": requested_tokens,
        "stream": False,
    }
    if reasoning_family == "glm":
        body["max_tokens"] = max(requested_tokens, configured_max or requested_tokens)
        body["thinking"] = {
            "type": "disabled" if str(req.glm_thinking_type or "enabled").strip().lower() == "disabled" else "enabled"
        }
    else:
        reasoning_effort = str(req.reasoning_effort or "").strip().lower()
        if reasoning_effort and reasoning_effort != "none":
            body["reasoning_effort"] = reasoning_effort
            body["max_completion_tokens"] = max(requested_tokens, configured_max or requested_tokens)
            body.pop("max_tokens", None)

    response, data = await _proxy_fetch_json(client, url, headers=headers, body=body)
    if response.status_code == 400 and any(key in body for key in ("reasoning_effort", "max_completion_tokens", "thinking")) and _proxy_plugin_main_is_unsupported_parameter_error(response.text):
        fallback_body = dict(body)
        fallback_body.pop("reasoning_effort", None)
        fallback_body.pop("max_completion_tokens", None)
        fallback_body.pop("thinking", None)
        fallback_body["max_tokens"] = requested_tokens
        response, data = await _proxy_fetch_json(client, url, headers=headers, body=fallback_body)
    if response.status_code != 200:
        raise httpx.HTTPStatusError("openai-like upstream error", request=response.request, response=response)
    if not isinstance(data, dict):
        raise RuntimeError("OpenAI-like provider returned invalid JSON")
    return data


async def _proxy_call_claude(client: httpx.AsyncClient, req: ProxyPluginMainRequest, endpoint: str, api_key: str, model: str) -> dict:
    url = endpoint.rstrip("/")
    if "/v1/" not in url:
        url = f"{url}/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body = {
        "model": model,
        "system": "\n\n".join(str(item.get("content") or "") for item in req.messages if item.get("role") == "system").strip(),
        "messages": [{
            "role": "user",
            "content": "\n\n".join(str(item.get("content") or "") for item in req.messages if item.get("role") != "system").strip(),
        }],
        "max_tokens": max(1, int(req.max_tokens or 1024)),
        "temperature": req.temperature,
        "stream": False,
    }
    budget_tokens = max(0, int(req.reasoning_budget_tokens or req.budget_tokens or 0))
    configured_max = max(0, int(req.max_completion_tokens or 0))
    if budget_tokens >= 1024:
        body["max_tokens"] = max(body["max_tokens"], configured_max or body["max_tokens"])
        body["thinking"] = {"type": "enabled", "budget_tokens": max(1024, budget_tokens)}
    response, data = await _proxy_fetch_json(client, url, headers=headers, body=body)
    if response.status_code != 200:
        raise httpx.HTTPStatusError("claude upstream error", request=response.request, response=response)
    blocks = (data or {}).get("content") or []
    parts = []
    for block in blocks:
        if isinstance(block, dict) and (block.get("type") == "text" or isinstance(block.get("text"), str)):
            text = str(block.get("text") or "").strip()
            if text:
                parts.append(text)
    content = "\n\n".join(parts).strip()
    if not content:
        raise RuntimeError("Claude returned no text content")
    return _proxy_normalize_chat_response(content, model, (data or {}).get("usage") or {})


async def _proxy_call_gemini(client: httpx.AsyncClient, req: ProxyPluginMainRequest, endpoint: str, api_key: str, model: str, *, is_vertex: bool = False) -> dict:
    requested_tokens = max(1, int(req.max_tokens or 1024))
    configured_max = max(0, int(req.max_completion_tokens or 0))
    budget_tokens = max(0, int(req.reasoning_budget_tokens or req.budget_tokens or 0))
    reasoning_effort = str(req.reasoning_effort or "").strip().lower()
    system_prompt = "\n\n".join(str(item.get("content") or "") for item in req.messages if item.get("role") == "system").strip()
    user_content = "\n\n".join(str(item.get("content") or "") for item in req.messages if item.get("role") != "system").strip()
    is_thinking_model = bool(re.search(r"gemini-(3|2\.5)", model, re.IGNORECASE))
    thinking_mode = _proxy_resolve_gemini_thinking_mode(model)
    max_output_tokens = max(requested_tokens, configured_max or requested_tokens) if is_thinking_model else requested_tokens
    body = {
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": {
            "temperature": req.temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    if is_thinking_model:
        body["generationConfig"]["thinkingConfig"] = {"includeThoughts": False}
        if thinking_mode == "level" and reasoning_effort in {"minimal", "low", "medium", "high"}:
            body["generationConfig"]["thinkingConfig"]["thinkingLevel"] = reasoning_effort
        elif budget_tokens > 0:
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"] = budget_tokens
    elif budget_tokens > 0:
        body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget_tokens}

    if is_vertex:
        access_token = await _proxy_get_vertex_access_token(client, api_key)
        base_url = endpoint.rstrip("/")
        url = base_url.replace(":streamGenerateContent", ":generateContent") if ":streamGenerateContent" in base_url else (base_url if ":generateContent" in base_url else f"{base_url}/{model}:generateContent")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
    else:
        url = _proxy_normalize_gemini_endpoint(endpoint, model, "generateContent")
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    response, data = await _proxy_fetch_json(client, url, headers=headers, body=body)
    if response.status_code != 200:
        raise httpx.HTTPStatusError("gemini upstream error", request=response.request, response=response)
    candidates = (data or {}).get("candidates") or []
    content = _proxy_extract_gemini_text(candidates[0] if candidates else None).strip()
    if not content:
        raise RuntimeError("Gemini/Vertex returned no text content")
    usage = (data or {}).get("usageMetadata") or (data or {}).get("usage") or {}
    return _proxy_normalize_chat_response(content, model, usage)


async def proxy_plugin_main_payload(*, req: ProxyPluginMainRequest, settings) -> ProxyPluginMainResult:
    endpoint = (req.endpoint or settings.PROJECT_MAIN_ENDPOINT or "").strip().rstrip("/")
    api_key = (req.api_key or settings.PROJECT_MAIN_API_KEY or "").strip()
    model = (req.model or settings.PROJECT_MAIN_MODEL or "").strip()

    if not endpoint or not api_key or not model:
        return ProxyPluginMainResult(
            status_code=400,
            content={"error": "endpoint / api_key / model is required."},
        )

    provider = str(req.provider or settings.PROJECT_MAIN_PROVIDER or "").strip().lower()
    if not provider:
        provider = "openai"

    timeout_ms = req.timeout_ms if req.timeout_ms is not None else int((settings.MAIN_TIMEOUT or 60) * 1000)
    timeout_ms = max(1, min(300000, int(timeout_ms)))

    try:
        async with httpx.AsyncClient(timeout=timeout_ms / 1000.0) as client:
            if provider == "claude":
                response_payload = await _proxy_call_claude(client, req, endpoint, api_key, model)
            elif provider == "gemini":
                response_payload = await _proxy_call_gemini(client, req, endpoint, api_key, model, is_vertex=False)
            elif provider == "vertex":
                response_payload = await _proxy_call_gemini(client, req, endpoint, api_key, model, is_vertex=True)
            else:
                response_payload = await _proxy_call_openai_like(client, req, endpoint, api_key, model)
        return ProxyPluginMainResult(content=response_payload)
    except httpx.TimeoutException:
        return ProxyPluginMainResult(status_code=504, content={"error": "upstream timeout"})
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            detail = exc.response.text[:500]
        except Exception:
            detail = str(exc)
        return ProxyPluginMainResult(
            status_code=getattr(exc.response, "status_code", 502),
            content={"error": detail or str(exc)},
        )
    except Exception as exc:
        return ProxyPluginMainResult(status_code=502, content={"error": str(exc)})
