import os
from pathlib import Path


_CONFIG_KEY_MAP: dict[str, str] = {
    "mainProvider": "PROJECT_MAIN_PROVIDER",
    "mainApiKey": "PROJECT_MAIN_API_KEY",
    "mainEndpoint": "PROJECT_MAIN_ENDPOINT",
    "mainModel": "PROJECT_MAIN_MODEL",
    "mainTimeout": "MAIN_TIMEOUT",
    "criticProvider": "PROJECT_CRITIC_PROVIDER",
    "criticApiKey": "PROJECT_CRITIC_API_KEY",
    "criticEndpoint": "PROJECT_CRITIC_ENDPOINT",
    "criticModel": "PROJECT_CRITIC_MODEL",
    "supervisorApiKey": "PROJECT_SUPERVISOR_API_KEY",
    "supervisorEndpoint": "PROJECT_SUPERVISOR_ENDPOINT",
    "supervisorModel": "PROJECT_SUPERVISOR_MODEL",
    "embeddingProvider": "PROJECT_EMBEDDING_PROVIDER",
    "embeddingApiKey": "PROJECT_EMBEDDING_API_KEY",
    "embeddingEndpoint": "PROJECT_EMBEDDING_ENDPOINT",
    "embeddingModel": "PROJECT_EMBEDDING_MODEL",
    "embeddingTimeout": "EMBEDDING_TIMEOUT",
    "supervisorTimeout": "SUPERVISOR_TIMEOUT",
    "criticTimeout": "CRITIC_TIMEOUT",
    "topK": "TOP_K_RESULTS",
    "chromaLiveCutoverMode": "CHROMA_LIVE_CUTOVER_MODE",
}

_STRING_CONFIG_ATTRS = {
    "PROJECT_MAIN_PROVIDER",
    "PROJECT_MAIN_API_KEY",
    "PROJECT_MAIN_ENDPOINT",
    "PROJECT_MAIN_MODEL",
    "PROJECT_CRITIC_PROVIDER",
    "PROJECT_CRITIC_API_KEY",
    "PROJECT_CRITIC_ENDPOINT",
    "PROJECT_CRITIC_MODEL",
    "PROJECT_SUPERVISOR_API_KEY",
    "PROJECT_SUPERVISOR_ENDPOINT",
    "PROJECT_SUPERVISOR_MODEL",
    "PROJECT_EMBEDDING_PROVIDER",
    "PROJECT_EMBEDDING_API_KEY",
    "PROJECT_EMBEDDING_ENDPOINT",
    "PROJECT_EMBEDDING_MODEL",
    "CHROMA_LIVE_CUTOVER_MODE",
}

_ENCRYPTED_CONFIG_ATTRS = {
    "PROJECT_MAIN_API_KEY",
    "PROJECT_CRITIC_API_KEY",
    "PROJECT_SUPERVISOR_API_KEY",
    "PROJECT_EMBEDDING_API_KEY",
}

_INT_CONFIG_ATTRS = {
    "MAIN_TIMEOUT",
    "EMBEDDING_TIMEOUT",
    "SUPERVISOR_TIMEOUT",
    "CRITIC_TIMEOUT",
    "TOP_K_RESULTS",
}


def _resolve_backend_env_path() -> Path:
    override = (os.getenv("BACKEND_ENV_FILE") or "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / ".env"


def _serialize_env_value(value: object) -> str:
    text = "" if value is None else str(value)
    if not text:
        return ""
    if any(ch.isspace() for ch in text) or "#" in text or '"' in text or "'" in text:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _prepare_env_persist_value(env_key: str, value: object) -> object:
    if env_key not in _ENCRYPTED_CONFIG_ATTRS:
        return value

    text = "" if value is None else str(value).strip()
    if not text:
        return ""

    try:
        from backend.key_vault import encrypt_value, is_encrypted
    except ImportError as exc:
        raise RuntimeError("cryptography package is required to persist API keys securely.") from exc

    if is_encrypted(text):
        return text
    return encrypt_value(text)


def _persist_config_to_env(updates: dict[str, object]) -> None:
    if not updates:
        return

    env_path = _resolve_backend_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    key_indexes: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key:
            key_indexes[key] = index

    for env_key, value in updates.items():
        rendered_value = _prepare_env_persist_value(env_key, value)
        rendered = f"{env_key}={_serialize_env_value(rendered_value)}"
        if env_key in key_indexes:
            lines[key_indexes[env_key]] = rendered
        else:
            lines.append(rendered)

    content = "\n".join(lines)
    env_path.write_text((content + "\n") if content else "", encoding="utf-8")


def _normalize_config_value(settings_attr: str, value: object) -> object | None:
    if settings_attr in _STRING_CONFIG_ATTRS:
        return "" if value is None else str(value).strip()
    if value == "" or value is None:
        return None
    if settings_attr in _INT_CONFIG_ATTRS:
        return int(value)
    return value


def update_runtime_config(*, body: dict, settings, critic, supervisor, logger, persist_config_to_env=None) -> dict:
    updated = []
    pending_settings: dict[str, object] = {}
    for plugin_key, settings_attr in _CONFIG_KEY_MAP.items():
        if plugin_key in body:
            val = _normalize_config_value(settings_attr, body[plugin_key])
            if val is None:
                continue
            pending_settings[settings_attr] = val
            updated.append(plugin_key)

    for settings_attr, val in pending_settings.items():
        setattr(settings, settings_attr, val)
        logger.info(
            "config/update: %s (%s)",
            settings_attr,
            "***" if "KEY" in settings_attr else val,
        )

    if persist_config_to_env is None:
        _persist_config_to_env(pending_settings)
    else:
        persist_config_to_env(pending_settings)

    if any(k.startswith("critic") for k in updated) and critic is not None:
        critic.refresh_runtime_config()
        logger.info("config/update: Critic instance reloaded")

    if any(k.startswith("supervisor") for k in updated) and supervisor is not None:
        supervisor.endpoint = settings.PROJECT_SUPERVISOR_ENDPOINT
        supervisor.api_key = settings.PROJECT_SUPERVISOR_API_KEY
        supervisor.model = settings.PROJECT_SUPERVISOR_MODEL
        supervisor.timeout = settings.SUPERVISOR_TIMEOUT
        logger.info("config/update: Supervisor instance reloaded")

    return {"status": "ok", "updated": updated}
