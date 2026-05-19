import os
import warnings
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings


_API_KEY_FIELDS = (
    "PROJECT_MAIN_API_KEY",
    "PROJECT_SUPERVISOR_API_KEY",
    "PROJECT_CRITIC_API_KEY",
    "PROJECT_EMBEDDING_API_KEY",
)


def _recovery_mode_enabled() -> bool:
    value = str(os.getenv("RISUAI_ALLOW_ENCRYPTED_KEY_RECOVERY") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_backend_env_path() -> Path:
    override = str(os.getenv("BACKEND_ENV_FILE") or "").strip()
    if override:
        return Path(override)
    return Path(__file__).with_name(".env")


def _clear_encrypted_api_keys_for_recovery(fields: list[str]) -> Path | None:
    env_path = _resolve_backend_env_path()
    if not env_path.exists():
        return None

    lines = env_path.read_text(encoding="utf-8").splitlines()
    changed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key not in fields:
            continue
        value = stripped.split("=", 1)[1].strip()
        if value.startswith("enc:") or value.startswith('"enc:') or value.startswith("'enc:"):
            lines[index] = f"{key}="
            changed = True

    if not changed:
        return None

    backup_path = env_path.with_suffix(env_path.suffix + ".recovery.bak")
    backup_path.write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")
    content = "\n".join(lines)
    env_path.write_text((content + "\n") if content else "", encoding="utf-8")
    return backup_path


def _apply_portability_recovery(settings_obj: "Settings", encrypted_fields: list[tuple[str, str]], reason: str) -> "Settings":
    field_names = [field for field, _ in encrypted_fields]
    backup_path = _clear_encrypted_api_keys_for_recovery(field_names)
    for field_name in field_names:
        object.__setattr__(settings_obj, field_name, "")

    backup_note = f" 백업: {backup_path}" if backup_path is not None else ""
    warnings.warn(
        f"암호화된 API 키 portability recovery 모드가 활성화되어 현재 PC에서 복호화할 수 없는 API 키를 비웠습니다. 이유: {reason}. backend 서버를 재기동한 후 API 키는 다시 입력해야 합니다.{backup_note}",
        stacklevel=2,
    )
    return settings_obj


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./memory.db"

    # Project Main LLM
    PROJECT_MAIN_PROVIDER: str = ""
    PROJECT_MAIN_ENDPOINT: str = ""
    PROJECT_MAIN_API_KEY: str = ""
    PROJECT_MAIN_MODEL: str = ""

    # Project Supervisor LLM
    PROJECT_SUPERVISOR_ENDPOINT: str = ""
    PROJECT_SUPERVISOR_API_KEY: str = ""
    PROJECT_SUPERVISOR_MODEL: str = ""

    # Project Critic LLM
    PROJECT_CRITIC_PROVIDER: str = ""
    PROJECT_CRITIC_ENDPOINT: str = ""
    PROJECT_CRITIC_API_KEY: str = ""
    PROJECT_CRITIC_MODEL: str = ""

    # Project Embedding LLM
    PROJECT_EMBEDDING_PROVIDER: str = "openai"
    PROJECT_EMBEDDING_ENDPOINT: str = ""
    PROJECT_EMBEDDING_API_KEY: str = ""
    PROJECT_EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_TIMEOUT: int = 30

    # 검???�정
    TOP_K_RESULTS: int = 5

    # Scoring weights (sum should be ~1.0)
    SCORING_SIMILARITY_WEIGHT: float = 0.60
    SCORING_IMPORTANCE_WEIGHT: float = 0.25
    SCORING_RECENCY_WEIGHT: float = 0.15

    # Recency decay
    RECENCY_HALF_LIFE_HOURS: float = 168.0   # 1 week
    RECENCY_FLOOR: float = 0.20              # 최소 20 % ???�래??memory???�전??밀?�나지 ?�음

    # ?�?�아??(�?
    SUPERVISOR_TIMEOUT: int = 60
    CRITIC_TIMEOUT: int = 90
    MAIN_TIMEOUT: int = 60

    # Sprint 4-C-2: Prune policy ("off" | "soft")
    PRUNE_POLICY: str = "soft"

    # Sprint 4-D-3: CORS
    ALLOWED_ORIGINS: str = "*"

    # 백엔???�트 (?�스 코드 변�??�이 uvicorn ?�행 ??--port ?�수�???값을 ?�용)
    # ?? python -m uvicorn backend.main:app --port $BACKEND_PORT --reload
    BACKEND_PORT: int = 8000

    # Step 17-C1/C2: Chroma shadow migration preflight/bootstrap baseline
    CHROMA_SHADOW_ENABLED: bool = True
    CHROMA_SHADOW_PERSIST_DIRECTORY: str = ".chroma_shadow"
    CHROMA_SHADOW_COLLECTION_NAME: str = "archive_center_shadow"
    CHROMA_SHADOW_DISK_BUDGET_MB: int = 2048
    CHROMA_LIVE_CUTOVER_MODE: str = "primary_memory"
    CHROMA_LIVE_CANDIDATE_LIMIT: int = 5

    model_config = {"env_file": str(_resolve_backend_env_path()), "env_file_encoding": "utf-8"}

    @model_validator(mode="after")
    def _decrypt_api_keys(self) -> "Settings":
        """enc: ?�두?��? ?�는 API ???�드�??�동?�로 복호?�한??

        encrypted ?�정???�으�?vault.key ?�락/불일치�? ?�버 ?�작 ?�점??즉시 차단?�다.
        """
        encrypted_fields = []
        for field in _API_KEY_FIELDS:
            value = getattr(self, field, "")
            if isinstance(value, str) and value.startswith("enc:"):
                encrypted_fields.append((field, value))

        if not encrypted_fields:
            return self

        try:
            from backend.key_vault import _KEY_FILE, decrypt_value as _decrypt, has_vault_key
        except ImportError as exc:
            if _recovery_mode_enabled():
                return _apply_portability_recovery(self, encrypted_fields, "cryptography_import_failed")
            raise ValueError(
                "?�호?�된 API ?��? ?�정?�어 ?��?�?cryptography 지?�을 불러?????�습?�다."
            ) from exc

        if not has_vault_key():
            if _recovery_mode_enabled():
                return _apply_portability_recovery(self, encrypted_fields, f"missing_vault_key:{_KEY_FILE}")
            raise ValueError(
                f"?�호?�된 API ?��? ?�정?�어 ?��?�?vault.key 가 ?�습?�다: {_KEY_FILE}"
            )

        for field, value in encrypted_fields:
            try:
                object.__setattr__(self, field, _decrypt(value))
            except Exception as exc:
                if _recovery_mode_enabled():
                    return _apply_portability_recovery(self, encrypted_fields, f"decrypt_failed:{field}")
                raise ValueError(
                    f"?�호?�된 API ?��? 복호?�할 ???�습?�다: {field}. vault.key �??�인?�세?? "
                    "?�른 PC?�서 ??�� .env ?�면 RISUAI_ALLOW_ENCRYPTED_KEY_RECOVERY=1 �?1??복구 기동 ??UI?�서 ?��? ?�시 ?�?�하?�요."
                ) from exc
        return self


settings = Settings()
