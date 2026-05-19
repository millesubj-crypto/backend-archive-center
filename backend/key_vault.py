"""
RisuAI Memory Orchestrator — API Key Vault

API 키를 Fernet symmetric encryption으로 암호화/복호화한다.
마스터 키는 사용자 홈 디렉토리 (~/.risuai/vault.key) 에 보관되며,
프로젝트 디렉토리에는 암호화된 값(enc:<ciphertext>)만 저장된다.

암호화된 값의 형식: enc:<Fernet base64 ciphertext>

사용법:
  from backend.key_vault import encrypt_value, decrypt_value

  enc = encrypt_value("sk-abc123")   # → "enc:gAAAAAB..."
  plain = decrypt_value(enc)         # → "sk-abc123"
  plain = decrypt_value("plaintext") # → "plaintext"  (enc: 없으면 그대로)
"""
from __future__ import annotations

from pathlib import Path

_ENC_PREFIX = "enc:"
_VAULT_DIR = Path.home() / ".risuai"
_KEY_FILE = _VAULT_DIR / "vault.key"


def has_vault_key() -> bool:
    """현재 vault.key 파일이 존재하고 비어 있지 않은지 확인한다."""
    try:
        return bool(_KEY_FILE.read_bytes().strip())
    except OSError:
        return False


def _load_existing_vault_key() -> bytes:
    """기존 vault.key 를 읽어온다.

    복호화 경로에서는 새 키를 만들면 안 된다. 암호문과 맞지 않는 새 키가 생성되면
    런타임 호출 실패가 늦게 드러나기 때문에, 서버 시작 시점에 바로 실패시킨다.
    """
    if not _KEY_FILE.exists():
        raise FileNotFoundError(f"vault.key 파일을 찾을 수 없습니다: {_KEY_FILE}")

    raw = _KEY_FILE.read_bytes().strip()
    if not raw:
        raise ValueError(f"vault.key 파일이 비어 있습니다: {_KEY_FILE}")
    return raw


def _get_or_create_vault_key() -> bytes:
    """Fernet 마스터 키를 로드하거나 새로 생성한다.

    키 파일이 없으면 ~/.risuai/vault.key 에 새로 생성한다.
    - Unix/macOS: chmod 0o600 (소유자 읽기/쓰기 전용)
    - Windows   : 파일 위치(사용자 홈 디렉토리)로 보호
    """
    try:
        from cryptography.fernet import Fernet as _Fernet
    except ImportError as exc:
        raise ImportError(
            "cryptography 패키지가 필요합니다: pip install cryptography"
        ) from exc

    if _KEY_FILE.exists():
        raw = _KEY_FILE.read_bytes().strip()
        if raw:
            return raw

    # 새 키 생성
    _VAULT_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    key = _Fernet.generate_key()
    _KEY_FILE.write_bytes(key)
    try:
        _KEY_FILE.chmod(0o600)  # Windows에서는 제한적으로 동작
    except OSError:
        pass
    return key


def encrypt_value(plaintext: str) -> str:
    """평문 API 키를 암호화하고 enc:<ciphertext> 형태로 반환한다.

    이미 enc: 접두사가 있으면 그대로 반환한다.
    빈 문자열은 그대로 반환한다.
    """
    if not plaintext:
        return plaintext
    if plaintext.startswith(_ENC_PREFIX):
        return plaintext  # 이미 암호화된 값은 재암호화하지 않음
    from cryptography.fernet import Fernet as _Fernet
    f = _Fernet(_get_or_create_vault_key())
    return _ENC_PREFIX + f.encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_value(value: str) -> str:
    """enc: 접두사가 있는 값을 복호화한다.

    enc: 접두사가 없으면 그대로 반환 (fail-open — 평문 키도 정상 동작).
    복호화에 필요한 vault.key 가 없거나 값이 맞지 않으면 예외를 발생시킨다.
    """
    if not value or not value.startswith(_ENC_PREFIX):
        return value
    from cryptography.fernet import Fernet as _Fernet

    f = _Fernet(_load_existing_vault_key())
    try:
        return f.decrypt(value[len(_ENC_PREFIX):].encode("ascii")).decode("utf-8")
    except Exception as exc:
        raise ValueError("암호화된 API 키를 현재 vault.key 로 복호화할 수 없습니다.") from exc


def is_encrypted(value: str) -> bool:
    """값이 암호화된 형태(enc: 접두사)인지 확인한다."""
    return isinstance(value, str) and value.startswith(_ENC_PREFIX)
