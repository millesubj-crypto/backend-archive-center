#!/usr/bin/env python3
"""
RisuAI Memory Orchestrator — .env API 키 암호화 도구

backend/.env 파일의 API 키 값을 Fernet 암호화로 변환한다.
암호화된 키는 마스터 키 없이 읽을 수 없으며,
마스터 키는 ~/.risuai/vault.key 에 저장된다.

사용법:
  python -m backend.encrypt_env              # backend/.env 처리 (기본)
  python -m backend.encrypt_env --env .env   # .env 경로 직접 지정
  python -m backend.encrypt_env --check      # 현재 암호화 상태 확인 (파일 변경 없음)
  python -m backend.encrypt_env --show-key   # 볼트 키 파일 위치 출력
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# API 키로 간주할 필드명 목록
_TARGET_FIELDS: frozenset[str] = frozenset({
    "PROJECT_MAIN_API_KEY",
    "PROJECT_SUPERVISOR_API_KEY",
    "PROJECT_CRITIC_API_KEY",
    "PROJECT_EMBEDDING_API_KEY",
})
# 이 접미사로 끝나는 필드도 처리 대상
_API_KEY_SUFFIXES = ("_API_KEY", "_API_SECRET", "_SECRET_KEY")

_ENV_LINE_RE = re.compile(r"^([A-Za-z0-9_]+)=(.*)$")


def _strip_quotes(value: str) -> str:
    """따옴표로 감싸진 값에서 따옴표를 제거한다."""
    if len(value) >= 2 and (
        (value[0] == '"' and value[-1] == '"')
        or (value[0] == "'" and value[-1] == "'")
    ):
        return value[1:-1]
    return value


def _process_env_file(
    env_path: Path,
    *,
    check_only: bool = False,
) -> tuple[list[str], int]:
    """
    .env 파일을 처리해 (새 라인 리스트, 변경 수)를 반환한다.
    check_only=True 이면 라인을 수정하지 않고 상태만 출력한다.
    """
    from backend.key_vault import encrypt_value, is_encrypted

    lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines: list[str] = []
    changed = 0

    for line in lines:
        stripped = line.rstrip("\r\n")
        m = _ENV_LINE_RE.match(stripped)

        if not m:
            new_lines.append(line)
            continue

        key, raw_value = m.group(1), _strip_quotes(m.group(2))

        is_target = key in _TARGET_FIELDS or any(
            key.endswith(s) for s in _API_KEY_SUFFIXES
        )

        if not is_target:
            new_lines.append(line)
            continue

        if check_only:
            if is_encrypted(raw_value):
                print(f"  {key}: [암호화됨 ✓]")
            elif raw_value:
                print(f"  {key}: [평문 ⚠  — 암호화 권장]")
            else:
                print(f"  {key}: [비어있음]")
            new_lines.append(line)
        else:
            if is_encrypted(raw_value):
                print(f"  {key}: 이미 암호화됨 (skip)")
                new_lines.append(line)
            elif raw_value:
                enc = encrypt_value(raw_value)
                new_lines.append(f"{key}={enc}\n")
                print(f"  {key}: ✓ 암호화 완료")
                changed += 1
            else:
                print(f"  {key}: 값 없음 (skip)")
                new_lines.append(line)

    return new_lines, changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RisuAI .env API 키 암호화 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  python -m backend.encrypt_env            # 암호화 실행\n"
            "  python -m backend.encrypt_env --check    # 상태 확인만\n"
            "  python -m backend.encrypt_env --show-key # 키 위치 출력\n"
        ),
    )
    parser.add_argument(
        "--env",
        default=None,
        help=".env 파일 경로 (기본: backend/.env)",
    )
    parser.add_argument(
        "--show-key",
        action="store_true",
        help="볼트 키 파일 위치 출력",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="암호화 상태 확인 (파일 변경 없음)",
    )
    args = parser.parse_args()

    from backend.key_vault import _KEY_FILE, _get_or_create_vault_key

    # 볼트 키 위치 출력
    if args.show_key:
        print(f"볼트 키 파일 위치: {_KEY_FILE}")
        print(f"파일 존재        : {_KEY_FILE.exists()}")
        return

    # .env 경로 결정
    env_path = Path(args.env) if args.env else Path(__file__).parent / ".env"

    if not env_path.exists():
        print(f"오류: .env 파일을 찾을 수 없습니다: {env_path}", file=sys.stderr)
        print("backend/.env 파일을 먼저 생성해주세요.", file=sys.stderr)
        sys.exit(1)

    print(f"처리 중: {env_path}\n")

    if args.check:
        print("── 암호화 상태 확인 ──")
        _process_env_file(env_path, check_only=True)
        return

    # 볼트 키 확보 (없으면 자동 생성)
    _get_or_create_vault_key()

    print("── API 키 암호화 ──")
    new_lines, changed = _process_env_file(env_path)

    if changed > 0:
        backup = env_path.with_suffix(".env.bak")
        backup.write_text(env_path.read_text(encoding="utf-8"), encoding="utf-8")
        env_path.write_text("".join(new_lines), encoding="utf-8")
        print(f"\n✓ {changed}개 키 암호화 완료.")
        print(f"  백업 파일: {backup}")
        print(f"  볼트 키  : {_KEY_FILE}")
        print()
        print("⚠ 주의: 볼트 키 파일을 안전한 곳에 백업해두세요.")
        print("  키를 분실하면 암호화된 API 키를 복구할 수 없습니다.")
        print("  볼트 키는 절대 .env 파일과 같은 곳(git repo 등)에 두지 마세요.")
    else:
        print("\n변경 사항 없음.")


if __name__ == "__main__":
    main()
