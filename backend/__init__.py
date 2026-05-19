from datetime import datetime, timezone, timedelta

_KST = timezone(timedelta(hours=9))


def to_kst(dt) -> str | None:
    """UTC naive datetime → KST 문자열. None이면 None 반환."""
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_KST).strftime("%Y-%m-%d %H:%M:%S")
