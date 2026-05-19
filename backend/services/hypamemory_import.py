from __future__ import annotations

from fastapi import BackgroundTasks

from backend.services.explorer import HypaImportRequest


def handle_import_hypamemory(
    req: HypaImportRequest,
    background_tasks: BackgroundTasks,
    *,
    critic,
    append_audit_log,
    logger,
    response_cls,
) -> dict:
    """HypaMemory import body: validate, schedule background task, return accepted payload."""
    sid = (req.chat_session_id or "").strip()
    if not sid:
        return response_cls(
            content={"status": "error", "detail": "chat_session_id가 필요합니다."},
            status_code=400,
        )
    if not req.summaries:
        return response_cls(
            content={"status": "error", "detail": "summaries가 비어 있습니다."},
            status_code=400,
        )

    total = len(req.summaries)
    if total > 500:
        return response_cls(
            content={"status": "error", "detail": f"최대 500개까지 가능합니다 (요청: {total})"},
            status_code=400,
        )

    def _run_import():
        succeeded = 0
        failed = 0
        for idx, s in enumerate(req.summaries):
            text = (s.text or "").strip()
            if not text:
                failed += 1
                continue
            importance_hint = "(가중한 내용이므요 추가 주의를 해주세요.) " if s.is_important else ""
            tag_hint = f"(태그: {', '.join(s.tags)}) " if s.tags else ""
            turn_content = f"{importance_hint}{tag_hint}{text}"
            turn_index = -(idx + 1)
            try:
                critic.run(
                    turn_index=turn_index,
                    turn_content=turn_content,
                    context=[],
                    chat_session_id=sid,
                )
                succeeded += 1
            except Exception as exc:
                logger.warning("HypaMemory import failed for item %d: %s", idx, exc)
                failed += 1

        append_audit_log(
            event_type="hypamemory_import",
            chat_session_id=sid,
            target_type="session",
            summary=f"HypaMemory import: {succeeded}/{total} succeeded",
            details={"total": total, "succeeded": succeeded, "failed": failed},
            source="import",
        )
        logger.info("HypaMemory import done (session=%s, succeeded=%d, failed=%d)", sid, succeeded, failed)

    background_tasks.add_task(_run_import)

    logger.info("POST /import/hypamemory accepted (session=%s, count=%d)", sid, total)
    return {"status": "ok", "detail": f"{total}개의 내용이 가져오기가 시작되었습니다.", "total": total}
