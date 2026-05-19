"""Step 17 Metrics (LC1) generic helpers.

These helpers are pure or DB-agnostic wrappers used by the LC1 metric routes.
They do not contain metric-specific builder logic.
"""

from __future__ import annotations

import logging

from fastapi.responses import JSONResponse

from backend.database import SessionLocal

logger = logging.getLogger(__name__)

_LC1P_COMPLETENESS_HEALTHY_MIN = 0.75
_LC1P_ANSWER_QUALITY_HEALTHY_MIN = 0.75


class Utf8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


def _run_step17_metric_route_c17(*, sid: str, payload_key: str, metric_label: str, log_label: str, builder_runner):
    db = SessionLocal()
    try:
        report = builder_runner(db)
        return Utf8JSONResponse(content={
            "status": "ok",
            "chat_session_id": sid,
            payload_key: report,
        })
    except Exception as exc:
        if exc.__class__.__name__ == "OperationalError":
            return Utf8JSONResponse(content={
                "status": "ok",
                "chat_session_id": sid,
                payload_key: {
                    "status": "off",
                    "reason": "source_tables_unavailable",
                    "gate_result": "off",
                    "warnings": [f"{metric_label}_source_tables_unavailable:{exc.__class__.__name__}"],
                    "fresh_baseline": True,
                },
            })
        logger.error("%s/%s 실패: %s", log_label, sid, exc)
        return Utf8JSONResponse(
            content={"status": "error", "detail": str(exc)},
            status_code=500,
        )
    finally:
        db.close()


def _build_step17_metric_component_off_report_c17(*, metric_label: str, exc: Exception) -> dict:
    return {
        "status": "off",
        "reason": "source_tables_unavailable" if exc.__class__.__name__ == "OperationalError" else "component_failed",
        "gate_result": "off",
        "warnings": [f"{metric_label}_component_unavailable:{exc.__class__.__name__}"],
        "fresh_baseline": exc.__class__.__name__ == "OperationalError",
    }


def _run_step17_metric_component_c17(*, db, metric_label: str, builder_runner):
    try:
        return builder_runner(db)
    except Exception as exc:
        return _build_step17_metric_component_off_report_c17(metric_label=metric_label, exc=exc)


def _classify_lc1p_split_summary_c17(*, completeness_score: float | None, answer_quality_score: float | None, any_signal: bool) -> tuple[str, str]:
    if not any_signal or completeness_score is None or answer_quality_score is None:
        return "off", "insufficient_signal"
    if completeness_score >= _LC1P_COMPLETENESS_HEALTHY_MIN and answer_quality_score >= _LC1P_ANSWER_QUALITY_HEALTHY_MIN:
        return "ok", "healthy"
    if completeness_score < _LC1P_COMPLETENESS_HEALTHY_MIN and answer_quality_score >= _LC1P_ANSWER_QUALITY_HEALTHY_MIN:
        return "ok", "retrieval_failure_dominant"
    if completeness_score >= _LC1P_COMPLETENESS_HEALTHY_MIN and answer_quality_score < _LC1P_ANSWER_QUALITY_HEALTHY_MIN:
        return "ok", "reader_failure_dominant"
    return "ok", "mixed_failure"


def _query_lc1q_latest_timestamp_c17(*, db, sid: str, label: str, query_runner) -> tuple[object | None, list[str]]:
    try:
        return query_runner(), []
    except Exception as exc:
        return None, [f"{label}_query_unavailable:{exc.__class__.__name__}"]


def _seconds_between_c17(later: object, earlier: object) -> float | None:
    if later is None or earlier is None:
        return None
    if not hasattr(later, "timestamp") or not hasattr(earlier, "timestamp"):
        return None
    try:
        delta = later - earlier
    except Exception:
        return None
    try:
        return round(max(delta.total_seconds(), 0.0), 3)
    except Exception:
        return None


def _classify_lc1q_freshness_lag_c17(*, save_delay: float | None, extraction_delay: float | None, promotion_visibility_lag: float | None) -> tuple[str, str]:
    lag_map = {
        "save_delay_dominant": save_delay,
        "extraction_delay_dominant": extraction_delay,
        "promotion_visibility_lag_dominant": promotion_visibility_lag,
    }
    available = {name: value for name, value in lag_map.items() if value is not None}
    if not available:
        return "off", "insufficient_signal"
    dominant_value = max(available.values())
    if dominant_value <= 5.0:
        return "ok", "healthy"
    dominant_labels = [name for name, value in available.items() if value == dominant_value]
    if len(dominant_labels) > 1:
        return "ok", "mixed_lag"
    return "ok", dominant_labels[0]
