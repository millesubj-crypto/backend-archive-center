import json
import logging

from backend.archive.bridge import ArchiveBridge
from backend.settings import settings

logger = logging.getLogger(__name__)


class Storyteller:
    def __init__(self) -> None:
        self.bridge = ArchiveBridge()

    def search(
        self,
        user_input: str,
        wing: str | None = None,
        top_k: int | None = None,
        chat_session_id: str | None = None,
    ) -> list[dict]:
        if not user_input or not str(user_input).strip():
            return []

        try:
            limit = top_k if isinstance(top_k, int) and top_k > 0 else settings.TOP_K_RESULTS
            if not isinstance(limit, int) or limit <= 0:
                limit = 5
            limit = min(limit, 50)
            return self.bridge.search_memories(
                query=str(user_input).strip(),
                wing=wing,
                n_results=limit,
                chat_session_id=chat_session_id,
            )
        except Exception as exc:
            logger.error("Storyteller.search failed: %s", exc)
            return []

    def format_for_injection(self, items: list[dict]) -> str:
        if not items or not isinstance(items, list):
            return ""

        lines: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                source = item.get("source", "memory")
                if source == "chat_log":
                    lines.append(self._format_chat_log_item(item))
                else:
                    lines.append(self._format_memory_item(item))
            except Exception as exc:
                logger.warning("format_for_injection item skipped: %s", exc)

        return "\n".join(lines)

    def _format_memory_item(self, memory: dict) -> str:
        try:
            summary_text = self._extract_summary_text(memory.get("summary_json"))
            if not summary_text:
                summary_text = "(no summary)"
            summary_text = " ".join(summary_text.split())
            if len(summary_text) > 220:
                summary_text = summary_text[:217] + "..."

            importance = memory.get("importance")
            importance_text = "-" if importance is None else str(importance)
            archive_wing = memory.get("archive_wing") or memory.get("library_wing") or ""
            archive_room = memory.get("archive_room") or memory.get("library_room") or ""
            turn_index = memory.get("turn_index", "?")

            similarity = memory.get("similarity")
            sim_text = f"{similarity:.2f}" if similarity is not None else "-"

            recency = memory.get("recency_score")
            rec_text = f"{recency:.2f}" if recency is not None else "-"

            final = memory.get("final_score")
            final_text = f"{final:.2f}" if final is not None else sim_text

            support_trace = ""
            if bool(memory.get("chroma_live_hydrated") or memory.get("chroma_live_merged")):
                support_mode = str(memory.get("chroma_live_support_surface_mode") or "sqlite_hydrated_memory_support")
                truth_authority = str(memory.get("chroma_live_truth_authority") or "sqlite")
                precedence_ceiling = str(
                    memory.get("chroma_live_precedence_ceiling")
                    or "below_current_user_input_and_verified_direct_evidence"
                )
                support_trace = (
                    f" | support={support_mode} authority={truth_authority} ceiling={precedence_ceiling}"
                )

            return (
                "[memory|turn {turn}] score={final} (sim={sim} imp={importance} rec={rec}) | {wing}/{room}{support_trace} | {summary}".format(
                    turn=turn_index,
                    final=final_text,
                    sim=sim_text,
                    importance=importance_text,
                    rec=rec_text,
                    wing=archive_wing,
                    room=archive_room,
                    support_trace=support_trace,
                    summary=summary_text,
                )
            )
        except Exception as exc:
            logger.warning("_format_memory_item failed: %s", exc)
            return f"[memory|turn {memory.get('turn_index', '?')}] (format error)"

    def _format_chat_log_item(self, item: dict) -> str:
        try:
            turn_index = item.get("turn_index", "?")
            role = str(item.get("role") or "unknown")
            content = str(item.get("content") or "")
            content = " ".join(content.split())
            if len(content) > 150:
                content = content[:147] + "..."

            return "[chat_log|turn {turn}] role={role} | {content}".format(
                turn=turn_index,
                role=role,
                content=content,
            )
        except Exception as exc:
            logger.warning("_format_chat_log_item failed: %s", exc)
            return f"[chat_log|turn {item.get('turn_index', '?')}] (format error)"

    def _extract_summary_text(self, summary_json: object) -> str:
        if summary_json is None or summary_json == "":
            return ""

        # bool, int, float 등 단순 타입
        if isinstance(summary_json, (bool, int, float)):
            return str(summary_json)

        # list 타입
        if isinstance(summary_json, list):
            try:
                return json.dumps(summary_json, ensure_ascii=False)[:300]
            except Exception:
                return str(summary_json)[:300]

        if isinstance(summary_json, dict):
            parsed = summary_json
        elif isinstance(summary_json, str):
            try:
                parsed = json.loads(summary_json)
            except Exception:
                return summary_json
        else:
            return str(summary_json)

        if not isinstance(parsed, dict):
            return str(parsed)

        relationship_memory = parsed.get("relationship_memory")
        if isinstance(relationship_memory, dict):
            relationship_text = relationship_memory.get("bond_and_distance")
            if relationship_text:
                return str(
                    parsed.get("turn_summary")
                    or parsed.get("summary")
                    or relationship_text
                )

        return str(
            parsed.get("turn_summary")
            or parsed.get("summary")
            or json.dumps(parsed, ensure_ascii=False)
        )