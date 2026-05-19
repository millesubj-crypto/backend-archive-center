"""Shared embedding provenance helpers for Step 13 runtime surfaces."""

STEP13_EMBEDDING_PROVENANCE_POLICY_VERSION = "em1a.v1"
STEP13_NEEDS_REEMBED_POLICY_VERSION = "em1b.v1"
STEP13_RETRIEVAL_FALLBACK_POLICY_VERSION = "em1c.v1"


def assess_memory_embedding_state(*, embedding_value, embedding_model, current_model) -> dict:
    current_model_text = str(current_model or "").strip()
    stored_model_text = str(embedding_model or "").strip()

    if isinstance(embedding_value, list):
        has_embedding_vector = len(embedding_value) > 0
    else:
        has_embedding_vector = bool(str(embedding_value or "").strip())

    if not current_model_text:
        return {
            "provenance_policy_version": STEP13_EMBEDDING_PROVENANCE_POLICY_VERSION,
            "needs_reembed_policy_version": STEP13_NEEDS_REEMBED_POLICY_VERSION,
            "current_model": None,
            "stored_model": stored_model_text or None,
            "has_embedding_vector": has_embedding_vector,
            "status": "project_model_unset",
            "reason": "project_model_unset",
            "needs_reembed": None,
        }

    if not has_embedding_vector and not stored_model_text:
        status = "missing_embedding_and_model"
        needs_reembed = True
    elif not has_embedding_vector:
        status = "missing_embedding_vector"
        needs_reembed = True
    elif not stored_model_text:
        status = "missing_embedding_model"
        needs_reembed = True
    elif stored_model_text != current_model_text:
        status = "model_mismatch"
        needs_reembed = True
    else:
        status = "current_model_match"
        needs_reembed = False

    return {
        "provenance_policy_version": STEP13_EMBEDDING_PROVENANCE_POLICY_VERSION,
        "needs_reembed_policy_version": STEP13_NEEDS_REEMBED_POLICY_VERSION,
        "current_model": current_model_text,
        "stored_model": stored_model_text or None,
        "has_embedding_vector": has_embedding_vector,
        "status": status,
        "reason": status,
        "needs_reembed": needs_reembed,
    }