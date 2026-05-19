from dataclasses import dataclass
import re


@dataclass(frozen=True)
class HybridTailBudgetPolicy:
    version: str
    max_promoted_items: int
    max_score_gap: float


@dataclass(frozen=True)
class HybridRetrievalPolicy:
    hybrid_baseline_version: str
    soft_bias_version: str
    keyword_boost_cap: float
    speaker_bias_weight: float
    location_bias_weight: float
    storyline_bias_weight: float
    soft_bias_cap: float
    generic_archive_slots: frozenset[str]
    term_pattern: re.Pattern[str]
    stopwords: frozenset[str]
    tail_budget_policy: HybridTailBudgetPolicy


_HY1_KEYWORD_BOOST_CAP = 0.12
_HY1_SPEAKER_BIAS_WEIGHT = 0.04
_HY1_LOCATION_BIAS_WEIGHT = 0.05
_HY1_STORYLINE_BIAS_WEIGHT = 0.06
_HY1_SOFT_BIAS_CAP = 0.12


HY1_POLICY = HybridRetrievalPolicy(
    hybrid_baseline_version="hy1a.v1",
    soft_bias_version="hy1b.v1",
    keyword_boost_cap=_HY1_KEYWORD_BOOST_CAP,
    speaker_bias_weight=_HY1_SPEAKER_BIAS_WEIGHT,
    location_bias_weight=_HY1_LOCATION_BIAS_WEIGHT,
    storyline_bias_weight=_HY1_STORYLINE_BIAS_WEIGHT,
    soft_bias_cap=_HY1_SOFT_BIAS_CAP,
    generic_archive_slots=frozenset({
        "wing_general",
        "wing_default",
        "room_general",
        "room_events",
        "room_default",
        "room_misc",
        "room_unknown",
    }),
    term_pattern=re.compile(r"[0-9a-zA-Z가-힣\u3040-\u30ff\u4e00-\u9fff_:-]+"),
    stopwords=frozenset({
        "a",
        "an",
        "and",
        "are",
        "at",
        "be",
        "did",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "who",
        "why",
        "with",
    }),
    tail_budget_policy=HybridTailBudgetPolicy(
        version="hy1d.v1",
        max_promoted_items=1,
        max_score_gap=max(_HY1_KEYWORD_BOOST_CAP, _HY1_SOFT_BIAS_CAP),
    ),
)


HYBRID_RETRIEVAL_POLICY_REGISTRY = {
    HY1_POLICY.hybrid_baseline_version: HY1_POLICY,
}

CURRENT_HYBRID_RETRIEVAL_POLICY = HY1_POLICY