"""Precision-first candidate ranking for the manual Spot workflow.

The radar does not create a new trading strategy. It ranks candidates that
already passed the existing empirical quality gate. A ranking score is only an
ordering aid; it is not a probability or a promise of profit.
"""
from __future__ import annotations


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
        return value if value == value else default
    except (TypeError, ValueError):
        return default


def rank_candidates(rows: list[dict], limit: int = 10) -> list[dict]:
    """Return qualified LONG candidates ordered by evidence strength.

    Only A-grade rows that already passed the quality gate are eligible. The
    composite score rewards positive out-of-sample expectancy, the conservative
    Wilson probability bound, signal agreement, and sample size. It is
    deliberately capped and labelled as a rank score rather than a win
    probability.
    """
    eligible = []
    for row in rows or []:
        if (row.get("action") != "TRADE" or row.get("direction") != "LONG"
                or str(row.get("grade", "")) != "A"):
            continue
        item = dict(row)
        probability = _float(item, "conservative_probability_pct") / 100.0
        expectancy = _float(item, "expected_value_r")
        agreement = _float(item, "agreement_pct") / 100.0
        samples = min(1.0, _float(item, "samples") / 100.0)
        ev_score = min(1.0, max(0.0, (expectancy + 0.10) / 0.60))
        item["rank_score"] = round(100.0 * (
            0.40 * ev_score + 0.28 * probability +
            0.17 * max(0.0, min(1.0, agreement)) +
            0.15 * samples), 1)
        eligible.append(item)
    eligible.sort(key=lambda item: (
        _float(item, "rank_score"),
        _float(item, "expected_value_r"),
        _float(item, "conservative_probability_pct"),
        _float(item, "samples"),
    ), reverse=True)
    return eligible[:max(0, int(limit))]
