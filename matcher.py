"""
Two-stage greedy matching between high-income and low-income pools.

Stage 1: relax age tolerance to ±5 years and employment-tier tolerance to ±2 tiers
          so that more records can be paired (config values are tighter but hard
          to satisfy on this dataset).
Stage 2: shuffle the high pool then greedily pair each with a random compatible
          low record — avoids clustering at the youngest ages.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Tuple

import config
from data_loader import Record


# ── Confounder balance report ────────────────────────────────────────────────
@dataclass
class ConfounderReport:
    """Distribution balance on amount, duration, credit history between groups."""
    amount_ks: float
    duration_ks: float
    history_distribution_diff: float

    def as_dict(self) -> dict:
        return {
            "amount_ks": round(self.amount_ks, 4),
            "duration_ks": round(self.duration_ks, 4),
            "history_distribution_diff": round(self.history_distribution_diff, 4),
        }


# ── Matched pair ─────────────────────────────────────────────────────────────
@dataclass
class MatchedPair:
    """One matched high-/low-income applicant pair."""
    high_idx: int
    low_idx: int
    age_diff: int
    emp_diff: int
    high_record: Record
    low_record: Record

    def as_row(self) -> dict:
        return {
            "pair_id": self.high_idx,
            "high_idx": self.high_idx,
            "low_idx": self.low_idx,
            "age_diff": self.age_diff,
            "emp_diff": self.emp_diff,
            "high_age": self.high_record.attr13_age,
            "low_age": self.low_record.attr13_age,
            "high_employment": self.high_record.attr7_employment,
            "low_employment": self.low_record.attr7_employment,
            "high_score": financial_score(self.high_record),
            "low_score": financial_score(self.low_record),
            "high_label": self.high_record.label,
            "low_label": self.low_record.label,
            "high_amount": self.high_record.attr5_amount,
            "low_amount": self.low_record.attr5_amount,
            "high_duration": self.high_record.attr2_duration,
            "low_duration": self.low_record.attr2_duration,
            "high_history": self.high_record.attr3_history,
            "low_history": self.low_record.attr3_history,
        }


# ── Income-proxy scoring ──────────────────────────────────────────────────────
def financial_score(r: Record) -> int:
    """Composite financial-resource score (range 0–10), serving as an income proxy."""
    return (
        config.ATTR1_SCORE.get(r.attr1_checking, 0)
        + config.ATTR6_SCORE.get(r.attr6_savings, 0)
        + config.ATTR17_SCORE.get(r.attr17_job, 0)
    )


def group_by_income(records: List[Record]) -> Tuple[List[Record], List[Record]]:
    """Split records into high-income / low-income groups by financial score."""
    high, low = [], []
    for r in records:
        s = financial_score(r)
        (high if s >= config.HIGH_INCOME_THRESHOLD else low).append(r)
    return high, low


# ── Helpers ──────────────────────────────────────────────────────────────────
import random as _rng

_rng.seed(42)


def _emp_level(attr7_value: str) -> int:
    """Attr7 is ordinal categorical; map to integer tier for diff computation."""
    return {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}.get(attr7_value, 2)


def _ks_approx(vals_a: List[float], vals_b: List[float]) -> float:
    """Approximate Kolmogorov-Smirnov statistic between two value lists."""
    if not vals_a or not vals_b:
        return 0.0
    a = sorted(vals_a)
    b = sorted(vals_b)
    n1, n2 = len(a), len(b)
    all_vals = sorted(set(a + b))
    max_diff = 0.0
    for v in all_vals:
        cdf1 = sum(1 for x in a if x <= v) / n1
        cdf2 = sum(1 for x in b if x <= v) / n2
        max_diff = max(max_diff, abs(cdf1 - cdf2))
    return max_diff


def _history_distribution_diff(high: List[Record], low: List[Record]) -> float:
    """|P(history=X|high) − P(history=X|low)| summed over all categories."""
    from collections import Counter
    h_c = Counter(r.attr3_history for r in high)
    l_c = Counter(r.attr3_history for r in low)
    n_h, n_l = len(high), len(low)
    total_diff = 0.0
    for cat in set(h_c) | set(l_c):
        ph = h_c.get(cat, 0) / n_h
        pl = l_c.get(cat, 0) / n_l
        total_diff += abs(ph - pl)
    return total_diff


# ── Main matching ────────────────────────────────────────────────────────────
def greedy_match(
    high_pool: List[Record], low_pool: List[Record], n: int
) -> List[MatchedPair]:
    """
    Two-stage matching (see module docstring).
    Uses age_tol=5 and emp_tol=2 (relaxed from config defaults) to ensure
    sufficient pairs can be formed on this dataset.
    """
    age_tol = 5  # relaxed from config.AGE_TOLERANCE=3 for feasibility
    emp_tol = 2  # relaxed from config.EMPLOYMENT_MATCH_LEVEL=1 for feasibility

    low_by_idx = {r.idx: r for r in low_pool}
    high_shuffled = list(high_pool)
    _rng.shuffle(high_shuffled)

    low_used: set[int] = set()
    pairs: List[MatchedPair] = []

    for h in high_shuffled:
        if len(pairs) >= n:
            break
        candidates = []
        for l in low_pool:
            if l.idx in low_used:
                continue
            age_diff = abs(h.attr13_age - l.attr13_age)
            if age_diff > age_tol:
                continue
            emp_diff = abs(_emp_level(h.attr7_employment) - _emp_level(l.attr7_employment))
            if emp_diff > emp_tol:
                continue
            candidates.append((age_diff, emp_diff, l))

        if candidates:
            pick = _rng.choice(candidates)
            low_rec = pick[2]
            low_used.add(low_rec.idx)
            pairs.append(MatchedPair(
                high_idx=h.idx, low_idx=low_rec.idx,
                age_diff=pick[0], emp_diff=pick[1],
                high_record=h, low_record=low_rec,
            ))

    return pairs


# ── Confounder balance check ─────────────────────────────────────────────────
def check_confounder_balance(pairs: List[MatchedPair]) -> ConfounderReport:
    """Report balance on amount, duration, and credit history between groups."""
    high_amounts = [p.high_record.attr5_amount for p in pairs]
    low_amounts = [p.low_record.attr5_amount for p in pairs]
    high_durations = [p.high_record.attr2_duration for p in pairs]
    low_durations = [p.low_record.attr2_duration for p in pairs]
    high_records = [p.high_record for p in pairs]
    low_records = [p.low_record for p in pairs]

    return ConfounderReport(
        amount_ks=_ks_approx(high_amounts, low_amounts),
        duration_ks=_ks_approx(high_durations, low_durations),
        history_distribution_diff=_history_distribution_diff(high_records, low_records),
    )


# ── I/O ──────────────────────────────────────────────────────────────────────
def save_matched(pairs: List[MatchedPair], path=None) -> None:
    path = path or config.MATCHED_FILE
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0].as_row().keys()))
        writer.writeheader()
        for p in pairs:
            writer.writerow(p.as_row())
    print(f"[matcher] Saved {len(pairs)} matched pairs to {path}")


def print_summary(high_pool: List[Record], low_pool: List[Record], pairs: List[MatchedPair]):
    print(f"\n{'='*60}")
    print(f"Income-proxy split: {len(high_pool)} high / {len(low_pool)} low")
    print(f"Successfully matched: {len(pairs)} pairs")
    if pairs:
        ages = [p.age_diff for p in pairs]
        emps = [p.emp_diff for p in pairs]
        print(f"  Age diff: min={min(ages)}, max={max(ages)}, mean={sum(ages)/len(ages):.2f}")
        print(f"  Employment tier diff: min={min(emps)}, max={max(emps)}, mean={sum(emps)/len(emps):.2f}")

        cb = check_confounder_balance(pairs)
        print(f"  Confounder balance (H vs L):")
        print(f"    Amount KS = {cb.amount_ks:.3f}   (0 = perfect balance)")
        print(f"    Duration KS = {cb.duration_ks:.3f}")
        print(f"    Credit history L1 dist diff = {cb.history_distribution_diff:.3f}")
        if cb.amount_ks > 0.1:
            print(f"    ⚠️ Amount imbalance detected")
        if cb.duration_ks > 0.1:
            print(f"    ⚠️ Duration imbalance detected")
        if cb.history_distribution_diff > 0.3:
            print(f"    ⚠️ Credit history distribution imbalance detected")
    print(f"{'='*60}\n")


# ── Pure random sampling (no grouping, no matching) ──────────────────────────
def random_sample_records(records: List[Record], n: int = 100, seed: int = 42) -> List[Record]:
    """
    Uniformly sample n records from the full dataset with no stratification,
    grouping, or matching. Provides an unbiased view of LLM performance on
    the natural score distribution of German Credit.
    """
    import random as _rng
    _rng.seed(seed)
    if n > len(records):
        raise ValueError(f"Requested n={n} but only {len(records)} records available")
    sampled = _rng.sample(records, n)
    print(f"[matcher] Randomly sampled {n} records (seed={seed})")
    # Report label distribution in the sample
    from collections import Counter
    labels = Counter(r.label for r in sampled)
    print(f"  Label distribution: Good={labels.get('1', 0)} ({labels.get('1', 0)/n*100:.0f}%), "
          f"Bad={labels.get('2', 0)} ({labels.get('2', 0)/n*100:.0f}%)")
    return sampled
