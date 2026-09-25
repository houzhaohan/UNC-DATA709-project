"""
Evaluator: derive approval decisions from LLM scores, compute ground-truth
metrics (AUC, KS, confusion matrix, cost-sensitive score), per-group summary,
and per-attribute real default rates for bias analysis.

Ground-truth label convention (from config):
    "1" = Good (low risk), "2" = Bad (high risk)
We treat "2" as the positive (target-of-interest) class for scoring purposes,
i.e., higher score → higher risk → more likely Bad.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict
import statistics
import math

import config


# ── Score row ───────────────────────────────────────────────────────────────
@dataclass
class ScoreRow:
    idx: int
    group: str                           # "high" | "low" | "unknown"
    score: float                         # LLM risk score 0-100 (nan if missing)
    decision: str | None                 # "approve" | "reject", derived from score
    n_advice: int                        # number of counterfactual suggestions
    label: str = ""                      # ground-truth "1" (Good) or "2" (Bad)


def _decision_from_score(score: float, threshold: float | None = None) -> str | None:
    thr = threshold if threshold is not None else config.SCORE_THRESHOLD
    if math.isnan(score):
        return None
    return "reject" if score >= thr else "approve"


def score_one(idx: int, group: str, llm_result: dict, label: str = "") -> ScoreRow:
    score_raw = llm_result.get("score")
    score = float(score_raw) if score_raw is not None else float("nan")
    decision = _decision_from_score(score)
    cf = llm_result.get("counterfactual") or []
    return ScoreRow(
        idx=idx, group=group, score=score, decision=decision,
        n_advice=len(cf), label=label,
    )


def score_all(results: Dict[int, dict], idx_to_group: Dict[int, str],
              records_by_idx: Dict[int, "Record"] | None = None) -> List[ScoreRow]:
    from data_loader import Record
    rows: List[ScoreRow] = []
    for idx, r in results.items():
        result = r.get("result") if isinstance(r, dict) and "result" in r else r
        group = (
            r.get("group") if isinstance(r, dict) else None
        ) or idx_to_group.get(idx, "unknown")
        label = ""
        if records_by_idx and idx in records_by_idx:
            label = records_by_idx[idx].label
        rows.append(score_one(idx, group, result, label))
    return rows


# ── Per-group summary ────────────────────────────────────────────────────────
def summarize(rows: List[ScoreRow]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    # Discover group labels dynamically — works for high/low split or any
    # other label (e.g., "random" from pure uniform sampling).
    for g in sorted({r.group for r in rows if r.group}):
        g_rows = [r for r in rows if r.group == g]
        if not g_rows:
            continue
        scores = [r.score for r in g_rows if not math.isnan(r.score)]
        approved = sum(1 for r in g_rows if r.decision == "approve")
        n = len(g_rows)
        summary[g] = {
            "n": n,
            "approved": approved,
            "rejected": n - approved,
            "approval_rate": approved / n if n else None,
            "score_mean": statistics.mean(scores) if scores else None,
            "score_median": statistics.median(scores) if scores else None,
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "total_advice": sum(r.n_advice for r in g_rows),
            "avg_advice_per_case": (sum(r.n_advice for r in g_rows) / n) if n else None,
        }
    return summary


# ── Ground-truth metrics ─────────────────────────────────────────────────────
def compute_ground_truth_metrics(rows: List[ScoreRow]) -> dict:
    """
    Compare LLM decisions & scores against ground-truth labels.
    Treats "2" (Bad) as the positive class.

    Returns a dict with:
      - confusion_matrix: {"tp", "fp", "tn", "fn"}
      - accuracy, precision, recall, f1
      - auc: approximated via Mann-Whitney U statistic
      - ks: Kolmogorov-Smirnov statistic between Good and Bad score distributions
      - cost: total cost using config.COST_MATRIX
      - cost_per_case: mean cost per record
    """
    good_scores: List[float] = []
    bad_scores: List[float] = []
    tp = fp = tn = fn = 0
    total_cost = 0
    counted = 0

    for r in rows:
        if not r.label or math.isnan(r.score) or r.decision is None:
            continue
        counted += 1
        actual_is_bad = (r.label == config.NEGATIVE_LABEL)
        pred_is_reject = (r.decision == "reject")
        # reject ↔ predict Bad (high risk)
        if pred_is_reject and actual_is_bad:
            tp += 1
        elif pred_is_reject and not actual_is_bad:
            fp += 1
        elif not pred_is_reject and not actual_is_bad:
            tn += 1
        elif not pred_is_reject and actual_is_bad:
            fn += 1

        cost_key = (r.label, "2" if pred_is_reject else "1")
        total_cost += config.COST_MATRIX.get(cost_key, 0)

        if actual_is_bad:
            bad_scores.append(r.score)
        else:
            good_scores.append(r.score)

    if counted == 0:
        return {"_error": "no labeled rows with valid scores"}

    accuracy = (tp + tn) / counted
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # AUC via Mann-Whitney U (equivalent to AUC / AUC = 1 - U / (n_good * n_bad))
    auc = _auc_from_mann_whitney(good_scores, bad_scores)

    # KS statistic
    ks = _ks_statistic(good_scores, bad_scores)

    return {
        "n_counted": counted,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "ks": round(ks, 4),
        "cost_total": total_cost,
        "cost_per_case": round(total_cost / counted, 4),
    }


def _auc_from_mann_whitney(good_scores: List[float], bad_scores: List[float]) -> float:
    """AUC-ROC computed via the Mann-Whitney U rank-sum statistic.
    AUC = 1 - U / (n1 * n2). Returns probability that a randomly-chosen Bad
    record scores strictly higher than a randomly-chosen Good record."""
    n1, n2 = len(good_scores), len(bad_scores)
    if n1 == 0 or n2 == 0:
        return 0.5
    # Combine, sort ascending, assign ranks (average for ties)
    combined = [(s, 0) for s in good_scores] + [(s, 1) for s in bad_scores]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    # U_bad = rank sum of bad records − n2(n2+1)/2
    rank_sum_bad = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 1)
    u_bad = rank_sum_bad - n2 * (n2 + 1) / 2
    auc = u_bad / (n1 * n2)
    return auc


def _ks_statistic(good_scores: List[float], bad_scores: List[float]) -> float:
    """Kolmogorov-Smirnov two-sample statistic."""
    if not good_scores or not bad_scores:
        return 0.0
    good = sorted(good_scores)
    bad = sorted(bad_scores)
    n1, n2 = len(good), len(bad)
    all_vals = sorted(set(good + bad))
    max_diff = 0.0
    for v in all_vals:
        cdf1 = sum(1 for x in good if x <= v) / n1
        cdf2 = sum(1 for x in bad if x <= v) / n2
        max_diff = max(max_diff, abs(cdf1 - cdf2))
    return max_diff


def scan_thresholds(rows: List[ScoreRow], lo: float = 10, hi: float = 90, step: float = 5) -> List[dict]:
    """
    Sweep threshold ∈ [lo, hi] with step and compute metrics at each.
    Returns list of rows sorted by the primary optimization target (default: min cost).

    Note: AUC and KS are threshold-independent and remain constant; only
    accuracy/precision/recall/f1/cost change.

    IMPORTANT: Uses deepcopy so the caller's rows list is NOT mutated.
    """
    import copy as _copy
    candidates: List[ScoreRow] = [_copy.deepcopy(r) for r in rows]  # deep copy to avoid mutating caller's rows
    results = []
    thr = lo
    while thr <= hi + 1e-9:
        # Re-derive decisions with this threshold on the deep-copied candidates only
        for r in candidates:
            r.decision = _decision_from_score(r.score, thr)
        gt = compute_ground_truth_metrics(candidates)
        if "_error" in gt:
            thr += step
            continue
        results.append({
            "threshold": thr,
            **gt,
        })
        thr += step

    if not results:
        return results

    # Rank by different targets
    best_acc = max(results, key=lambda r: r["accuracy"])
    best_f1 = max(results, key=lambda r: r["f1"])
    best_cost = min(results, key=lambda r: r["cost_per_case"])
    best_recall_bad = max(results, key=lambda r: r["recall"])  # recall = P(Bad | rejected)

    # Attach winners as attributes on the list-like container
    results.sort(key=lambda r: r["cost_per_case"])  # default sort by cost (most relevant for credit)
    print("\n" + "=" * 80)
    print(f"Threshold scan (lo={lo}, hi={hi}, step={step})")
    print("=" * 80)
    print(f"{'Thr':>5} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>6} "
          f"{'Cost/case':>10} {'Best?':>7}")
    print("-" * 64)
    for r in results:
        # Mark which ones are best for which metric
        tags = []
        if r["threshold"] == best_acc["threshold"]:
            tags.append("max_acc")
        if r["threshold"] == best_f1["threshold"]:
            tags.append("max_f1")
        if r["threshold"] == best_cost["threshold"]:
            tags.append("min_cost")
        if r["threshold"] == best_recall_bad["threshold"]:
            tags.append("max_recall")
        tag_str = ",".join(tags) if tags else ""
        print(f"{r['threshold']:>5.0f} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} {r['f1']:>6.4f} "
              f"{r['cost_per_case']:>10.2f}  {tag_str}")

    # Header explaining what each recall means
    print(f"\n  Interpretation:")
    print(f"    min_cost — lowest total cost using German Credit 5:1 cost matrix")
    print(f"    max_acc  — highest overall accuracy")
    print(f"    max_recall — lowest false-negative rate (catches the most Bad applicants)")
    print(f"    AUC = {results[0]['auc']:.4f} (threshold-independent, constant across all thresholds)")
    return results


# ── Per-attribute real default rates ────────────────────────────────────────
def compute_attribute_default_rates(records: List["Record"]) -> Dict[int, List[dict]]:
    """
    For each of the 20 attributes, group records by value and compute:
      n, n_good, n_bad, default_rate (= n_bad / n).

    IMPORTANT: Uses config.bin_numerical for numerical attributes so that
    groupings are IDENTICAL to bias_analyzer and baseline_model. This lets us
    directly compare "LLM gap direction" vs "true default-rate direction".
    Returns dict[attr_num] -> list of rows sorted by default_rate descending.
    """
    from data_loader import Record
    out: Dict[int, List[dict]] = {}
    for attr_num in range(1, 21):
        meta = config.ATTR_DICT[attr_num]
        numerical = meta["type"] == "numerical"
        groups: Dict[str, List[str]] = defaultdict(list)

        # Collect all values for quartile-based binning
        all_vals: List[float] = []
        if numerical and config.NUMERICAL_BINS.get(attr_num) == "quartiles":
            for rec in records:
                try:
                    all_vals.append(float(rec.attrs[attr_num - 1]))
                except ValueError:
                    pass

        for rec in records:
            raw = rec.attrs[attr_num - 1]
            if meta["type"] == "categorical":
                label = meta["values"].get(raw, raw)
            else:
                try:
                    label = config.bin_numerical(float(raw), attr_num, all_vals or None)
                except ValueError:
                    label = str(raw)
            groups[label].append(rec.label)

        rows = []
        for label, labels in groups.items():
            n = len(labels)
            n_bad = sum(1 for l in labels if l == config.NEGATIVE_LABEL)
            rows.append({
                "value": label,
                "n": n,
                "n_good": n - n_bad,
                "n_bad": n_bad,
                "default_rate": round(n_bad / n, 4) if n else None,
            })
        rows.sort(key=lambda r: -(r["default_rate"] or 0))
        out[attr_num] = rows
    return out
