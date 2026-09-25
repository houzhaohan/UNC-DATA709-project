"""
Baseline model: train a LogisticRegression on the NON-SAMPLED records of the
German Credit Dataset (out-of-sample w.r.t. the LLM-evaluated 100 records),
then produce per-attribute gap metrics that are directly comparable to the LLM.

Four critical fixes (P0) over the previous version:
  1. NO DATA LEAKAGE — train split ≠ test split. The baseline has NEVER seen
     the labels of the records we compare it against.
  2. IDENTICAL GROUPING — numerical attributes use the same config.NUMERICAL_BINS
     as bias_analyzer.py, so baseline and LLM gaps measure exactly the same thing.
  3. STANDARD SCALING — numerical features are StandardScaler fit on the train
     split only. Eliminates the lbfgs ConvergenceWarning and gives the baseline
     a fair chance.
  4. AUC BENCHMARK — reports baseline Mann-Whitney AUC on the test split so we
     can judge whether the LLM clears the simplest possible hurdle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import config
from data_loader import Record


# ── Feature encoding ─────────────────────────────────────────────────────────
def _encode_records(
    records: List[Record],
    cat_vocabs: Dict[int, List[str]] | None = None,
) -> Tuple[List[List[float]], List[str], List[int], Dict[int, List[str]]]:
    """
    One-hot encode categoricals + pass-through numericals (raw, unscaled).

    cat_vocabs: If None, built from this records list. Otherwise, use provided
                vocabularies (critical for train/test split — both splits must
                use identical vocabularies or one-hot dimensions won't match).

    Returns (X_raw, feature_names, y_binary, cat_vocabs_used).
    Scaling is applied by the caller on the train split only.
    """
    X: List[List[float]] = []
    y: List[int] = []

    if cat_vocabs is None:
        cat_vocabs = {}
        for attr_num in range(1, 21):
            meta = config.ATTR_DICT[attr_num]
            if meta["type"] == "categorical":
                cat_vocabs[attr_num] = sorted(set(
                    rec.attrs[attr_num - 1] for rec in records
                ))

    # Build feature names from vocabularies
    feature_names: List[str] = []
    for attr_num in range(1, 21):
        meta = config.ATTR_DICT[attr_num]
        if meta["type"] == "categorical":
            for raw in cat_vocabs[attr_num]:
                label = meta["values"].get(raw, raw)
                feature_names.append(f"A{attr_num}: {label}")
        else:
            feature_names.append(f"A{attr_num}: {meta['name']}")

    for rec in records:
        row: List[float] = []
        for attr_num in range(1, 21):
            meta = config.ATTR_DICT[attr_num]
            raw = rec.attrs[attr_num - 1]
            if meta["type"] == "categorical":
                onehots = [1.0 if raw == v else 0.0 for v in cat_vocabs[attr_num]]
                row.extend(onehots)
            else:
                row.append(float(raw))
        X.append(row)
        y.append(1 if rec.label == config.NEGATIVE_LABEL else 0)

    return X, feature_names, y, cat_vocabs


def train_baseline(
    train_records: List[Record],
    test_records: List[Record],
):
    """
    Train LogisticRegression on train_records only, predict P(Bad) on
    test_records only. Scaling is fit on train, applied to both splits.

    Returns (clf, test_scores_by_idx, feature_names, scaler).
    test_scores_by_idx[idx] = P(Bad) ∈ [0, 100] — out-of-sample, leak-free.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # ── Build categorical vocab from ALL records so train/test dims match ──
    all_for_vocab = train_records + test_records
    _, feature_names_train, _, cat_vocabs = _encode_records(all_for_vocab)
    # Re-encode train with shared vocab (feature_names identical)
    X_train_raw, feature_names, y_train, _ = _encode_records(train_records, cat_vocabs)
    X_test_raw, _, _, _ = _encode_records(test_records, cat_vocabs)

    X_train_raw_np = np.array(X_train_raw, dtype=float)
    X_test_raw_np = np.array(X_test_raw, dtype=float)

    # ── StandardScaler: fit on train ONLY ──
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw_np)
    X_test = scaler.transform(X_test_raw_np)

    # ── Train ──
    clf = LogisticRegression(
        penalty="l2", C=1.0, max_iter=5000, solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    # ── Predict P(Bad) on test set only ──
    proba = clf.predict_proba(X_test)[:, 1]   # P(Bad) ∈ [0, 1]
    proba_0_100 = proba * 100.0

    test_by_idx: Dict[int, float] = {}
    for rec, score in zip(test_records, proba_0_100):
        test_by_idx[rec.idx] = float(score)

    return clf, test_by_idx, feature_names, scaler


# ── Numerical binning (IDENTICAL to bias_analyzer.py — single source of truth) ──
def _bin_numerical(value: float, attr_num: int, all_values: List[float] | None = None) -> str:
    """Mirror — delegates to config.bin_numerical."""
    return config.bin_numerical(value, attr_num, all_values)


# ── Baseline gap computation ─────────────────────────────────────────────────
def compute_baseline_gaps(
    records: List[Record],
    baseline_scores: Dict[int, float],
    matched_indices: List[int] | None = None,
) -> Dict[int, List[dict]]:
    """
    For each attribute, compute per-group mean baseline score and gap
    (same definition as bias_analyzer.py: max mean − min mean).

    NOW USES IDENTICAL NUMERICAL_BINS AS BIAS_ANALYZER — so gaps are
    apples-to-apples for both baseline and LLM.
    """
    from collections import defaultdict

    idx_set = set(matched_indices) if matched_indices else None
    out: Dict[int, List[dict]] = {}

    for attr_num in range(1, 21):
        meta = config.ATTR_DICT[attr_num]
        groups: Dict[str, List[float]] = defaultdict(list)

        # Collect all values for quartile-based binning
        all_vals_for_quartiles: List[float] = []
        if meta["type"] != "categorical" and config.NUMERICAL_BINS.get(attr_num) == "quartiles":
            for rec in records:
                if idx_set is not None and rec.idx not in idx_set:
                    continue
                if rec.idx not in baseline_scores:
                    continue
                all_vals_for_quartiles.append(float(rec.attrs[attr_num - 1]))

        for rec in records:
            if idx_set is not None and rec.idx not in idx_set:
                continue
            if rec.idx not in baseline_scores:
                continue
            raw = rec.attrs[attr_num - 1]
            if meta["type"] == "categorical":
                label = meta["values"].get(raw, raw)
            else:
                label = _bin_numerical(float(raw), attr_num, all_vals_for_quartiles)
            groups[label].append(baseline_scores[rec.idx])

        rows = []
        for label, scores in groups.items():
            if not scores:
                continue
            n = len(scores)
            rows.append({
                "value": label,
                "n": n,
                "mean_score": round(sum(scores) / n, 2),
                "median_score": round(sorted(scores)[n // 2], 2),
            })
        rows.sort(key=lambda r: r["mean_score"])
        out[attr_num] = rows

    return out


# ── AUC helper (Mann-Whitney U probability) ───────────────────────────────────
def _mann_whitney_auc(good_scores: List[float], bad_scores: List[float]) -> float:
    """P(bad_score > good_score) using Mann-Whitney U (same convention as evaluator.py)."""
    if not good_scores or not bad_scores:
        return float("nan")
    import numpy as np
    combined = [(s, 0) for s in good_scores] + [(s, 1) for s in bad_scores]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    U_bad = sum(r for (_, cls), r in zip(combined, ranks) if cls == 1) - len(bad_scores) * (len(bad_scores) + 1) / 2
    n1, n2 = len(bad_scores), len(good_scores)
    return float(U_bad / (n1 * n2))


def compute_baseline_auc(
    records: List[Record],
    baseline_scores: Dict[int, float],
    matched_indices: List[int] | None = None,
) -> float:
    """Baseline Mann-Whitney AUC on the test subset. Directly comparable to LLM AUC."""
    idx_set = set(matched_indices) if matched_indices else None
    good, bad = [], []
    for rec in records:
        if idx_set is not None and rec.idx not in idx_set:
            continue
        if rec.idx not in baseline_scores:
            continue
        if rec.label == config.NEGATIVE_LABEL:
            bad.append(baseline_scores[rec.idx])
        else:
            good.append(baseline_scores[rec.idx])
    return _mann_whitney_auc(good, bad)


# ── η² (variance explained) helper ──────────────────────────────────────────
def _eta_squared(groups_scores: Dict[str, List[float]]) -> float:
    """
    η² = SS_between / SS_total. Range [0, 1].
    Cross-scale invariant — directly comparable across LLM and baseline,
    and across attributes with different numbers of groups.
    """
    all_scores = [s for scores in groups_scores.values() for s in scores]
    if len(all_scores) < 2:
        return 0.0
    grand_mean = sum(all_scores) / len(all_scores)
    ss_total = sum((s - grand_mean) ** 2 for s in all_scores)
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for label, scores in groups_scores.items():
        if not scores:
            continue
        n = len(scores)
        mean_k = sum(scores) / n
        ss_between += n * (mean_k - grand_mean) ** 2
    return float(ss_between / ss_total)


def _build_groups_for_attr(
    attr_num: int,
    records: List[Record],
    scores_by_idx: Dict[int, float],
    idx_set: set | None = None,
) -> Dict[str, List[float]]:
    """Build {group_label: [scores]} for one attribute, using identical binning."""
    from collections import defaultdict
    meta = config.ATTR_DICT[attr_num]
    groups: Dict[str, List[float]] = defaultdict(list)
    numerical = meta["type"] != "categorical"

    # Collect all values for quartile-based binning
    all_vals: List[float] = []
    if numerical and config.NUMERICAL_BINS.get(attr_num) == "quartiles":
        for rec in records:
            if idx_set is not None and rec.idx not in idx_set:
                continue
            if rec.idx not in scores_by_idx:
                continue
            try:
                all_vals.append(float(rec.attrs[attr_num - 1]))
            except ValueError:
                pass

    for rec in records:
        if idx_set is not None and rec.idx not in idx_set:
            continue
        if rec.idx not in scores_by_idx:
            continue
        raw = rec.attrs[attr_num - 1]
        if meta["type"] == "categorical":
            label = meta["values"].get(raw, raw)
        else:
            try:
                label = config.bin_numerical(float(raw), attr_num, all_vals or None)
            except ValueError:
                label = str(raw)
        groups[label].append(scores_by_idx[rec.idx])
    return dict(groups)


def _bootstrap_eta_ratio(
    attr_num: int,
    records: List[Record],
    baseline_scores: Dict[int, float],
    llm_groups_rows: List[dict],   # llm per-group rows (from bias_analyzer)
    idx_set: set,
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI for Δη² = eta_llm − eta_baseline on a single attribute.
    Resamples test-set records (with replacement), recomputes both η² each time.

    Returns (delta_mean, ci_lower, ci_upper).
    """
    import random as _rng
    _rng.seed(seed)

    test_records = [r for r in records if r.idx in idx_set]
    if len(test_records) < 5:
        return float("nan"), float("nan"), float("nan")

    # Build LLM scores per idx (from llm_groups_rows — but we need idx→score map)
    # Instead, derive from llm_groups_rows: each row has "value" (group label) + per-row n
    # We need actual scores to bootstrap. Let's recompute from groups directly.
    # The caller will pass llm_scores_by_idx instead.
    # ... We'll handle this in save_baseline_comparison by building both score maps upfront.
    return 0.0, 0.0, 0.0  # placeholder; real impl below


# ── Gap + AUC helpers ────────────────────────────────────────────────────────
def gap_from_rows(rows: List[dict]) -> float:
    """Max-min gap from sorted rows (kept for backward compat)."""
    if len(rows) < 2:
        return 0.0
    def _get_mean(r):
        return r.get("mean_score", r.get("score_mean", 0.0))
    return _get_mean(rows[-1]) - _get_mean(rows[0])


def _bootstrap_attr_deltas(
    attr_num: int,
    test_records: List[Record],
    baseline_scores_test: Dict[int, float],
    llm_scores_test: Dict[int, float],
    n_boot: int = 500,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap for Δη² = eta_llm − eta_baseline on ONE attribute.
    Returns (Δ_mean, CI_lower, CI_upper).
    """
    import random as _rng
    _rng.seed(seed)
    n = len(test_records)
    if n < 5:
        return float("nan"), float("nan"), float("nan")

    deltas: List[float] = []
    for _ in range(n_boot):
        # Resample test records with replacement
        resampled = [test_records[_rng.randrange(n)] for _ in range(n)]
        rs_idx_set = {r.idx for r in resampled}

        baseline_groups = _build_groups_for_attr(
            attr_num, resampled, baseline_scores_test, rs_idx_set)
        llm_groups = _build_groups_for_attr(
            attr_num, resampled, llm_scores_test, rs_idx_set)

        eta_b = _eta_squared(baseline_groups)
        eta_l = _eta_squared(llm_groups)
        deltas.append(eta_l - eta_b)

    if not deltas:
        return float("nan"), float("nan"), float("nan")

    deltas.sort()
    mean_d = sum(deltas) / len(deltas)
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[int(0.975 * len(deltas))]
    return float(mean_d), float(lo), float(hi)


def save_baseline_comparison(
    records: List[Record],
    baseline_scores: Dict[int, float],
    llm_gaps: Dict[int, List[dict]],
    matched_indices: List[int] | None = None,
    llm_auc: float | None = None,
    llm_scores_by_idx: Dict[int, float] | None = None,
    path: Path | None = None,
) -> None:
    """
    Write baseline-vs-LLM comparison. KEY FIXES:
      #1 Cross-attribute/cross-model comparability: uses η² (variance explained)
         instead of raw gaps — η² is scale-invariant and comparable across attributes.
      #2 Direction vs ground truth: compares each model's group ordering against
         the TRUE default-rate ordering.
      #3 Bootstrap CI: Δη² CI from 500 resamples. If CI upper > 0 but lower ≈ 0,
         amplification is marginal; if CI upper < 0, LLM is NOT amplifying.
    """
    path = path or config.BASELINE_FILE
    idx_set = set(matched_indices) if matched_indices else None
    baseline_auc = compute_baseline_auc(records, baseline_scores, matched_indices)

    # ── Need LLM scores per idx for bootstrap ──
    if llm_scores_by_idx is None:
        # Fall back: derive from llm_gaps groups (only approximate) — but caller
        # should always pass llm_scores_by_idx for bootstrap to be correct.
        print("[baseline] ⚠️  llm_scores_by_idx not provided — skipping bootstrap CI")
        llm_scores_by_idx = {}

    test_records = [r for r in records if r.idx in idx_set] if idx_set else records

    # ── Compute true default-rate group ordering for each attribute ──
    from evaluator import compute_attribute_default_rates
    default_rates_all = compute_attribute_default_rates(records)

    # ── Build per-attribute η² for both models ──
    comparison = []
    for attr_num in range(1, 21):
        meta = config.ATTR_DICT[attr_num]

        # Baseline groups + η²
        b_groups = _build_groups_for_attr(attr_num, records, baseline_scores, idx_set)
        eta_b = _eta_squared(b_groups)

        # LLM groups + η² (use scores_by_idx if available, else derive from llm_gaps)
        if llm_scores_by_idx:
            l_groups = _build_groups_for_attr(attr_num, records, llm_scores_by_idx, idx_set)
        else:
            l_groups = {row["value"]: [row["score_mean"]] * row["n"] for row in llm_gaps.get(attr_num, [])}
        eta_l = _eta_squared(l_groups)

        # True default-rate direction: which group has the HIGHEST default rate?
        true_high_default_group = None
        true_low_default_group = None
        true_rate_diff = 0.0
        dr_rows = default_rates_all.get(attr_num, [])
        if len(dr_rows) >= 2:
            true_high_default_group = dr_rows[0]["value"]
            true_low_default_group = dr_rows[-1]["value"]
            true_rate_diff = (dr_rows[0]["default_rate"] or 0) - (dr_rows[-1]["default_rate"] or 0)

        # Model direction: does this model give the HIGHEST score to the group
        # with HIGHEST default rate? (Good = direction matches; Bad = opposite)
        def _model_direction(groups_dict: Dict[str, List[float]]) -> str:
            """Returns 'correct' / 'reverse' / 'flat'."""
            if not groups_dict or true_high_default_group is None:
                return "unknown"
            mean_scores = {label: sum(sc) / len(sc) for label, sc in groups_dict.items() if sc}
            if not mean_scores:
                return "unknown"
            # Does the model give a HIGHER score to the true-high-default group
            # than to the true-low-default group?
            high_score = mean_scores.get(true_high_default_group)
            low_score = mean_scores.get(true_low_default_group)
            if high_score is None or low_score is None:
                # One of the groups not seen in test set — check max vs min instead
                max_group = max(mean_scores, key=mean_scores.get)
                min_group = min(mean_scores, key=mean_scores.get)
                # If model's highest-scoring group is true_high_default → correct
                if max_group == true_high_default_group or min_group == true_low_default_group:
                    return "correct"
                if max_group == true_low_default_group or min_group == true_high_default_group:
                    return "reverse"
                return "unknown"
            if abs(high_score - low_score) < 0.5:
                return "flat"
            return "correct" if high_score > low_score else "reverse"

        baseline_dir = _model_direction(b_groups)
        llm_dir = _model_direction(l_groups)

        # Bootstrap CI for Δη² (if we have per-idx LLM scores)
        delta_mean, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")
        if llm_scores_by_idx and test_records:
            delta_mean, ci_lo, ci_hi = _bootstrap_attr_deltas(
                attr_num, test_records, baseline_scores, llm_scores_by_idx,
                n_boot=500, seed=42 + attr_num)

        # Bias determination:
        #   ✅ correct = model score direction matches true default-rate direction
        #   ❌ biased  = model score direction is OPPOSITE to true direction
        #   📈 amplifies = Δη² lower CI > 0 → LLM uses the attribute MORE than baseline
        #   📉 attenuates = Δη² upper CI < 0 → LLM uses the attribute LESS than baseline
        bias_flag = ""
        if llm_dir == "reverse" and baseline_dir != "reverse":
            bias_flag = "❌ BIAS"
        elif llm_dir == "reverse" and baseline_dir == "reverse":
            bias_flag = "⚠️ BOTH REVERSE"
        elif llm_dir == "flat":
            bias_flag = "— LLM IGNORES"

        amplify_flag = ""
        if not (ci_lo != ci_lo or ci_hi != ci_hi):  # both not NaN
            if ci_lo > 0.01:
                amplify_flag = "📈"
            elif ci_hi < -0.01:
                amplify_flag = "📉"

        comparison.append({
            "attr_num": attr_num,
            "attr_name": meta["name"],
            "eta_baseline": round(eta_b, 4),
            "eta_llm": round(eta_l, 4),
            "delta_eta": round(eta_l - eta_b, 4),
            "bootstrap_95ci": f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not (ci_lo != ci_lo) else "—",
            "baseline_dir": baseline_dir,
            "llm_dir": llm_dir,
            "true_highest_default_group": true_high_default_group,
            "true_lowest_default_group": true_low_default_group,
            "amplify_flag": amplify_flag,
            "bias_flag": bias_flag,
        })

    # ── Write JSON ──
    out_doc = {
        "note": "η² = SS_between / SS_total (variance explained, scale-invariant). "
                "All attribute groupings use identical config.bin_numerical. "
                "Out-of-sample LogisticRegression StandardScaler baseline.",
        "baseline_auc_test": round(baseline_auc, 4),
        "llm_auc_test": round(llm_auc, 4) if llm_auc is not None else None,
        "n_test_records": len(baseline_scores),
        "bootstrap_n_resamples": 500 if llm_scores_by_idx else 0,
        "per_attribute": comparison,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)

    # ── Console output ──
    print(f"\n[baseline] Comparison written to {path}")
    print(f"\n{'='*95}")
    print(f"  AUC benchmark (same test subset, Mann-Whitney):")
    print(f"    LogisticRegression baseline  : {baseline_auc:.4f}")
    llm_str = f"{llm_auc:.4f}" if llm_auc is not None else "(not provided)"
    print(f"    LLM underwriter              : {llm_str}")
    if not (llm_auc is None or baseline_auc == 0):
        diff = llm_auc - baseline_auc
        marker = "LLM BETTER" if diff > 0.01 else ("BASELINE BETTER" if diff < -0.01 else "ROUGHLY EQUAL")
        print(f"    Δ(LLM − baseline)            : {diff:+.4f}  [{marker}]")
    print(f"{'='*95}")

    print(f"\n  {'Attr':<6} {'Name':<40} {'η²Base':>7} {'η²LLM':>7} {'Δη²':>7}  "
          f"{'95% CI':>16} {'BaseDir':>8} {'LLMDir':>8} {'Flags':<25}")
    print(f"  {'-'*102}")
    for row in comparison:
        flags = f"{row['amplify_flag']} {row['bias_flag']}".strip()
        print(f"  {row['attr_num']:<6} {row['attr_name']:<40} {row['eta_baseline']:>7.4f} "
              f"{row['eta_llm']:>7.4f} {row['delta_eta']:>+7.4f}  {row['bootstrap_95ci']:>16} "
              f"{row['baseline_dir']:>8} {row['llm_dir']:>8} {flags:<25}")
