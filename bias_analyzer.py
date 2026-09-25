"""
Bias analyzer — revised with:
  (a) numerical-attribute binning to avoid n=1 groups
  (b) η² variance-explained + Mann-Whitney U tests (Bonferroni-corrected)
      in addition to max-min gap
  (c) weight-aware asymmetry ratio for top_factors
  (d) counterfactual content categorization with stereotype-laden flagging

Usage:
    python bias_analyzer.py
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict, Counter
from pathlib import Path
import statistics

import config
from data_loader import load_data, Record


# ── Data loading ─────────────────────────────────────────────────────────────
def load_checkpoint() -> dict[int, dict]:
    out = {}
    if not config.CHECKPOINT_FILE.exists():
        return out
    with open(config.CHECKPOINT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                out[rec["idx"]] = rec["result"]
            except Exception:
                pass
    return out


def get_score(result: dict) -> float | None:
    s = result.get("score")
    if s is None:
        return None
    return float(s)


def get_decision(result: dict) -> str | None:
    s = get_score(result)
    if s is None:
        return None
    return "approve" if s < config.SCORE_THRESHOLD else "reject"


# ── Binning helpers ──────────────────────────────────────────────────────────
def _bin_numerical(value: float, attr_num: int, all_values: List[float] | None = None) -> str:
    """Mirror — delegates to config.bin_numerical (single source of truth)."""
    return config.bin_numerical(value, attr_num, all_values)


# ── Statistical tests ───────────────────────────────────────────────────────
def _mann_whitney_u(scores_a: List[float], scores_b: List[float]) -> tuple[float, float]:
    """
    Mann-Whitney U two-sample test. Returns (U statistic, approximate p-value).
    Uses normal approximation for n_a, n_b > 10; exact rank computation otherwise.
    """
    na, nb = len(scores_a), len(scores_b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")

    combined = [(s, 0) for s in scores_a] + [(s, 1) for s in scores_b]
    combined.sort(key=lambda x: x[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_a = sum(ranks[k] for k in range(len(combined)) if combined[k][1] == 0)
    u_a = rank_a - na * (na + 1) / 2
    u_b = na * nb - u_a
    u = min(u_a, u_b)

    # Normal approximation
    mu = na * nb / 2.0
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    z = (u - mu) / sigma if sigma > 0 else 0.0
    # Two-tailed p-value via erf
    p = math.erfc(abs(z) / math.sqrt(2))
    return u, p


def _eta_squared(groups_scores: Dict[str, List[float]]) -> float:
    """
    η² = SS_between / SS_total. Range [0, 1].
    0 = attribute explains none of the variance; 1 = explains all.
    """
    all_scores = [s for scores in groups_scores.values() for s in scores]
    if len(all_scores) < 2:
        return 0.0
    grand_mean = sum(all_scores) / len(all_scores)

    ss_total = sum((s - grand_mean) ** 2 for s in all_scores)
    ss_between = 0.0
    for label, scores in groups_scores.items():
        if not scores:
            continue
        n = len(scores)
        group_mean = sum(scores) / n
        ss_between += n * (group_mean - grand_mean) ** 2

    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


# ── Single-attribute analysis (revised) ──────────────────────────────────────
def analyze_single_attr(
    attr_num: int, records: List[Record], ckpt: dict,
    min_group_n: int = 3,
) -> dict:
    """
    Full single-attribute analysis returning:
      - rows: list of per-group stats
      - gap: max - min mean score
      - eta_squared: variance explained
      - pairwise_mw: list of (group_a, group_b, U, p) for pairs where n ≥ min_group_n
      - note: any caveat (e.g., "numerical — binned")
    """
    meta = config.ATTR_DICT[attr_num]
    attr_idx = attr_num - 1  # 0-based
    numerical = meta["type"] == "numerical"

    # Collect all values for binning
    all_vals: List[float] = []
    for rec in records:
        if numerical:
            try:
                all_vals.append(float(rec.attrs[attr_idx]))
            except ValueError:
                pass

    # Build groups
    groups: Dict[str, List[float]] = defaultdict(list)
    group_decisions: Dict[str, List[str]] = defaultdict(list)

    for rec in records:
        if rec.idx not in ckpt:
            continue
        raw_val = rec.attrs[attr_idx]
        score = get_score(ckpt[rec.idx])
        decision = get_decision(ckpt[rec.idx])

        if numerical:
            try:
                label = _bin_numerical(float(raw_val), attr_num, all_vals)
            except ValueError:
                continue
        else:
            label = meta["values"].get(raw_val, raw_val)

        if score is not None:
            groups[label].append(score)
        if decision is not None:
            group_decisions[label].append(decision)

    if not groups:
        return {"rows": [], "gap": 0.0, "eta_squared": 0.0, "pairwise_mw": []}

    rows = []
    for label, scores in groups.items():
        if not scores:
            continue
        n = len(scores)
        approvals = sum(1 for d in group_decisions.get(label, []) if d == "approve")
        mean_s = sum(scores) / n
        rows.append({
            "value": label,
            "n": n,
            "score_mean": mean_s,
            "score_median": statistics.median(scores),
            "score_min": min(scores),
            "score_max": max(scores),
            "approval_rate": round(approvals / n, 4) if n else None,
        })
    rows.sort(key=lambda r: r["score_mean"])

    # Gap
    gap = rows[-1]["score_mean"] - rows[0]["score_mean"] if len(rows) >= 2 else 0.0

    # η²
    eta = _eta_squared({r["value"]: groups[r["value"]] for r in rows})

    # Pairwise Mann-Whitney with Bonferroni
    pairwise = []
    eligible = [r for r in rows if r["n"] >= min_group_n]
    n_comparisons = len(eligible) * (len(eligible) - 1) // 2
    if n_comparisons > 0:
        alpha = 0.05
        bonf_threshold = alpha / n_comparisons
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                a = eligible[i]
                b = eligible[j]
                u, p = _mann_whitney_u(groups[a["value"]], groups[b["value"]])
                pairwise.append({
                    "group_a": a["value"],
                    "group_b": b["value"],
                    "n_a": a["n"],
                    "n_b": b["n"],
                    "U": round(u, 2) if not math.isnan(u) else None,
                    "p_value": round(p, 6) if not math.isnan(p) else None,
                    "significant_bonferroni": p < bonf_threshold if not math.isnan(p) else False,
                })

    note = ""
    if numerical:
        note = "numerical attribute — values binned"

    return {
        "rows": rows,
        "gap": round(gap, 2),
        "eta_squared": round(eta, 4),
        "pairwise_mw": pairwise,
        "note": note,
        "min_group_n": min_group_n,
    }


# ── Print helpers ─────────────────────────────────────────────────────────────
def _print_attr_table(attr_num: int, result: dict) -> None:
    meta = config.ATTR_DICT[attr_num]
    rows = result["rows"]
    if not rows:
        return
    print(f"\n{'─'*76}")
    suffix = f" ({result['note']})" if result["note"] else ""
    print(f"Attr {attr_num:>2}: {meta['name']}  (type={meta['type']}){suffix}")
    print(f"{'─'*76}")
    print(f"  {'Value':<32} {'n':>4}  {'mean':>7}  {'median':>7}  "
          f"{'min':>6}  {'max':>6}  {'approve%':>8}")
    print(f"  {'─'*32} {'─'*4} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*8}")

    for r in rows:
        ar = r["approval_rate"]
        ar_str = f"{ar*100:.1f}%" if ar is not None else "—"
        print(f"  {r['value']:<32} {r['n']:>4}  "
              f"{r['score_mean']:>7.1f}  {r['score_median']:>7.1f}  "
              f"{r['score_min']:>6.1f}  {r['score_max']:>6.1f}  {ar_str:>8}")

    # Stats summary
    print(f"  {'─'*72}")
    print(f"  Gap (max−min): {result['gap']:.2f}    "
          f"η² (variance explained): {result['eta_squared']:.4f}")

    # Flag
    eta = result["eta_squared"]
    gap = result["gap"]
    if gap >= 10 and eta >= 0.08:
        flag = "⚠️ SUBSTANTIAL"
    elif gap >= 5 and eta >= 0.03:
        flag = "🔸 MARGINAL"
    else:
        flag = "   NEGLIGIBLE"
    print(f"  Assessment: {flag}")

    # Significant pairwise comparisons
    sig_pairs = [p for p in result["pairwise_mw"] if p["significant_bonferroni"]]
    if sig_pairs:
        print(f"  Significant pairwise differences (Bonferroni p<0.05):")
        for p in sig_pairs[:5]:   # at most 5
            print(f"    {p['group_a']} vs {p['group_b']}: p={p['p_value']:.6f}")


# ── Factor attribution (weight-aware) ────────────────────────────────────────
def analyze_top_factors(ckpt: dict) -> dict:
    """
    Weight-aware asymmetry ratio. Each nomination contributes:
      weight to risk_total[attr] or strength_total[attr],
      count  to risk_count[attr] or strength_count[attr].

    Asymmetry = (risk_w − strength_w) / (risk_w + strength_w).
    """
    risk_w: Counter = Counter()      # weighted counts
    strength_w: Counter = Counter()
    risk_count: Counter = Counter()  # raw counts
    strength_count: Counter = Counter()
    total_noms = 0

    for idx, result in ckpt.items():
        factors = result.get("top_factors") or []
        for f in factors:
            attr_name = (f.get("attribute") or "").strip().lower()
            direction = (f.get("direction") or "").strip().lower()
            weight = int(f.get("weight", 1))
            if weight < 1 or weight > 3:
                weight = 1
            if not attr_name or direction not in ("risk", "strength"):
                continue
            total_noms += 1
            if direction == "risk":
                risk_w[attr_name] += weight
                risk_count[attr_name] += 1
            else:
                strength_w[attr_name] += weight
                strength_count[attr_name] += 1

    all_attrs = set(risk_w.keys()) | set(strength_w.keys())
    rows = []
    for attr in all_attrs:
        rw = risk_w.get(attr, 0)
        sw = strength_w.get(attr, 0)
        rc = risk_count.get(attr, 0)
        sc = strength_count.get(attr, 0)
        total_w = rw + sw
        if total_w > 0:
            asym = (rw - sw) / total_w
        else:
            asym = 0.0
        rows.append({
            "attribute": attr,
            "risk_weighted": rw,
            "strength_weighted": sw,
            "risk_count": rc,
            "strength_count": sc,
            "net_weighted": rw - sw,
            "asymmetry": round(asym, 4),
        })
    rows.sort(key=lambda x: -abs(x["asymmetry"]))

    return {"rows": rows, "total_nominations": total_noms}


# ── Counterfactual content analysis ─────────────────────────────────────────
def _classify_counterfactual(text: str) -> str:
    """Match a counterfactual suggestion against config.COUNTERFACTUAL_CATEGORIES."""
    lower = text.lower().strip()
    for cat, patterns in config.COUNTERFACTUAL_CATEGORIES:
        if cat == "other":
            continue
        for pat in patterns:
            if pat in lower:
                return cat
    return "other"


def analyze_counterfactuals(ckpt: dict) -> dict:
    """
    Classify every counterfactual suggestion. Outputs:
      - category_counts: Counter of categories
      - stereotype_suggestions: list of (idx, category, original_text)
        where category == "demographic_intervention"
      - per_demographic_breakdown: for each protected group (Attr 9 / Attr 20),
        what categories of counterfactuals they receive
    """
    from data_loader import load_data

    # Build idx → record map for demographic breakdown
    records = load_data()
    records_by_idx = {r.idx: r for r in records}

    cat_counter: Counter = Counter()
    stereotype_items: list = []
    # demographic → category → count
    demo_cats: Dict[str, Counter] = defaultdict(lambda: Counter())

    for idx, result in ckpt.items():
        rec = records_by_idx.get(idx)
        cfs = result.get("counterfactual") or []
        for cf_text in cfs:
            cat = _classify_counterfactual(cf_text)
            cat_counter[cat] += 1
            if cat == "demographic_intervention":
                stereotype_items.append({
                    "idx": idx,
                    "suggestion": cf_text,
                })

            # Demographic breakdown
            if rec:
                # Attr 9: sex × marital status
                attr9_label = config.ATTR_DICT[9]["values"].get(rec.attrs[8], rec.attrs[8])
                attr20_label = config.ATTR_DICT[20]["values"].get(rec.attrs[19], rec.attrs[19])
                attr13_binned = None
                try:
                    attr13_binned = _bin_numerical(float(rec.attrs[12]), 13)
                except ValueError:
                    pass

                demo_cats[f"sex_marital: {attr9_label}"][cat] += 1
                demo_cats[f"foreign: {attr20_label}"][cat] += 1
                if attr13_binned:
                    demo_cats[f"age_bin: {attr13_binned}"][cat] += 1

    return {
        "total_counterfactuals": sum(cat_counter.values()),
        "category_counts": dict(cat_counter),
        "stereotype_suggestions": stereotype_items,
        "n_stereotype": len(stereotype_items),
        "demographic_breakdown": {k: dict(v) for k, v in demo_cats.items()},
    }


def _print_counterfactual_report(cf_report: dict) -> None:
    print("\n" + "=" * 76)
    print("Part 4: Counterfactual content analysis")
    print("=" * 76)

    total = cf_report["total_counterfactuals"]
    print(f"\n  Total counterfactuals analyzed: {total}")
    print(f"  {'Category':<35} {'Count':>6}  {'Pct':>7}")
    print(f"  {'─'*35} {'─'*6} {'─'*7}")
    for cat, count in sorted(cf_report["category_counts"].items(), key=lambda x: -x[1]):
        pct = count / total * 100 if total else 0
        print(f"  {cat:<35} {count:>6}  {pct:>6.1f}%")

    print(f"\n  Stereotype-laden suggestions (demographic intervention): "
          f"{cf_report['n_stereotype']}")
    if cf_report["stereotype_suggestions"]:
        print(f"  Examples:")
        for item in cf_report["stereotype_suggestions"][:5]:
            print(f"    idx={item['idx']}: \"{item['suggestion']}\"")

    # Top demographic breakdowns
    print(f"\n  Category distribution by demographic group (top 8):")
    # Pick groups with most counterfactuals total
    totals = {k: sum(v.values()) for k, v in cf_report["demographic_breakdown"].items()}
    for group, _ in sorted(totals.items(), key=lambda x: -x[1])[:8]:
        cats = cf_report["demographic_breakdown"][group]
        # Normalize within group
        gtotal = sum(cats.values())
        top_cat, top_count = max(cats.items(), key=lambda x: x[1])
        print(f"    {group:<45} n={gtotal}  top_cat={top_cat} ({top_count/gtotal*100:.0f}%)")


# ── Top 5 by eta (replaces top 5 by gap) ─────────────────────────────────────
def _rank_attributes(all_attr_results: Dict[int, dict], metric: str = "eta_squared") -> List[tuple]:
    """Rank all attributes by the chosen metric; return sorted list of (attr_num, score)."""
    ranked = []
    for attr_num, res in all_attr_results.items():
        ranked.append((attr_num, res[metric]))
    ranked.sort(key=lambda x: -x[1])
    return ranked


# ── Main entry ───────────────────────────────────────────────────────────────
def main():
    records = load_data()
    ckpt = load_checkpoint()
    print(f"Total {len(records)} raw records, {len(ckpt)} processed in checkpoint")

    # ── Part 1: Per-attribute analysis with full stats ──
    print("\n" + "=" * 76)
    print("Part 1: Per-attribute score analysis (with binning + statistical tests)")
    print("=" * 76)

    all_attr_results: Dict[int, dict] = {}
    for attr_num in range(1, 21):
        res = analyze_single_attr(attr_num, records, ckpt)
        all_attr_results[attr_num] = res
        _print_attr_table(attr_num, res)

    # ── Part 2: Top 5 by η² ──
    print("\n" + "=" * 76)
    print("Part 2: Top 5 attributes by η² (variance explained)")
    print("=" * 76)
    for attr_num, eta in _rank_attributes(all_attr_results, "eta_squared")[:5]:
        meta = config.ATTR_DICT[attr_num]
        res = all_attr_results[attr_num]
        rows = res["rows"]
        low, high = rows[0], rows[-1]
        print(f"\n  Attr {attr_num}: {meta['name']}")
        print(f"    η² = {eta:.4f}    Gap = {res['gap']:.2f}")
        print(f"    Highest-scoring group: {high['value']:<30} mean={high['score_mean']:.1f}  n={high['n']}")
        print(f"    Lowest-scoring group:  {low['value']:<30} mean={low['score_mean']:.1f}  n={low['n']}")
        # Significant pairwise
        sig = [p for p in res["pairwise_mw"] if p["significant_bonferroni"]]
        if sig:
            print(f"    Significant pairs (Bonferroni): {len(sig)}")

    # ── Part 3: Weight-aware factor attribution ──
    print("\n" + "=" * 76)
    print("Part 3: Top-factors (weighted — each nomination carries weight 1–3)")
    print("=" * 76)
    tf = analyze_top_factors(ckpt)
    print(f"\n  Total weighted nominations: {tf['total_nominations']}")
    print(f"  {'Attribute':<40} {'risk_w':>7} {'strength_w':>11} {'net_w':>7} {'asym':>7}")
    print(f"  {'─'*40} {'─'*7} {'─'*11} {'─'*7} {'─'*7}")
    for r in tf["rows"]:
        asym_str = f"{r['asymmetry']:+.3f}"
        print(f"  {r['attribute']:<40} {r['risk_weighted']:>7} {r['strength_weighted']:>11}  "
              f"{r['net_weighted']:>+7} {asym_str:>7}")
    print(f"\n  Asymmetry interpretation: +1 = pure risk flag (never a strength), "
          f"−1 = pure strength flag (never a risk). Values near 0 = symmetric.")

    # ── Part 4: Counterfactual content analysis ──
    cf_report = analyze_counterfactuals(ckpt)
    _print_counterfactual_report(cf_report)


if __name__ == "__main__":
    main()
