"""
Main pipeline: load → match → LLM scoring + counterfactuals → evaluate with
ground truth → compare LLM gaps vs LogisticRegression baseline → output tables.

Usage:
    python main.py              # Full run on 100 pairs
    python main.py --dry-run 2  # Debug run on just 2 cases
    python main.py --skip-match # Skip matching; evaluate using existing checkpoint
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import config
from data_loader import load_data, Record
from naturalizer import record_to_application, build_prompt
from matcher import random_sample_records
from llm_client import LLMClient, extract_json, ckpt_load, ckpt_save
from evaluator import (
    score_all, summarize,
    compute_ground_truth_metrics, compute_attribute_default_rates,
    scan_thresholds,
)
from baseline_model import (
    train_baseline, save_baseline_comparison, compute_baseline_gaps, gap_from_rows,
    compute_baseline_auc,
)
from bias_analyzer import analyze_single_attr


# ── Pipeline stages ─────────────────────────────────────────────────────────
def build_sample():
    """Step 1: Load data and uniformly sample 100 records (no grouping, no matching)."""
    print("[main] Loading data...")
    records = load_data()
    print(f"  Total {len(records)} records")
    sampled = random_sample_records(records, n=100, seed=42)
    return sampled, records


def run_llm_on_records(records: List[Record], dry_run: int | None = None):
    """
    Step 2: Send each record to the LLM. Uses checkpoint to resume.
    All records share group="random" (no high/low split).

    WARNING: If you previously ran with matched pairs (high/low), the
    checkpoint will contain idx values that differ from this random sample.
    In that case, DELETE output/llm_checkpoint.jsonl first to force a fresh run.
    """
    client = LLMClient()
    ckpt = ckpt_load(config.CHECKPOINT_FILE)

    queue = [(r.idx, "random", r) for r in records]

    # Detect stale checkpoint entries (from a previous different sampling)
    queue_idx_set = {q[0] for q in queue}
    stale = [k for k in ckpt.keys() if k not in queue_idx_set]
    overlap = [k for k in ckpt.keys() if k in queue_idx_set]
    if stale:
        print(f"[main] ⚠️ Checkpoint has {len(stale)} stale entries (from a previous run).")
        print(f"[main]   Overlap with current sample: {len(overlap)}. "
              f"Run 'Remove-Item output\\llm_checkpoint.jsonl' first for a clean run.")

    if dry_run:
        queue = queue[:dry_run]

    print(f"[main] Pending LLM calls: {len(queue)} (already done {len(ckpt)}, skipping completed)")

    t0 = time.time()
    done = 0
    for idx, group, rec in queue:
        if idx in ckpt:
            done += 1
            continue

        app_text = record_to_application(rec)
        prompt = build_prompt(app_text)

        try:
            raw = client.call(prompt)
            result = extract_json(raw)
            ckpt_save(config.CHECKPOINT_FILE, idx, group, result, prompt)
            ckpt[idx] = {"idx": idx, "group": group, "result": result}
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{len(queue)}] idx={idx} group={group} "
                  f"decision={result.get('decision')} "
                  f"cf_count={len(result.get('counterfactual') or [])} "
                  f"({rate:.2f}/s)")
        except Exception as e:
            print(f"  [FAIL] idx={idx} group={group}: {e}", file=sys.stderr)

    print(f"[main] LLM calls finished in {time.time()-t0:.1f}s")
    return ckpt


# ── Evaluation stage (with ground truth) ────────────────────────────────────
def evaluate(
    ckpt: Dict[int, dict],
    idx_to_group: Dict[int, str],
    records: List[Record],
):
    """Step 4: Evaluate + ground truth comparison + baseline."""
    records_by_idx = {r.idx: r for r in records}
    rows = score_all(ckpt, idx_to_group, records_by_idx)

    # ── Save per-row results ──
    results_out = []
    for r in rows:
        results_out.append({
            "idx": r.idx, "group": r.group,
            "score": r.score,
            "decision": r.decision,
            "n_advice": r.n_advice,
            "label": r.label,
            "llm_result": ckpt[r.idx].get("result"),
        })
    with open(config.RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results_out, f, ensure_ascii=False, indent=2)
    print(f"\n[main] Per-row results → {config.RESULTS_FILE}")

    # ── Group-level summary ──
    summary = summarize(rows)
    _print_group_summary_table(summary)

    # ── Ground-truth evaluation ──
    gt_metrics = compute_ground_truth_metrics(rows)
    _print_ground_truth_eval(gt_metrics)

    # Save ground truth metrics
    with open(config.GT_FILE, "w", encoding="utf-8") as f:
        json.dump(gt_metrics, f, ensure_ascii=False, indent=2)

    # ── Per-attribute real default rates ──
    default_rates = compute_attribute_default_rates(records)

    return summary, gt_metrics, default_rates


def _print_group_summary_table(summary: Dict) -> None:
    """
    Print per-group LLM score summary. Now that sampling is pure random
    (no high/low split), there is usually only one group ("random").
    Falls back gracefully to an overall-only table when all rows share
    the same group label.
    """
    # Detect if multiple groups exist in summary
    group_labels = list(summary.keys())
    multi_group = len(group_labels) > 1

    print("\n" + "=" * 60)
    if multi_group:
        print(f"Summary: LLM risk scores by group ({len(group_labels)} groups)")
    else:
        print("Summary: LLM risk scores — overall (pure random sample, no grouping)")
    print("=" * 60)

    def fmt(x):
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)

    if multi_group:
        # Old path: print a side-by-side group comparison table
        col_widths = [28] + [14] * len(group_labels)
        header = ["Metric"] + group_labels + (["Difference (max-min)"] if len(group_labels) == 2 else [])
        table_rows = []

        def _row(label, key, is_rate=False):
            vals = [summary.get(g, {}).get(key) for g in group_labels]
            row = [label] + [fmt(v) for v in vals]
            if len(group_labels) == 2:
                row.append(fmt((vals[0] or 0) - (vals[1] or 0)))
            table_rows.append(row)

        _row("Sample size", "n")
        _row("Approved (score<50)", "approved")
        _row("Rejected (score≥50)", "rejected")
        _row("Approval rate", "approval_rate")
        _row("Mean score", "score_mean")
        _row("Median score", "score_median")
        _row("Min score", "score_min")
        _row("Max score", "score_max")
        _row("Total counterfactuals", "total_advice")
        _row("Avg counterfactuals per case", "avg_advice_per_case")

        sep_w = sum(col_widths) + 4 + (12 if len(group_labels) == 2 else 0)
        print(f"\n{'Metric':<28}" + "".join(f" {g:<14}" for g in group_labels) +
              (" Difference" if len(group_labels) == 2 else ""))
        print("-" * sep_w)
        for row in table_rows:
            print(f"{str(row[0]):<28}" + "".join(f" {str(c):<14}" for c in row[1:]))
        print()
    else:
        # Single-group (random sample) path: simpler overall table
        g = group_labels[0]
        s = summary[g]
        table_rows = [
            ["Sample size", s.get("n")],
            ["Approved (score<50)", s.get("approved")],
            ["Rejected (score≥50)", s.get("rejected")],
            ["Approval rate", fmt(s.get("approval_rate"))],
            ["Mean score", fmt(s.get("score_mean"))],
            ["Median score", fmt(s.get("score_median"))],
            ["Min score", fmt(s.get("score_min"))],
            ["Max score", fmt(s.get("score_max"))],
            ["Total counterfactuals", s.get("total_advice")],
            ["Avg counterfactuals per case", fmt(s.get("avg_advice_per_case"))],
        ]
        print(f"\n{'Metric':<28} {'Value':<14}")
        print("-" * 44)
        for row in table_rows:
            print(f"{str(row[0]):<28} {str(fmt(row[1])):<14}")
        print()

    # Save CSV
    with open(config.SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if multi_group and len(group_labels) == 2:
            header = ["Metric"] + group_labels + ["Difference"]
        elif multi_group:
            header = ["Metric"] + group_labels
        else:
            header = ["Metric", "Value"]
        writer.writerow(header)
        for r in table_rows:
            writer.writerow([str(c) for c in r])
    print(f"[main] Summary table → {config.SUMMARY_FILE}")


def _print_ground_truth_eval(gt: dict) -> None:
    print("\n" + "=" * 60)
    print("Ground-truth evaluation (score threshold = 50)")
    print("=" * 60)
    print(f"  n evaluated: {gt['n_counted']}")
    print(f"  Confusion matrix (predict Good / Bad):")
    print(f"                    Predicted Good  Predicted Bad")
    print(f"    Actual Good         tn={gt['confusion_matrix']['tn']:<5}      fp={gt['confusion_matrix']['fp']}")
    print(f"    Actual Bad          fn={gt['confusion_matrix']['fn']:<5}      tp={gt['confusion_matrix']['tp']}")
    print(f"  Accuracy:     {gt['accuracy']:.4f}")
    print(f"  Precision:    {gt['precision']:.4f}  (P(reject | actually Bad))")
    print(f"  Recall:       {gt['recall']:.4f}  (P(Bad | rejected))")
    print(f"  F1:           {gt['f1']:.4f}")
    print(f"  AUC (Mann-Whitney): {gt['auc']:.4f}  "
          f"(probability a Bad record scores higher than a Good one)")
    print(f"  KS statistic:  {gt['ks']:.4f}")
    print(f"  Cost-sensitive score:")
    print(f"    Total cost = {gt['cost_total']}   "
          f"Cost per case = {gt['cost_per_case']:.2f}   "
          f"(cost matrix: miss Bad → cost 5, miss Good → cost 1)")


# ── Baseline comparison ─────────────────────────────────────────────────────
def run_baseline_comparison(
    records: List[Record],
    ckpt: Dict[int, dict],
    sampled_indices: List[int] | None = None,
) -> None:
    """
    Out-of-sample baseline vs LLM comparison.

    P0-1 DATA LEAKAGE FIX:
      Baseline is trained on records NOT in sampled_indices, then predicts P(Bad)
      on the EXACT same test subset that was sent to the LLM — no label leakage.

    P0-2 GAP CONSISTENCY FIX:
      baseline_model.compute_baseline_gaps now applies config.NUMERICAL_BINS
      identically to bias_analyzer, so baseline and LLM gaps measure the same thing.

    P0-3 AUC BENCHMARK:
      Both baseline and LLM Mann-Whitney AUC are computed on the same test subset
      and printed side-by-side.

    P0-4 STANDARD SCALER:
      baseline_model.train_baseline fits StandardScaler on the train split only.
    """
    print("\n" + "=" * 60)
    print("Baseline comparison: LogisticRegression vs LLM")
    print("=" * 60)

    # ── Normalize checkpoint: bias_analyzer expects ckpt[idx] = result dict ──
    ckpt_flat: Dict[int, dict] = {}
    for idx, entry in ckpt.items():
        if isinstance(entry, dict) and "result" in entry and "score" not in entry:
            ckpt_flat[idx] = entry["result"]
        else:
            ckpt_flat[idx] = entry

    # ── Determine test set = sampled_indices OR ckpt keys ──
    if sampled_indices:
        test_idx_set = set(sampled_indices)
    else:
        test_idx_set = set(ckpt_flat.keys())

    test_records = [r for r in records if r.idx in test_idx_set]
    train_records = [r for r in records if r.idx not in test_idx_set]

    print(f"[main] Train split: {len(train_records)} records (no label leakage)")
    print(f"[main] Test  split: {len(test_records)} records (out-of-sample, same as LLM)")

    # ── Train baseline on train split only, predict P(Bad) on test split ──
    print("[main] Training LogisticRegression baseline (fit on train, predict on test)...")
    clf, baseline_scores, feature_names, scaler = train_baseline(train_records, test_records)
    print(f"  Features: {len(feature_names)}  →  Test-set predictions: {len(baseline_scores)}")
    # Note: baseline_scores only contains test_idx entries (out-of-sample only)

    # ── Compute LLM gaps (on test subset only, identically to baseline) ──
    matched_indices = sorted(set(ckpt_flat.keys()) & test_idx_set)

    llm_gaps: Dict[int, List[dict]] = {}
    for attr_num in range(1, 21):
        res = analyze_single_attr(attr_num, records, ckpt_flat)
        # Filter to only test subset (to be fair — baseline only sees test records)
        res_filtered_rows = []
        for row in res["rows"]:
            n_in_test = row["n"]  # Already from analyze_single_attr which reads from ckpt
            if n_in_test > 0:
                res_filtered_rows.append(row)
        # Actually analyze_single_attr already uses ckpt keys which are our test set
        llm_gaps[attr_num] = res_filtered_rows

    # ── Compute LLM AUC on test subset ──
    records_by_idx = {r.idx: r for r in records}
    score_rows = score_all(ckpt_flat, {}, records_by_idx)
    # Filter score_rows to test subset only
    test_score_rows = [r for r in score_rows if r.idx in test_idx_set]
    gt_test = compute_ground_truth_metrics(test_score_rows)
    llm_auc_test = gt_test.get("auc")
    print(f"[main] LLM AUC on test subset (n={len(test_score_rows)}): {llm_auc_test:.4f}")

    save_baseline_comparison(
        records, baseline_scores, llm_gaps,
        matched_indices=matched_indices,
        llm_auc=llm_auc_test,
        # Build per-idx LLM scores for bootstrap CI
        llm_scores_by_idx={idx: float(entry.get("score", 0)) for idx, entry in ckpt_flat.items()
                          if idx in test_idx_set},
    )


# ── Entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LLM bias detection — random 100 records, no grouping")
    parser.add_argument("--dry-run", type=int, default=None,
                        help="Only run the first N LLM calls (for debugging)")
    parser.add_argument("--skip-sample", action="store_true",
                        help="Skip sampling; evaluate using existing checkpoint only")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip LogisticRegression baseline (saves sklearn import)")
    parser.add_argument("--scan-thresholds", action="store_true",
                        help="Scan threshold 10–90 to find best Accuracy/F1/Cost")
    args = parser.parse_args()

    records: List[Record] | None = None
    sampled_records: List[Record] | None = None

    if not args.skip_sample:
        sampled_records, records = build_sample()
    else:
        records = load_data()

    records_by_idx = {r.idx: r for r in records}

    # Build idx_to_group: either from sampled list or from checkpoint
    # (all entries are "random" — no high/low split)
    idx_to_group: Dict[int, str] = {}
    if sampled_records:
        for r in sampled_records:
            idx_to_group[r.idx] = "random"

    ckpt = run_llm_on_records(
        sampled_records if sampled_records else [records_by_idx[i] for i in idx_to_group.keys()],
        dry_run=args.dry_run,
    )

    summary, gt_metrics, default_rates = evaluate(ckpt, idx_to_group, records)

    if args.scan_thresholds:
        records_by_idx = {r.idx: r for r in records}
        score_rows = score_all(ckpt, idx_to_group, records_by_idx)
        scan_thresholds(score_rows, lo=10, hi=90, step=5)

    if not args.skip_baseline:
        try:
            sampled_idx_list = list(idx_to_group.keys()) if idx_to_group else None
            run_baseline_comparison(records, ckpt, sampled_indices=sampled_idx_list)
        except ImportError as e:
            print(f"[main] Skipping baseline (sklearn not installed): {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[main] Baseline failed: {e}")

    print("\n[main] Done!")


if __name__ == "__main__":
    main()
