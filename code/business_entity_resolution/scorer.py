"""
F_0.5 scorer for the Business Entity Resolution challenge.

Implements exactly what the brief specifies:

    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

computed PER Source 1 entity, then macro-averaged across every Source 1
entity in the ground truth (singletons included).

Per-entity edge cases (this is where scorers usually get it wrong, so it's
spelled out):
  - true=[]  , pred=[]  -> F0.5 = 1.0   (correctly predicted singleton)
  - true=[]  , pred!=[] -> F0.5 = 0.0   (false merge on a singleton;
                                          precision=0, so the formula's
                                          numerator is 0 - handled naturally,
                                          just called out explicitly here)
  - true!=[] , pred=[]  -> F0.5 = 0.0   (recall=0 -> numerator 0)
  - true!=[] , pred!=[] , no overlap -> F0.5 = 0.0
  - precision=recall=0                 -> F0.5 = 0.0 (avoid 0/0 -> define as 0,
                                          matches the "predict nothing when
                                          you should predict nothing" reward
                                          structure: the ONLY way to score 1.0
                                          on a singleton is an empty prediction)

This module is intentionally dependency-light (pandas only) and is meant to
be called by anyone: A/B/C can each run it against their own val predictions
without needing the rest of D's harness.
"""
from __future__ import annotations

import argparse
import sys
import os

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.io import read_ground_truth, parse_id_list  # noqa: E402


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def score_entity(true_ids: set[str], pred_ids: set[str]) -> tuple[float, float, float]:
    """Returns (precision, recall, f0.5) for one Source 1 entity."""
    if not true_ids and not pred_ids:
        return 1.0, 1.0, 1.0  # correct singleton: defined as perfect, not undefined
    if not pred_ids:
        return 0.0, 0.0, 0.0
    if not true_ids:
        return 0.0, 0.0, 0.0  # any prediction on a true singleton is 0 precision

    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    return precision, recall, f_beta(precision, recall, beta=0.5)


def score(
    predictions_path: str,
    ground_truth_path: str,
    id_col: str = "matched_entity_ids",
    per_entity_out: str | None = None,
) -> dict:
    gt = read_ground_truth(ground_truth_path)
    pred = pd.read_csv(predictions_path, sep="\t", dtype=str, keep_default_na=False)

    if "source1_entity_id" not in pred.columns or id_col not in pred.columns:
        raise ValueError(
            f"{predictions_path} must have columns "
            f"['source1_entity_id', '{id_col}'], found {list(pred.columns)}"
        )

    if pred["source1_entity_id"].duplicated().any():
        dupes = pred.loc[pred["source1_entity_id"].duplicated(), "source1_entity_id"].tolist()
        raise ValueError(f"Duplicate source1_entity_id rows in predictions: {dupes[:10]} ...")

    pred_map = dict(zip(pred["source1_entity_id"], pred[id_col].apply(parse_id_list)))

    rows = []
    n_missing_pred = 0
    for _, r in gt.iterrows():
        s1_id = r["source1_entity_id"]
        true_ids = set(parse_id_list(r["matched_entity_ids"]))
        if s1_id not in pred_map:
            n_missing_pred += 1
            pred_ids: set[str] = set()
        else:
            pred_ids = set(pred_map[s1_id])
        p, rcl, f = score_entity(true_ids, pred_ids)
        rows.append({
            "source1_entity_id": s1_id,
            "n_true": len(true_ids),
            "n_pred": len(pred_ids),
            "precision": p,
            "recall": rcl,
            "f0.5": f,
        })

    detail = pd.DataFrame(rows)
    macro_f05 = detail["f0.5"].mean()
    macro_precision = detail["precision"].mean()
    macro_recall = detail["recall"].mean()

    is_singleton = detail["n_true"] == 0
    singleton_acc = (detail.loc[is_singleton, "f0.5"] == 1.0).mean() if is_singleton.any() else float("nan")

    if per_entity_out:
        detail.to_csv(per_entity_out, sep="\t", index=False)

    result = {
        "n_entities": len(detail),
        "n_missing_from_predictions": n_missing_pred,
        "macro_f0.5": macro_f05,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "n_singletons": int(is_singleton.sum()),
        "singleton_accuracy": singleton_acc,
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="Path to matching_results.tsv")
    ap.add_argument("--ground-truth", required=True, help="Path to (val_)ground_truth.tsv")
    ap.add_argument("--per-entity-out", default=None, help="Optional: dump per-entity P/R/F0.5 to this path")
    args = ap.parse_args()

    result = score(args.predictions, args.ground_truth, per_entity_out=args.per_entity_out)

    print("=" * 50)
    print(f"Entities scored:         {result['n_entities']:,}")
    if result["n_missing_from_predictions"]:
        print(f"  WARNING: {result['n_missing_from_predictions']:,} entities missing "
              f"from predictions (scored as empty prediction)")
    print(f"Macro Precision:         {result['macro_precision']:.4f}")
    print(f"Macro Recall:            {result['macro_recall']:.4f}")
    print(f"Macro F0.5:              {result['macro_f0.5']:.4f}")
    print(f"Singletons in GT:        {result['n_singletons']:,}")
    print(f"Singleton accuracy:      {result['singleton_accuracy']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
