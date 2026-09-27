"""
validate_topk_v52.py

Correct replacement for the old validate_topk.py (V5.1 K-sweep).

The old script rebuilt the full S2+S3 blocking index from scratch every
time it ran, using V5.1 keys and its own separate inline normalize_record.
That's redundant now: build_candidates_v52.py already builds the index
once (V5.2 keys, inverted-index, real normalize_record) and writes out
per-S1 candidates with a score (number of distinct block types shared).

This script does NOT rebuild anything. It just:
  1. Loads the *_scored.tsv that build_candidates_v52.py already produced.
  2. Loads ground truth for the same S1 entities.
  3. For each K in K_VALUES, keeps only the top-K scored candidates per S1
     entity and recomputes recall -- exactly what the old sweep did, but
     against real V5.2 output instead of a rebuilt V5.1 index.

Usage:
    python scripts\\validate_topk_v52.py \
        --scored candidate_pairs_trial_scored.tsv \
        --ground_truth datasets\\train\\train_ground_truth.tsv

Note: only S1 entities present in the scored file are evaluated, so run
this against whatever --limit_s1 batch you used to build the scored file
(e.g. the same 1,000 or 50,000 entities).
"""

from pathlib import Path
from collections import defaultdict
import argparse

import pandas as pd

K_VALUES = [10, 15, 20, 30, 40, 50]


def load_scored(scored_path):
    df = pd.read_csv(scored_path, sep="\t", dtype=str)
    df["score"] = df["score"].astype(int)

    by_s1 = defaultdict(list)
    for row in df.itertuples(index=False):
        by_s1[row.source1_entity_id].append(
            (row.candidate_entity_id, row.score)
        )

    # Rank once, descending by score, tie-break by candidate id for
    # determinism -- same convention the old sweep used.
    for s1_id in by_s1:
        by_s1[s1_id].sort(key=lambda item: (-item[1], item[0]))

    return by_s1


def load_ground_truth(gt_path, s1_ids):
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_ids)]

    ground_truth = {}
    for row in gt_df.itertuples(index=False):
        matched = row.matched_entity_ids
        ids = {x.strip() for x in matched.split(",") if x.strip()} if matched else set()
        ground_truth[row.source1_entity_id] = ids

    return ground_truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", required=True,
                         help="*_scored.tsv from build_candidates_v52.py")
    parser.add_argument("--ground_truth", required=True)
    args = parser.parse_args()

    print("=" * 75)
    print("V5.2 TOP-K VALIDATION (from precomputed candidates -- no rebuild)")
    print("=" * 75)

    ranked_by_s1 = load_scored(Path(args.scored))
    s1_ids = set(ranked_by_s1.keys())
    print(f"\nS1 entities in scored file: {len(s1_ids):,}")

    ground_truth = load_ground_truth(Path(args.ground_truth), s1_ids)

    total_true_matches = sum(len(v) for v in ground_truth.values())
    print(f"S1 entities with ground truth: {len(ground_truth):,}")
    print(f"Total true matches: {total_true_matches:,}")

    print("\n" + "=" * 75)
    print("TOP-K SWEEP (V5.2)")
    print("=" * 75)
    print("\nK\tAvg Candidates\tRecall\t\tTrue Found\tS1 All Found")
    print("-" * 75)

    n_s1 = len(s1_ids)

    for k in K_VALUES:
        total_kept = 0
        true_found = 0
        all_found_s1 = 0

        for s1_id in s1_ids:
            top_k = {cand for cand, _ in ranked_by_s1[s1_id][:k]}
            true_ids = ground_truth.get(s1_id, set())

            total_kept += len(top_k)
            found = true_ids & top_k
            true_found += len(found)

            if found == true_ids:
                all_found_s1 += 1

        avg_candidates = total_kept / n_s1 if n_s1 else 0
        recall = (true_found / total_true_matches) if total_true_matches else 1.0

        print(f"{k}\t{avg_candidates:.2f}\t\t{recall * 100:.4f}%\t\t"
              f"{true_found:,}/{total_true_matches:,}\t{all_found_s1:,}/{n_s1:,}")

    print("\n" + "=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    main()
