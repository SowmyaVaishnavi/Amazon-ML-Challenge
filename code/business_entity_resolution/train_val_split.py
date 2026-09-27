"""
Build a local train/val split out of the training data, so A/B/C can
develop and score their own work before touching the real test set.

Design goal: the val split should look EXACTLY like the real test set from
every downstream script's point of view.
  - dataset/train_split/train_source1.tsv        <- Source 1 records to train on
  - dataset/train_split/train_source2.tsv        <- full Source 2 pool (unchanged)
  - dataset/train_split/train_source3.tsv        <- full Source 3 pool (unchanged)
  - dataset/train_split/train_ground_truth.tsv   <- ground truth for the training rows

  - dataset/val_split/val_source1.tsv            <- held-out Source 1 records,
                                                     NO ground truth column
                                                     (same schema as test_source1.tsv)
  - dataset/val_split/val_source2.tsv            <- full Source 2 pool (unchanged)
  - dataset/val_split/val_source3.tsv            <- full Source 3 pool (unchanged)
  - dataset/val_split/val_ground_truth.tsv       <- HIDDEN answer key, only ever
                                                     touched by scorer.py, never by
                                                     the blocking/feature/model code

Source 2 and Source 3 are NOT split. In the real challenge, a Source 1 test
entity can match anything in the full test_source2/test_source3 pool - if we
shrunk Source 2/3 for validation too, blocking's reduction ratio and recall
numbers on the val split would not transfer to the real test set. Splitting
only Source 1 keeps the val split honest for A's blocking metrics as well as
B/C's downstream numbers.

The split is done by source1_entity_id (not by row position) and is
reproducible via --seed. Use --stratify-country to keep the US/India mix
in both splits proportional (recommended, since the real test set adds a
third country - France - not seen in training at all; that domain-shift
gap can't be fixed by stratifying train, but it can be *measured* by
holding out a clean, representative val set).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.io import read_source, read_ground_truth  # noqa: E402


def split(
    source1_path: str,
    source2_path: str,
    source3_path: str,
    ground_truth_path: str,
    out_dir: str,
    val_frac: float = 0.15,
    seed: int = 42,
    stratify_country: bool = True,
) -> None:
    s1 = read_source(source1_path)
    gt = read_ground_truth(ground_truth_path)

    if set(s1["entity_id"]) != set(gt["source1_entity_id"]):
        missing_gt = set(s1["entity_id"]) - set(gt["source1_entity_id"])
        missing_s1 = set(gt["source1_entity_id"]) - set(s1["entity_id"])
        raise ValueError(
            f"source1/ground_truth entity_id mismatch: "
            f"{len(missing_gt)} S1 entities have no ground-truth row, "
            f"{len(missing_s1)} ground-truth rows reference an unknown S1 entity."
        )

    rng = np.random.default_rng(seed)

    if stratify_country:
        val_ids = []
        for _, grp in s1.groupby("country"):
            ids = grp["entity_id"].to_numpy()
            rng.shuffle(ids)
            n_val = max(1, int(round(len(ids) * val_frac)))
            val_ids.append(ids[:n_val])
        val_ids = set(np.concatenate(val_ids))
    else:
        ids = s1["entity_id"].to_numpy().copy()
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_frac)))
        val_ids = set(ids[:n_val])

    is_val = s1["entity_id"].isin(val_ids)
    train_s1, val_s1 = s1[~is_val].reset_index(drop=True), s1[is_val].reset_index(drop=True)

    gt_indexed = gt.set_index("source1_entity_id")
    train_gt = gt_indexed.loc[train_s1["entity_id"]].reset_index()
    val_gt = gt_indexed.loc[val_s1["entity_id"]].reset_index()

    train_dir = os.path.join(out_dir, "train_split")
    val_dir = os.path.join(out_dir, "val_split")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    train_s1.to_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", index=False)
    train_gt.to_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t", index=False)
    val_s1.to_csv(os.path.join(val_dir, "val_source1.tsv"), sep="\t", index=False)
    val_gt.to_csv(os.path.join(val_dir, "val_ground_truth.tsv"), sep="\t", index=False)

    # Source 2 / 3 are linked (not physically copied) into both split dirs:
    # only Source 1 is split, so the candidate pool must stay the full, real
    # size for both A's blocking metrics and B/C's downstream numbers.
    # Symlinks keep each split dir self-contained from pandas' point of view
    # (read_csv follows them fine) without duplicating ~1GB of data on disk.
    for split_dir, prefix in ((train_dir, "train"), (val_dir, "val")):
        for src_path, name in ((source2_path, "source2"), (source3_path, "source3")):
            link_path = os.path.join(split_dir, f"{prefix}_{name}.tsv")
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(os.path.abspath(src_path), link_path)

    n_train_matches = train_gt["matched_entity_ids"].apply(lambda c: c != "").sum()
    n_val_matches = val_gt["matched_entity_ids"].apply(lambda c: c != "").sum()
    print(f"[split] train: {len(train_s1):,} S1 entities "
          f"({n_train_matches:,} with >=1 match, {len(train_s1) - n_train_matches:,} singletons)")
    print(f"[split] val:   {len(val_s1):,} S1 entities "
          f"({n_val_matches:,} with >=1 match, {len(val_s1) - n_val_matches:,} singletons)")
    print(f"[split] country mix - train: {train_s1['country'].value_counts().to_dict()}")
    print(f"[split] country mix - val:   {val_s1['country'].value_counts().to_dict()}")
    print(f"[split] wrote to {train_dir}/ and {val_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source1", required=True)
    ap.add_argument("--source2", required=True)
    ap.add_argument("--source3", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-stratify-country", action="store_true")
    args = ap.parse_args()

    split(
        source1_path=args.source1,
        source2_path=args.source2,
        source3_path=args.source3,
        ground_truth_path=args.ground_truth,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        seed=args.seed,
        stratify_country=not args.no_stratify_country,
    )


if __name__ == "__main__":
    main()
