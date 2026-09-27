"""
scripts/build_features.py

Person B, step 2: point this at Person A's REAL candidates file (once it
lands) plus the source TSVs, and it writes features.tsv -- one row per
(source1_entity_id, candidate_entity_id) pair, with all similarity
features computed by src/features.py.

Zero code change needed if A's schema matches:
    candidates.tsv columns: source1_entity_id <tab> candidate_entity_ids
    (candidate_entity_ids = comma-separated list of "S2-..."/"S3-..." ids)

Usage:
    python scripts\\build_features.py ^
        --candidates output\\candidate_pairs_sample.tsv ^
        --source1 datasets\\train\\train_source1.tsv ^
        --source2 datasets\\train\\train_source2.tsv ^
        --source3 datasets\\train\\train_source3.tsv ^
        --out output\\features.tsv
"""

from pathlib import Path
import argparse
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import featurize


def load_lookup(path):
    """entity_id -> {business_name, business_address, country} dict."""
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[row.entity_id] = {
            "business_name": row.business_name,
            "business_address": row.business_address,
            "country": row.country,
        }
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True,
                         help="Person A's candidates.tsv "
                              "(source1_entity_id, candidate_entity_ids)")
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    t0 = time.time()

    print("Loading source lookups...", flush=True)
    lookup_s1 = load_lookup(Path(args.source1))
    lookup_s2 = load_lookup(Path(args.source2))
    lookup_s3 = load_lookup(Path(args.source3))
    print(f"  S1: {len(lookup_s1):,} | S2: {len(lookup_s2):,} | S3: {len(lookup_s3):,}")

    def resolve(entity_id):
        if entity_id.startswith("S1-") or entity_id in lookup_s1:
            return lookup_s1.get(entity_id)
        if entity_id.startswith("S2-"):
            return lookup_s2.get(entity_id)
        if entity_id.startswith("S3-"):
            return lookup_s3.get(entity_id)
        # fallback: check all three
        return lookup_s2.get(entity_id) or lookup_s3.get(entity_id) or lookup_s1.get(entity_id)

    print("Reading candidates file...", flush=True)
    candidates_df = pd.read_csv(args.candidates, sep="\t", dtype=str).fillna("")

    rows = []
    n_pairs = 0
    n_missing = 0

    for i, row in enumerate(candidates_df.itertuples(index=False), start=1):
        s1_id = row.source1_entity_id
        s1_record = resolve(s1_id)
        if s1_record is None:
            n_missing += 1
            continue

        cand_str = row.candidate_entity_ids
        if not cand_str:
            continue

        for cand_id in cand_str.split(","):
            cand_id = cand_id.strip()
            if not cand_id:
                continue
            cand_record = resolve(cand_id)
            if cand_record is None:
                n_missing += 1
                continue

            feats = featurize(s1_record, cand_record)
            feats["source1_entity_id"] = s1_id
            feats["candidate_entity_id"] = cand_id
            rows.append(feats)
            n_pairs += 1

        if i % 100 == 0 or i == len(candidates_df):
            print(f"  processed {i:,}/{len(candidates_df):,} S1 rows "
                  f"| {n_pairs:,} pairs featurized so far", flush=True)

    out_df = pd.DataFrame(rows)
    # put id columns first
    id_cols = ["source1_entity_id", "candidate_entity_id"]
    feature_cols = [c for c in out_df.columns if c not in id_cols]
    out_df = out_df[id_cols + feature_cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)

    print()
    print(f"Wrote {len(out_df):,} feature rows -> {out_path}")
    if n_missing:
        print(f"WARNING: {n_missing:,} entity IDs from the candidates file "
              f"were not found in any source lookup -- check schema/ids.")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()