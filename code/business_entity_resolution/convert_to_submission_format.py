"""
convert_to_submission_format.py

build_candidates_v52.py writes one row per (S1, candidate) pair -- the
right shape for Person B's feature pipeline and for K-sweep validation.

The final submission format (matching generate_candidates.py /
candidate_generation.py) is one row per S1 entity, with all of its
candidates comma-joined into a single column:

    source1_entity_id    candidate_entity_ids
    S1-123                S2-9,S2-44,S3-7

Run this only on the final frozen candidate file, right before packaging
the submission -- not on every intermediate batch.

Usage:
    python scripts\\convert_to_submission_format.py \
        --pairs candidate_pairs_test.tsv \
        --out matching_candidates_submission.tsv
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True,
                         help="pair-format file from build_candidates_v52.py "
                              "(source1_entity_id, candidate_entity_id)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.pairs, sep="\t", dtype=str)

    grouped = (
        df.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(ids)))
        .reset_index()
        .rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    )

    grouped.to_csv(args.out, sep="\t", index=False)

    print(f"Converted {df['source1_entity_id'].nunique():,} S1 entities "
          f"({len(df):,} pairs) -> {args.out}")


if __name__ == "__main__":
    main()
