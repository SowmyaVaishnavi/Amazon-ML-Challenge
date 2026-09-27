"""
01_build_candidates.py
Lightweight blocking stand-in: generates candidate_pairs.tsv for a split
(train or test) directly from the raw source files, using
common_features.compute_block_keys_vectorized() (vectorized pandas ops,
NOT a per-row Python loop -- this matters a lot on multi-million-row files).

NOTE: This is a placeholder for Person B's real blocking/candidate
generation. It exists so the matching model (Person C's job) has real
candidate pairs to train and score against right now. When B's real
candidate_pairs.tsv lands, point 02_train_matcher.py / 03_predict_test.py
at that file instead and skip this script entirely -- nothing downstream
needs to change.

Usage:
    python 01_build_candidates.py train            # full file
    python 01_build_candidates.py train 200000      # only first 200,000 rows of each source (for quick testing)
    python 01_build_candidates.py test
"""

import sys
import time
import pandas as pd
from collections import defaultdict
from common_features import compute_block_keys_vectorized

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "train"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
assert SPLIT in ("train", "test")

DATA_DIR = r"C:\Users\LENOVO\Desktop\AWA hack\dataset"
OUT_DIR = r"C:\Users\LENOVO\Desktop\AWA hack\outputs"

s1_path = f"{DATA_DIR}/{SPLIT}/{SPLIT}_source1.tsv"
s2_path = f"{DATA_DIR}/{SPLIT}/{SPLIT}_source2.tsv"
s3_path = f"{DATA_DIR}/{SPLIT}/{SPLIT}_source3.tsv"


def build_index(path, chunksize=200_000, limit=None):
    """Stream a source file in chunks, return dict: blocking_key -> list[entity_id].
    Uses vectorized key computation per chunk (fast) instead of per-row (slow)."""
    index = defaultdict(list)
    rows_read = 0
    for chunk in pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str,
                              chunksize=chunksize):
        if limit is not None and rows_read >= limit:
            break
        if limit is not None:
            chunk = chunk.iloc[: max(0, limit - rows_read)]

        keys = compute_block_keys_vectorized(chunk["business_name"], chunk["country"])
        # group within this chunk, then merge into the running index
        grouped = chunk.assign(_key=keys.values).groupby("_key")["entity_id"].apply(list)
        for k, ids in grouped.items():
            index[k].extend(ids)

        rows_read += len(chunk)
    return index


def main():
    t0 = time.time()
    print(f"[{SPLIT}] Building blocking index for source2...")
    idx2 = build_index(s2_path, limit=LIMIT)
    print(f"  ({time.time()-t0:.1f}s elapsed)")

    print(f"[{SPLIT}] Building blocking index for source3...")
    idx3 = build_index(s3_path, limit=LIMIT)
    print(f"  ({time.time()-t0:.1f}s elapsed)")

    print(f"[{SPLIT}] Streaming source1 and generating candidates...")
    out_rows = []
    s1_rows_read = 0
    for chunk in pd.read_csv(s1_path, sep="\t", keep_default_na=False, dtype=str,
                              chunksize=200_000):
        if LIMIT is not None and s1_rows_read >= LIMIT:
            break
        if LIMIT is not None:
            chunk = chunk.iloc[: max(0, LIMIT - s1_rows_read)]
        s1_rows_read += len(chunk)

        keys = compute_block_keys_vectorized(chunk["business_name"], chunk["country"])
        for eid, key in zip(chunk["entity_id"], keys):
            cands = idx2.get(key, []) + idx3.get(key, [])
            seen = set()
            uniq = []
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
            out_rows.append((eid, ",".join(uniq)))

    out_df = pd.DataFrame(out_rows, columns=["source1_entity_id", "candidate_entity_ids"])
    out_path = f"{OUT_DIR}/candidate_pairs_{SPLIT}.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)

    n_empty = (out_df["candidate_entity_ids"] == "").sum()
    avg_cands = out_df["candidate_entity_ids"].apply(lambda x: 0 if x == "" else len(x.split(","))).mean()
    print(f"[{SPLIT}] Wrote {len(out_df)} rows -> {out_path}")
    print(f"[{SPLIT}] Entities with zero candidates: {n_empty} ({n_empty/len(out_df):.2%})")
    print(f"[{SPLIT}] Avg candidates per entity: {avg_cands:.2f}")
    print(f"[{SPLIT}] Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()