"""
build_candidates_v52.py

Fast, scale-ready candidate generation.

CHANGES IN THIS VERSION (marked with "# NEW:" comments below):
  1. Index caching: the S2+S3 inverted index is pickled to disk after the
     first build. Re-running with the same source2/source3 files (same
     path + size + mtime) loads the cache in seconds instead of rebuilding
     for ~2 hours. Use --rebuild_index to force a fresh build.
  2. Posting-list skew diagnostics: right after the index is ready, prints
     the top 20 largest posting lists so you can SEE which blocking keys
     are toxic (e.g. short generic tokens, empty addresses) instead of
     discovering it as a multi-hour hang.
  3. --max_postings cap: any block key whose posting list is longer than
     this threshold is dropped before candidate generation, so one S1
     entity can never trigger an O(huge) scan. Defaults to 20,000.
  4. Per-entity progress logging in generate_candidates (every
     --progress_every entities, default 25) with elapsed time and rate,
     so a slow-but-working run is distinguishable from a hung one.

Usage (same as before, plus new optional flags):
    python scripts\\build_candidates_v52.py \
        --source1 datasets\\train\\train_source1.tsv \
        --source2 datasets\\train\\train_source2.tsv \
        --source3 datasets\\train\\train_source3.tsv \
        --out output\\candidate_pairs_trial.tsv \
        --limit_s1 1000 \
        --top_k 30 \
        --max_postings 20000 \
        --progress_every 25

Outputs:
  - <out>            : source1_entity_id <tab> candidate_entity_id
  - <out>_scored.tsv  : source1_entity_id <tab> candidate_entity_id <tab> score
"""

from pathlib import Path
from collections import defaultdict
import argparse
import pickle
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import normalize_record  # SCHEMA


# ============================================================
# V5.2 BLOCKING KEYS BY TYPE (unchanged from the validated diagnostic)
# ============================================================

def make_blocking_keys_by_type(record):
    name = record["business_name"]
    address = record["business_address"]
    country = record["country"]

    blocks = defaultdict(set)
    name_tokens = name.split()

    # NAME PREFIX 3 / 4
    if country and name_tokens:
        first_word = name_tokens[0]
        for n in (3, 4):
            if len(first_word) >= n:
                blocks[f"nameprefix{n}"].add(
                    f"country|{country}|nameprefix{n}|{first_word[:n]}"
                )

    # TWO WORD PREFIX
    if country and len(name_tokens) >= 2:
        w1, w2 = name_tokens[0], name_tokens[1]
        if len(w1) >= 3 and len(w2) >= 3:
            blocks["name2prefix"].add(
                f"country|{country}|name2prefix|{w1[:3]}|{w2[:3]}"
            )

    # NAME WORD PREFIX 5 / 6
    if country:
        for word in name_tokens:
            if len(word) >= 5:
                for n in (5, 6):
                    if len(word) >= n:
                        blocks[f"nameword{n}"].add(
                            f"country|{country}|nameword{n}|{word[:n]}"
                        )

    # SHORT NAME
    if country:
        for word in name_tokens:
            if 1 <= len(word) <= 3:
                blocks["nameshort"].add(
                    f"country|{country}|nameshort|{word}"
                )

    # FULL ADDRESS
    if country and address:
        blocks["fulladdress"].add(
            f"country|{country}|address|{address}"
        )

    # ADDRESS WORD 6 / 7
    if country:
        for word in address.split():
            if len(word) >= 6:
                blocks["addrword6"].add(
                    f"country|{country}|addrword6|{word[:6]}"
                )
            if len(word) >= 7:
                blocks["addrword7"].add(
                    f"country|{country}|addrword7|{word[:7]}"
                )

    return blocks


# ============================================================
# NEW: INDEX CACHE HELPERS
# ============================================================

def _cache_signature(paths):
    """A signature that changes if any source file changes (size+mtime)."""
    parts = []
    for p in paths:
        stat = p.stat()
        parts.append(f"{p.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return "|".join(parts)


def _cache_path_for(paths, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = _cache_signature(paths)
    # simple stable filename; signature is stored INSIDE the file too,
    # so even a hash collision would be caught on load.
    return cache_dir / "inverted_index_cache.pkl", sig


def load_or_build_inverted_index(paths, cache_dir, force_rebuild=False, chunksize=200_000):
    cache_file, sig = _cache_path_for(paths, cache_dir)

    if not force_rebuild and cache_file.exists():
        print(f"Found cached index at {cache_file} -- checking freshness...", flush=True)
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if cached.get("signature") == sig:
                print("Cache is fresh. Loading (this should take seconds, not hours).", flush=True)
                return cached["index"]
            else:
                print("Cache is stale (source files changed). Rebuilding.", flush=True)
        except Exception as e:
            print(f"Cache load failed ({e}). Rebuilding.", flush=True)

    inverted_index = build_inverted_index(paths, chunksize=chunksize)

    print(f"Saving index cache to {cache_file} ...", flush=True)
    with open(cache_file, "wb") as f:
        pickle.dump({"signature": sig, "index": dict(inverted_index)}, f, protocol=pickle.HIGHEST_PROTOCOL)

    return inverted_index


def build_inverted_index(paths, chunksize=200_000):
    inverted_index = defaultdict(set)
    total_rows = 0
    start = time.time()

    for path in paths:
        print(f"Indexing {path.name} ...", flush=True)
        file_rows = 0
        for chunk_num, chunk in enumerate(
            pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize), start=1
        ):
            chunk = chunk.fillna("")
            for row in chunk.itertuples(index=False):
                record = normalize_record(  # SCHEMA
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                    row.country,
                )
                blocks = make_blocking_keys_by_type(record)
                for key_set in blocks.values():
                    for key in key_set:
                        inverted_index[key].add(row.entity_id)
                total_rows += 1
                file_rows += 1

            elapsed = time.time() - start
            rate = total_rows / elapsed if elapsed > 0 else 0
            print(
                f"  {path.name}: chunk {chunk_num} done | "
                f"{file_rows:,} rows this file | "
                f"{total_rows:,} rows total | "
                f"{len(inverted_index):,} keys so far | "
                f"{rate:,.0f} rows/sec | "
                f"{elapsed / 60:.1f} min elapsed",
                flush=True,
            )

    print(f"Indexed {total_rows:,} reference records "
          f"into {len(inverted_index):,} distinct block keys.")
    return inverted_index


# ============================================================
# NEW: SKEW DIAGNOSTICS + TOXIC-KEY CAP
# ============================================================

def print_posting_list_stats(inverted_index, top_n=20):
    sizes = [(len(v), k) for k, v in inverted_index.items()]
    sizes.sort(reverse=True)

    print()
    print("=" * 70)
    print("BLOCKING KEY SKEW DIAGNOSTICS")
    print("=" * 70)
    print(f"Total keys: {len(sizes):,}")
    if sizes:
        total_postings = sum(s for s, _ in sizes)
        print(f"Total (key -> id) postings: {total_postings:,}")
        print(f"Largest posting list: {sizes[0][0]:,}  (key: {sizes[0][1]})")
        print()
        print(f"Top {top_n} largest posting lists:")
        for size, key in sizes[:top_n]:
            print(f"  {size:>10,}  {key}")
    print("=" * 70)
    return sizes


def cap_toxic_keys(inverted_index, max_postings):
    """
    Drop any key whose posting list exceeds max_postings. Returns the
    filtered index plus a summary of what was dropped, so a handful of
    generic keys (e.g. nameshort:'co', empty fulladdress) can't blow up
    the per-entity candidate lookup.
    """
    if not max_postings:
        return inverted_index, []

    dropped = []
    filtered = {}
    for key, ids in inverted_index.items():
        if len(ids) > max_postings:
            dropped.append((len(ids), key))
        else:
            filtered[key] = ids

    if dropped:
        dropped.sort(reverse=True)
        print()
        print(f"Dropping {len(dropped):,} keys with > {max_postings:,} postings "
              f"(these were the likely cause of any hang):")
        for size, key in dropped[:20]:
            print(f"  DROPPED  {size:>10,}  {key}")
        if len(dropped) > 20:
            print(f"  ... and {len(dropped) - 20:,} more")

    return filtered, dropped


# ============================================================
# CANDIDATE GENERATION FOR ONE S1 BATCH
# ============================================================

def generate_candidates(s1_df, inverted_index, top_k, progress_every=25):
    rows_pairs = []
    rows_scored = []

    total = len(s1_df)
    start = time.time()

    for i, row in enumerate(s1_df.itertuples(index=False), start=1):
        s1_id = row.entity_id
        record = normalize_record(  # SCHEMA
            row.entity_id, row.business_name, row.business_address, row.country
        )
        blocks = make_blocking_keys_by_type(record)

        # candidate_id -> set of block_types matched
        candidate_types = defaultdict(set)

        for block_type, key_set in blocks.items():
            for key in key_set:
                for cand_id in inverted_index.get(key, ()):
                    candidate_types[cand_id].add(block_type)

        scored = sorted(
            candidate_types.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )[:top_k]

        for cand_id, types_matched in scored:
            rows_pairs.append((s1_id, cand_id))
            rows_scored.append((s1_id, cand_id, len(types_matched)))

        # NEW: real per-entity progress, so a slow run doesn't look hung
        if i % progress_every == 0 or i == total:
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else float("inf")
            print(
                f"  S1 progress: {i:,}/{total:,} "
                f"| {rate:,.1f} entities/sec "
                f"| elapsed {elapsed/60:.1f} min "
                f"| est. remaining {remaining/60:.1f} min",
                flush=True,
            )

    return rows_pairs, rows_scored


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source1", required=True)
    parser.add_argument("--source2", required=True)
    parser.add_argument("--source3", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit_s1", type=int, default=None,
                         help="Number of S1 rows to process. Omit to run all.")
    parser.add_argument("--top_k", type=int, default=30,
                         help="Max candidates kept per S1 entity.")
    # NEW flags
    parser.add_argument("--cache_dir", default=None,
                         help="Where to store/load the pickled index. "
                              "Defaults to <project_root>/.index_cache")
    parser.add_argument("--rebuild_index", action="store_true",
                         help="Force rebuilding the index even if a fresh cache exists.")
    parser.add_argument("--max_postings", type=int, default=20_000,
                         help="Drop block keys with more than this many matches "
                              "before candidate generation. 0 disables capping.")
    parser.add_argument("--progress_every", type=int, default=25,
                         help="Print S1 progress every N entities.")
    args = parser.parse_args()

    s1_path = Path(args.source1)
    s2_path = Path(args.source2)
    s3_path = Path(args.source3)
    out_path = Path(args.out)
    scored_path = out_path.with_name(out_path.stem + "_scored.tsv")

    cache_dir = Path(args.cache_dir) if args.cache_dir else PROJECT_ROOT / ".index_cache"

    print("=" * 80)
    print("V5.2 CANDIDATE GENERATION (inverted-index, real scale)")
    print("=" * 80)

    t0 = time.time()
    inverted_index = load_or_build_inverted_index(
        [s2_path, s3_path],
        cache_dir=cache_dir,
        force_rebuild=args.rebuild_index,
    )
    t1 = time.time()
    print(f"Index ready in {t1 - t0:.1f}s")

    # NEW: show the skew, then cap it
    print_posting_list_stats(inverted_index, top_n=20)
    inverted_index, dropped = cap_toxic_keys(inverted_index, args.max_postings)

    read_kwargs = {"sep": "\t", "dtype": str}
    if args.limit_s1:
        read_kwargs["nrows"] = args.limit_s1
    s1_df = pd.read_csv(s1_path, **read_kwargs).fillna("")
    print(f"Loaded {len(s1_df):,} S1 entities "
          f"({'all' if not args.limit_s1 else args.limit_s1} requested).")

    t2 = time.time()
    pairs, scored = generate_candidates(
        s1_df, inverted_index, args.top_k, progress_every=args.progress_every
    )
    t3 = time.time()
    print(f"Candidate generation took {t3 - t2:.1f}s "
          f"for {len(s1_df):,} S1 entities.")

    pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"]) \
        .to_csv(out_path, sep="\t", index=False)

    pd.DataFrame(scored, columns=["source1_entity_id", "candidate_entity_id", "score"]) \
        .to_csv(scored_path, sep="\t", index=False)

    n_with_candidates = len(set(p[0] for p in pairs))
    print()
    print(f"Wrote {len(pairs):,} candidate pairs -> {out_path}")
    print(f"Wrote scored pairs         -> {scored_path}")
    print(f"S1 entities with >=1 candidate: {n_with_candidates:,} / {len(s1_df):,}")
    if dropped:
        print(f"NOTE: {len(dropped):,} toxic block keys were dropped "
              f"(threshold: {args.max_postings:,} postings). See list above.")
    print(f"Total wall time: {time.time() - t0:.1f}s")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()