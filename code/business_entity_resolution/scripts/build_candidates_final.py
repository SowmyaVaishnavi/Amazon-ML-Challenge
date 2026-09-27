"""
build_candidates_final.py

Person A -- final candidate generation.

Combines:
  - build_candidates_v52.py's speed features: cached inverted index
    (rebuild once, reuse forever), posting-list skew diagnostics, a
    --max_postings cap so no single toxic key can blow up a lookup, and
    real per-entity progress logging.
  - candidate_generation.py / blocking.py's AGREED OUTPUT SCHEMA: one row
    per S1 entity, with all its candidates comma-joined into a single
    "candidate_entity_ids" column. This is the schema build_features.py
    (Person B's script) and everything downstream already expects and has
    been validated against -- so this is now the canonical output format,
    not the one-row-per-pair format the older v52 script used.
  - NEW: incremental / streaming writes. Earlier versions built the whole
    output in memory (a list of rows) and only wrote the CSV once, at the
    very end of main(). That meant the output file stayed empty/blank
    until the entire run finished, so there was no way to tell "slow but
    working" from "hung" by watching the file. This version opens the
    output file up front, writes the header immediately, and writes +
    flushes each S1 row to disk the moment it's computed. Open the file
    in VS Code (or `type`/`Get-Content -Wait` it in a second terminal)
    while this runs and you'll see rows appear in real time.

Usage:
    python scripts\\build_candidates_final.py \
        --source1 datasets\\train\\train_source1.tsv \
        --source2 datasets\\train\\train_source2.tsv \
        --source3 datasets\\train\\train_source3.tsv \
        --out output\\candidate_pairs_final.tsv \
        --top_k 30

Omit --limit_s1 to run the full S1 file; keep it for a quick trial run.

Output schema (one row per S1 entity):
    source1_entity_id <tab> candidate_entity_ids
    (candidate_entity_ids = comma-joined, sorted, e.g. "S2-1,S2-9,S3-4")
"""

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import pickle
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import normalize_record  # SCHEMA


# ============================================================
# V5.2 BLOCKING KEYS BY TYPE (unchanged)
# ============================================================

def make_blocking_keys_by_type(record):
    name = record["business_name"]
    address = record["business_address"]
    country = record["country"]

    blocks = defaultdict(set)
    name_tokens = name.split()

    if country and name_tokens:
        first_word = name_tokens[0]
        for n in (3, 4):
            if len(first_word) >= n:
                blocks[f"nameprefix{n}"].add(
                    f"country|{country}|nameprefix{n}|{first_word[:n]}"
                )

    if country and len(name_tokens) >= 2:
        w1, w2 = name_tokens[0], name_tokens[1]
        if len(w1) >= 3 and len(w2) >= 3:
            blocks["name2prefix"].add(
                f"country|{country}|name2prefix|{w1[:3]}|{w2[:3]}"
            )

    if country:
        for word in name_tokens:
            if len(word) >= 5:
                for n in (5, 6):
                    if len(word) >= n:
                        blocks[f"nameword{n}"].add(
                            f"country|{country}|nameword{n}|{word[:n]}"
                        )

    if country:
        for word in name_tokens:
            if 1 <= len(word) <= 3:
                blocks["nameshort"].add(
                    f"country|{country}|nameshort|{word}"
                )

    if country and address:
        blocks["fulladdress"].add(
            f"country|{country}|address|{address}"
        )

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
# INDEX CACHE HELPERS (unchanged from v52)
# ============================================================

def _cache_signature(paths):
    parts = []
    for p in paths:
        stat = p.stat()
        parts.append(f"{p.name}:{stat.st_size}:{int(stat.st_mtime)}")
    return "|".join(parts)


def _cache_path_for(paths, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = _cache_signature(paths)
    return cache_dir / "inverted_index_cache.pkl", sig


def load_or_build_inverted_index(paths, cache_dir, force_rebuild=False, chunksize=200_000):
    cache_file, sig = _cache_path_for(paths, cache_dir)

    if not force_rebuild and cache_file.exists():
        print(f"Found cached index at {cache_file} -- checking freshness...", flush=True)
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if cached.get("signature") == sig:
                print("Cache is fresh. Loading (seconds, not hours).", flush=True)
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
# SKEW DIAGNOSTICS + TOXIC-KEY CAP (unchanged from v52)
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
        print(f"Dropping {len(dropped):,} keys with > {max_postings:,} postings:")
        for size, key in dropped[:20]:
            print(f"  DROPPED  {size:>10,}  {key}")
        if len(dropped) > 20:
            print(f"  ... and {len(dropped) - 20:,} more")

    return filtered, dropped


# ============================================================
# NEW: STREAMING CANDIDATE GENERATION
#
# Writes one row per S1 entity to the output file THE MOMENT it's
# computed, then flushes -- no buffering the whole run in memory,
# no waiting until every S1 entity is done before anything hits disk.
# ============================================================

def generate_candidates_streaming(
    s1_path, inverted_index, top_k, out_path, scored_path,
    limit_s1=None, progress_every=25,
):
    read_kwargs = {"sep": "\t", "dtype": str}
    if limit_s1:
        read_kwargs["nrows"] = limit_s1
    s1_df = pd.read_csv(s1_path, **read_kwargs).fillna("")
    total = len(s1_df)
    print(f"Loaded {len(s1_df):,} S1 entities "
          f"({'all' if not limit_s1 else limit_s1} requested).")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_with_candidates = 0
    start = time.time()

    with open(out_path, "w", newline="", encoding="utf-8") as out_f, \
         open(scored_path, "w", newline="", encoding="utf-8") as scored_f:

        out_writer = csv.writer(out_f, delimiter="\t")
        scored_writer = csv.writer(scored_f, delimiter="\t")

        out_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        scored_writer.writerow(["source1_entity_id", "candidate_entity_id", "score"])
        out_f.flush()
        scored_f.flush()

        for i, row in enumerate(s1_df.itertuples(index=False), start=1):
            s1_id = row.entity_id
            record = normalize_record(  # SCHEMA
                row.entity_id, row.business_name, row.business_address, row.country
            )
            blocks = make_blocking_keys_by_type(record)

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

            candidate_ids = sorted(cand_id for cand_id, _ in scored)
            if candidate_ids:
                n_with_candidates += 1

            # ---- write + flush THIS row immediately ----
            out_writer.writerow([s1_id, ",".join(candidate_ids)])
            out_f.flush()

            for cand_id, types_matched in scored:
                scored_writer.writerow([s1_id, cand_id, len(types_matched)])
            scored_f.flush()

            if i % progress_every == 0 or i == total:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else float("inf")
                print(
                    f"  S1 progress: {i:,}/{total:,} "
                    f"| {rate:,.1f} entities/sec "
                    f"| elapsed {elapsed/60:.1f} min "
                    f"| est. remaining {remaining/60:.1f} min "
                    f"| rows written so far: {i:,}",
                    flush=True,
                )

    return total, n_with_candidates


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
    parser.add_argument("--cache_dir", default=None,
                         help="Where to store/load the pickled index. "
                              "Defaults to <project_root>/.index_cache")
    parser.add_argument("--rebuild_index", action="store_true",
                         help="Force rebuilding the index even if a fresh cache exists.")
    parser.add_argument("--max_postings", type=int, default=20_000,
                         help="Drop block keys with more than this many matches. "
                              "0 disables capping.")
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
    print("FINAL CANDIDATE GENERATION (cached index, capped, streaming writes)")
    print("=" * 80)

    t0 = time.time()
    inverted_index = load_or_build_inverted_index(
        [s2_path, s3_path],
        cache_dir=cache_dir,
        force_rebuild=args.rebuild_index,
    )
    t1 = time.time()
    print(f"Index ready in {t1 - t0:.1f}s")

    print_posting_list_stats(inverted_index, top_n=20)
    inverted_index, dropped = cap_toxic_keys(inverted_index, args.max_postings)

    print()
    print(f"Writing rows incrementally to: {out_path}")
    print(f"(open it now in VS Code / another terminal -- rows will "
          f"appear as they're generated, not only at the end)")
    print()

    t2 = time.time()
    total, n_with_candidates = generate_candidates_streaming(
        s1_path, inverted_index, args.top_k, out_path, scored_path,
        limit_s1=args.limit_s1, progress_every=args.progress_every,
    )
    t3 = time.time()

    print()
    print(f"Candidate generation took {t3 - t2:.1f}s for {total:,} S1 entities.")
    print(f"Wrote candidates  -> {out_path}")
    print(f"Wrote scored pairs -> {scored_path}")
    print(f"S1 entities with >=1 candidate: {n_with_candidates:,} / {total:,}")
    if dropped:
        print(f"NOTE: {len(dropped):,} toxic block keys were dropped "
              f"(threshold: {args.max_postings:,} postings).")
    print(f"Total wall time: {time.time() - t0:.1f}s")
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()