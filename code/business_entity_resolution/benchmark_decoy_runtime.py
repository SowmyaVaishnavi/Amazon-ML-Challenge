from pathlib import Path
from collections import defaultdict
import pandas as pd
import time
import sys


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"


# ============================================================
# IMPORTANT:
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT ACTUAL V5.1 LOGIC
# ============================================================

from src.normalization import normalize_record
from scripts.generate_candidates import make_blocking_keys


# ============================================================
# BENCHMARK SETTINGS
# ============================================================

# ONLY 30 S1 RECORDS
SAMPLE_SIZE = 30

# Large-file reading size
CHUNK_SIZE = 200_000


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("V5.1 FULL-POPULATION DECOY RUNTIME BENCHMARK")
print("=" * 80)

print()
print(f"Source 1 tested: FIRST {SAMPLE_SIZE} records")
print("Source 2: FULL TRAIN")
print("Source 3: FULL TRAIN")
print("Normalization: ACTUAL src/normalization.py")
print("Blocking: ACTUAL V5.1 make_blocking_keys()")
print()
print("TEST DATA WILL NOT BE TOUCHED.")
print()


# ============================================================
# LOAD FIRST 30 SOURCE 1 RECORDS
# ============================================================

print("=" * 80)
print("STEP 1 — LOADING SOURCE 1 SAMPLE")
print("=" * 80)

start_time = time.perf_counter()

s1_df = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    nrows=SAMPLE_SIZE
).fillna("")

load_s1_time = time.perf_counter() - start_time

print(
    f"S1 records loaded: {len(s1_df):,}"
)

print(
    f"S1 loading time: {load_s1_time:.2f} seconds"
)


# ============================================================
# BUILD V5.1 BLOCKING KEYS FOR S1
# ============================================================

print()
print("=" * 80)
print("STEP 2 — BUILDING V5.1 KEYS FOR S1")
print("=" * 80)

start_time = time.perf_counter()

query_keys_by_s1 = {}

all_query_keys = set()

for _, row in s1_df.iterrows():

    record = normalize_record(
        row["entity_id"],
        row["business_name"],
        row["business_address"],
        row["country"]
    )

    s1_id = record["entity_id"]

    keys = make_blocking_keys(record)

    query_keys_by_s1[s1_id] = keys

    all_query_keys.update(keys)

query_key_time = time.perf_counter() - start_time

print(
    f"S1 records processed: "
    f"{len(query_keys_by_s1):,}"
)

print(
    f"Unique query blocking keys: "
    f"{len(all_query_keys):,}"
)

print(
    f"Key-generation time: "
    f"{query_key_time:.2f} seconds"
)


# ============================================================
# BUILD REVERSE KEY LOOKUP
#
# blocking key
#       ↓
# S1 IDs that use that key
# ============================================================

print()
print("=" * 80)
print("STEP 3 — BUILDING QUERY KEY LOOKUP")
print("=" * 80)

start_time = time.perf_counter()

key_to_s1 = defaultdict(set)

for s1_id, keys in query_keys_by_s1.items():

    for key in keys:

        key_to_s1[key].add(s1_id)

lookup_time = time.perf_counter() - start_time

print(
    f"Query keys indexed: "
    f"{len(key_to_s1):,}"
)

print(
    f"Lookup construction time: "
    f"{lookup_time:.2f} seconds"
)


# ============================================================
# CANDIDATE SCORES
#
# candidate_scores[S1][candidate] = shared blocking-key count
# ============================================================

candidate_scores = {
    s1_id: defaultdict(int)
    for s1_id in query_keys_by_s1
}


# ============================================================
# FUNCTION TO SCAN S2/S3
# ============================================================

def scan_reference_source(source_name, path):

    source_start = time.perf_counter()

    source_rows = 0
    source_relevant_rows = 0

    chunk_number = 0

    print()
    print("=" * 80)
    print(f"SCANNING FULL {source_name}")
    print("=" * 80)

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        chunk = chunk.fillna("")

        chunk_number += 1

        for _, row in chunk.iterrows():

            source_rows += 1

            # ------------------------------------------------
            # ACTUAL V5.1 NORMALIZATION
            # ------------------------------------------------

            record = normalize_record(
                row["entity_id"],
                row["business_name"],
                row["business_address"],
                row["country"]
            )

            candidate_id = record["entity_id"]

            # ------------------------------------------------
            # ACTUAL V5.1 BLOCKING KEYS
            # ------------------------------------------------

            reference_keys = make_blocking_keys(record)

            # ------------------------------------------------
            # Only keys that are relevant to our 30 S1s
            # ------------------------------------------------

            relevant_keys = (
                reference_keys.intersection(
                    all_query_keys
                )
            )

            if not relevant_keys:
                continue

            source_relevant_rows += 1

            # ------------------------------------------------
            # Determine which S1 queries share these keys
            # ------------------------------------------------

            affected_s1_ids = set()

            for key in relevant_keys:

                affected_s1_ids.update(
                    key_to_s1.get(
                        key,
                        set()
                    )
                )

            # ------------------------------------------------
            # Calculate blocking-hit score
            # ------------------------------------------------

            for s1_id in affected_s1_ids:

                s1_keys = query_keys_by_s1[s1_id]

                shared_count = len(
                    s1_keys.intersection(
                        relevant_keys
                    )
                )

                if shared_count > 0:

                    candidate_scores[
                        s1_id
                    ][candidate_id] += shared_count

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - source_start
        )

        print(
            f"{source_name} chunk "
            f"{chunk_number} processed | "
            f"rows scanned: "
            f"{source_rows:,} | "
            f"relevant rows: "
            f"{source_relevant_rows:,} | "
            f"elapsed: "
            f"{elapsed / 60:.2f} min"
        )

    source_time = (
        time.perf_counter()
        - source_start
    )

    print()
    print(
        f"{source_name} COMPLETE"
    )

    print(
        f"Rows scanned: "
        f"{source_rows:,}"
    )

    print(
        f"Relevant reference rows: "
        f"{source_relevant_rows:,}"
    )

    print(
        f"Time: "
        f"{source_time:.2f} seconds "
        f"({source_time / 60:.2f} minutes)"
    )

    return source_time


# ============================================================
# START FULL-POPULATION BENCHMARK
# ============================================================

benchmark_start = time.perf_counter()


# ============================================================
# SCAN FULL S2
# ============================================================

s2_time = scan_reference_source(
    "S2",
    S2_PATH
)


# ============================================================
# SCAN FULL S3
# ============================================================

s3_time = scan_reference_source(
    "S3",
    S3_PATH
)


# ============================================================
# TOTAL SCAN TIME
# ============================================================

total_scan_time = (
    time.perf_counter()
    - benchmark_start
)


# ============================================================
# CANDIDATE STATISTICS
# ============================================================

print()
print("=" * 80)
print("CANDIDATE STATISTICS")
print("=" * 80)

total_candidates = 0
max_candidates = 0
s1_with_candidates = 0

for s1_id, scores in candidate_scores.items():

    count = len(scores)

    total_candidates += count

    max_candidates = max(
        max_candidates,
        count
    )

    if count > 0:

        s1_with_candidates += 1


average_candidates = (
    total_candidates
    / SAMPLE_SIZE
)


print(
    f"Average candidates per S1: "
    f"{average_candidates:.2f}"
)

print(
    f"Maximum candidates for one S1: "
    f"{max_candidates:,}"
)

print(
    f"S1s with candidates: "
    f"{s1_with_candidates:,}/"
    f"{SAMPLE_SIZE:,}"
)


# ============================================================
# TOP-K DIAGNOSTIC
# ============================================================

print()
print("=" * 80)
print("TOP-K DIAGNOSTIC — FIRST 5 S1s")
print("=" * 80)

for s1_id in list(
    candidate_scores.keys()
)[:5]:

    ranked = sorted(
        candidate_scores[s1_id].items(),
        key=lambda x: (-x[1], x[0])
    )

    print()
    print(
        f"S1: {s1_id}"
    )

    print(
        f"Total candidates: "
        f"{len(ranked):,}"
    )

    print(
        "Top 10 blocking-hit scores:"
    )

    for candidate_id, score in ranked[:10]:

        print(
            f"  {candidate_id} "
            f"-> score {score}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

total_runtime = (
    time.perf_counter()
    - benchmark_start
)

print()
print("=" * 80)
print("BENCHMARK COMPLETE")
print("=" * 80)

print()
print(
    f"S1 tested: "
    f"{SAMPLE_SIZE:,}"
)

print(
    "S2 population: FULL TRAIN"
)

print(
    "S3 population: FULL TRAIN"
)

print(
    f"S2 scan time: "
    f"{s2_time:.2f} sec "
    f"({s2_time / 60:.2f} min)"
)

print(
    f"S3 scan time: "
    f"{s3_time:.2f} sec "
    f"({s3_time / 60:.2f} min)"
)

print(
    f"Total S2 + S3 scan time: "
    f"{total_scan_time:.2f} sec "
    f"({total_scan_time / 60:.2f} min)"
)

print(
    f"Total benchmark runtime: "
    f"{total_runtime:.2f} sec "
    f"({total_runtime / 60:.2f} min)"
)

print()
print("IMPORTANT:")
print("- This was ONLY a 30-S1 speed trial.")
print("- Full S2 and full S3 were scanned.")
print("- TEST was NOT touched.")
print("- No final candidate_pairs.tsv was created.")
print("- K was NOT selected.")
print("- K was NOT frozen.")
print("- Do NOT run the 1,000-S1 validation until this runtime is checked.")
print()

print("=" * 80)
print("DONE")
print("=" * 80)