import os
import sys
import pandas as pd
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import normalize_record
from src.blocking import build_block_index, make_blocking_keys

SAMPLE_SIZE = 1000
CHUNK_SIZE = 100000

TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "train")

SOURCE1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
SOURCE2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
SOURCE3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
GROUND_TRUTH_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")


# ---------------------------------------------------------
# 1. Load Source 1 sample
# ---------------------------------------------------------

print("Loading Source 1 sample...")

source1_df = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype=str,
    nrows=SAMPLE_SIZE
)

source1_ids = set(source1_df["entity_id"].astype(str))

print(f"Source 1 sample: {len(source1_ids):,}")


# ---------------------------------------------------------
# 2. Load ground truth for those Source 1 records
# ---------------------------------------------------------

print("\nLoading ground truth...")

ground_truth_df = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str
)

ground_truth_df["source1_entity_id"] = (
    ground_truth_df["source1_entity_id"].astype(str)
)

ground_truth_df = ground_truth_df[
    ground_truth_df["source1_entity_id"].isin(source1_ids)
].copy()

ground_truth = {}

required_reference_ids = set()

for _, row in ground_truth_df.iterrows():

    source1_id = str(row["source1_entity_id"])
    matched = row["matched_entity_ids"]

    if pd.isna(matched) or str(matched).strip() == "":
        matches = set()
    else:
        matches = {
            x.strip()
            for x in str(matched).split(",")
            if x.strip()
        }

    ground_truth[source1_id] = matches
    required_reference_ids.update(matches)

print(f"Ground-truth S1 records: {len(ground_truth):,}")
print(f"Required reference IDs: {len(required_reference_ids):,}")


# ---------------------------------------------------------
# 3. Find those exact reference records in S2 and S3
# ---------------------------------------------------------

def load_required_records(file_path, required_ids, source_name):

    found_records = []
    total_rows = 0

    print(f"\nScanning {source_name}...")

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        total_rows += len(chunk)

        chunk["entity_id"] = chunk["entity_id"].astype(str)

        matching_rows = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for _, row in matching_rows.iterrows():

            found_records.append(
                normalize_record(
                    row["entity_id"],
                    row["business_name"],
                    row["business_address"],
                    row["country"]
                )
            )

        print(
            f"  Scanned {total_rows:,} rows... "
            f"Found {len(found_records):,}"
        )

        if len(found_records) >= len(required_ids):
            break

    return found_records


source2_records = load_required_records(
    SOURCE2_PATH,
    required_reference_ids,
    "Source 2"
)

source3_records = load_required_records(
    SOURCE3_PATH,
    required_reference_ids,
    "Source 3"
)

reference_records = source2_records + source3_records

print(
    f"\nReference records loaded: "
    f"{len(reference_records):,}"
)


# ---------------------------------------------------------
# 4. Build blocking index
# ---------------------------------------------------------

print("\nBuilding blocking index...")

block_index = build_block_index(reference_records)

print(
    f"Blocking keys created: "
    f"{len(block_index):,}"
)


# ---------------------------------------------------------
# 5. Normalize Source 1
# ---------------------------------------------------------

source1_records = []

for _, row in source1_df.iterrows():

    source1_records.append(
        normalize_record(
            row["entity_id"],
            row["business_name"],
            row["business_address"],
            row["country"]
        )
    )


# ---------------------------------------------------------
# 6. Measure each blocking family separately
# ---------------------------------------------------------

blocking_types = [
    "country_nameprefix",
    "country_namegram",
    "country_phonetic",
    "country_address",
]

stats = defaultdict(lambda: {
    "candidate_counts": [],
    "true_found": 0,
    "total_true": 0,
    "s1_with_match": 0,
    "s1_all_found": 0,
})


for record in source1_records:

    source1_id = record["entity_id"]

    true_matches = ground_truth.get(
        source1_id,
        set()
    )

    keys = make_blocking_keys(record)

    candidates_by_type = defaultdict(set)

    for key in keys:

        block_type = key.split(":", 1)[0]

        candidates = block_index.get(
            key,
            set()
        )

        candidates_by_type[block_type].update(
            candidates
        )

    for block_type in blocking_types:

        candidates = candidates_by_type[block_type]

        stats[block_type]["candidate_counts"].append(
            len(candidates)
        )

        if true_matches:

            stats[block_type]["s1_with_match"] += 1
            stats[block_type]["total_true"] += len(true_matches)

            found = true_matches.intersection(
                candidates
            )

            stats[block_type]["true_found"] += len(found)

            if found == true_matches:
                stats[block_type]["s1_all_found"] += 1


# ---------------------------------------------------------
# 7. Print results
# ---------------------------------------------------------

print("\n")
print("=" * 75)
print("BLOCKING DIAGNOSTIC")
print("=" * 75)

for block_type in blocking_types:

    counts = stats[block_type]["candidate_counts"]

    if counts:

        average_candidates = sum(counts) / len(counts)
        max_candidates = max(counts)

    else:

        average_candidates = 0
        max_candidates = 0

    total_true = stats[block_type]["total_true"]
    true_found = stats[block_type]["true_found"]

    if total_true > 0:
        recall = true_found / total_true
    else:
        recall = 0

    all_found = stats[block_type]["s1_all_found"]
    with_match = stats[block_type]["s1_with_match"]

    if with_match > 0:
        all_match_rate = all_found / with_match
    else:
        all_match_rate = 0

    print(f"\nBLOCKING TYPE: {block_type}")

    print(
        f"Average candidates: "
        f"{average_candidates:.2f}"
    )

    print(
        f"Maximum candidates: "
        f"{max_candidates:,}"
    )

    print(
        f"True-match recall: "
        f"{recall:.4%}"
    )

    print(
        f"S1 records with ALL matches found: "
        f"{all_found:,} / {with_match:,} "
        f"({all_match_rate:.4%})"
    )

print("\n")
print("=" * 75)
print("IMPORTANT")
print("=" * 75)

print(
    "This diagnostic index contains only the ground-truth reference "
    "records for the sample."
)

print(
    "Therefore candidate counts here are diagnostic only and are "
    "NOT the final production candidate counts."
)

print(
    "The recall numbers tell us which blocking rules are actually "
    "recovering the true matches."
)

print("\nDONE!")