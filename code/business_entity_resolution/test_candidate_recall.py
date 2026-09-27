import os
import sys
import pandas as pd

# ---------------------------------------------------------
# Allow imports from src/
# ---------------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import normalize_record
from src.blocking import build_block_index, get_candidates


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

SAMPLE_SIZE = 1000
CHUNK_SIZE = 100000


# ---------------------------------------------------------
# File locations
# ---------------------------------------------------------

TRAIN_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "train"
)

SOURCE1_PATH = os.path.join(
    TRAIN_DIR,
    "train_source1.tsv"
)

SOURCE2_PATH = os.path.join(
    TRAIN_DIR,
    "train_source2.tsv"
)

SOURCE3_PATH = os.path.join(
    TRAIN_DIR,
    "train_source3.tsv"
)

GROUND_TRUTH_PATH = os.path.join(
    TRAIN_DIR,
    "train_ground_truth.tsv"
)


# ---------------------------------------------------------
# Step 1: Load Source 1 sample
# ---------------------------------------------------------

print("Loading Source 1 sample...")

source1_df = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype=str,
    nrows=SAMPLE_SIZE
)

source1_ids = set(
    source1_df["entity_id"].astype(str)
)

print(
    f"Source 1 sample loaded: "
    f"{len(source1_ids):,}"
)


# ---------------------------------------------------------
# Step 2: Load COMPLETE ground truth
# ---------------------------------------------------------

print("\nLoading ground truth...")

ground_truth_df = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str
)

ground_truth_df["source1_entity_id"] = (
    ground_truth_df["source1_entity_id"]
    .astype(str)
)


# Keep only our Source 1 sample
ground_truth_df = ground_truth_df[
    ground_truth_df["source1_entity_id"].isin(
        source1_ids
    )
].copy()


print(
    f"Ground-truth rows for sample: "
    f"{len(ground_truth_df):,}"
)


# ---------------------------------------------------------
# Step 3: Extract the actual S2/S3 IDs required
# ---------------------------------------------------------

required_reference_ids = set()

ground_truth = {}

for _, row in ground_truth_df.iterrows():

    source1_id = str(
        row["source1_entity_id"]
    )

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

    required_reference_ids.update(
        matches
    )


print(
    f"Actual S2/S3 IDs needed for evaluation: "
    f"{len(required_reference_ids):,}"
)


# ---------------------------------------------------------
# Step 4: Find those exact records in Source 2
# ---------------------------------------------------------

def load_required_records(
    file_path,
    required_ids,
    source_name
):

    found_records = []

    total_rows = 0

    print(
        f"\nScanning {source_name}..."
    )

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        total_rows += len(chunk)

        chunk["entity_id"] = (
            chunk["entity_id"].astype(str)
        )

        matching_rows = chunk[
            chunk["entity_id"].isin(
                required_ids
            )
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
            f"  Scanned {total_rows:,} rows..."
            f" Found {len(found_records):,}"
        )

        # If all required IDs are found,
        # stop scanning this file.
        if len(found_records) >= len(required_ids):
            break

    return found_records


# ---------------------------------------------------------
# Step 5: Search Source 2
# ---------------------------------------------------------

source2_records = load_required_records(
    SOURCE2_PATH,
    required_reference_ids,
    "Source 2"
)


# ---------------------------------------------------------
# Step 6: Search Source 3
# ---------------------------------------------------------

source3_records = load_required_records(
    SOURCE3_PATH,
    required_reference_ids,
    "Source 3"
)


# ---------------------------------------------------------
# Step 7: Check whether all required IDs were found
# ---------------------------------------------------------

found_reference_ids = {
    record["entity_id"]
    for record in (
        source2_records + source3_records
    )
}

missing_reference_ids = (
    required_reference_ids -
    found_reference_ids
)

print(
    "\nReference-record verification:"
)

print(
    f"Required IDs: "
    f"{len(required_reference_ids):,}"
)

print(
    f"Found IDs:    "
    f"{len(found_reference_ids):,}"
)

print(
    f"Missing IDs:  "
    f"{len(missing_reference_ids):,}"
)


if missing_reference_ids:

    print(
        "\nWARNING:"
        " Some ground-truth reference IDs "
        "were not found."
    )

    print(
        "First missing IDs:"
    )

    for entity_id in list(
        missing_reference_ids
    )[:10]:

        print(
            f"  {entity_id}"
        )


# ---------------------------------------------------------
# Step 8: Build blocking index
# ---------------------------------------------------------

print(
    "\nBuilding blocking index..."
)

reference_records = (
    source2_records +
    source3_records
)

block_index = build_block_index(
    reference_records
)

print(
    f"Blocking keys created: "
    f"{len(block_index):,}"
)


# ---------------------------------------------------------
# Step 9: Normalize Source 1
# ---------------------------------------------------------

print(
    "\nNormalizing Source 1..."
)

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
# Step 10: Evaluate candidate recall
# ---------------------------------------------------------

print(
    "\nEvaluating candidate recall..."
)

total_true_matches = 0
total_found_matches = 0

source1_with_matches = 0
source1_with_all_matches_found = 0

candidate_counts = []


for record in source1_records:

    source1_id = record["entity_id"]

    candidates = get_candidates(
        record,
        block_index
    )

    candidate_counts.append(
        len(candidates)
    )

    true_matches = ground_truth.get(
        source1_id,
        set()
    )

    if not true_matches:
        continue

    source1_with_matches += 1

    found_matches = (
        true_matches.intersection(
            candidates
        )
    )

    total_true_matches += (
        len(true_matches)
    )

    total_found_matches += (
        len(found_matches)
    )

    if found_matches == true_matches:

        source1_with_all_matches_found += 1


# ---------------------------------------------------------
# Step 11: Calculate metrics
# ---------------------------------------------------------

if total_true_matches > 0:

    recall = (
        total_found_matches /
        total_true_matches
    )

else:

    recall = 0.0


if candidate_counts:

    average_candidates = (
        sum(candidate_counts) /
        len(candidate_counts)
    )

    max_candidates = max(
        candidate_counts
    )

else:

    average_candidates = 0
    max_candidates = 0


# ---------------------------------------------------------
# Reduction ratio
# ---------------------------------------------------------

total_reference_records = len(
    reference_records
)

if total_reference_records > 0:

    reduction_ratio = (
        1 -
        (
            average_candidates /
            total_reference_records
        )
    )

else:

    reduction_ratio = 0.0


# ---------------------------------------------------------
# Final report
# ---------------------------------------------------------

print("\n")
print("=" * 60)
print("CANDIDATE GENERATION EVALUATION")
print("=" * 60)

print(
    f"Source 1 records tested:       "
    f"{len(source1_records):,}"
)

print(
    f"Source 1 records with matches: "
    f"{source1_with_matches:,}"
)

print(
    f"\nReference records used:        "
    f"{len(reference_records):,}"
)

print(
    f"True matches:                   "
    f"{total_true_matches:,}"
)

print(
    f"True matches found:            "
    f"{total_found_matches:,}"
)

print(
    f"\nCandidate recall:              "
    f"{recall:.4%}"
)

print(
    f"Source 1 with ALL matches found:"
    f" {source1_with_all_matches_found:,}"
    f" / {source1_with_matches:,}"
)

print(
    f"\nAverage candidates per S1:     "
    f"{average_candidates:.2f}"
)

print(
    f"Maximum candidates for one S1: "
    f"{max_candidates:,}"
)

print(
    f"\nReduction ratio:               "
    f"{reduction_ratio:.4%}"
)

print("=" * 60)