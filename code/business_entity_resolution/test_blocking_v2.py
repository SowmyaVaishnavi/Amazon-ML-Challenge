import os
import sys
import pandas as pd
from collections import defaultdict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import normalize_record
from src.blocking import build_block_index


SAMPLE_SIZE = 1000
CHUNK_SIZE = 100000

TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "train")

SOURCE1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
SOURCE2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
SOURCE3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
GROUND_TRUTH_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")


# ---------------------------------------------------------
# V2 BLOCKING RULE
# ---------------------------------------------------------

def make_v2_blocking_keys(record):

    country = record.get("country", "")
    name = record.get("business_name", "")
    address = record.get("business_address", "")

    keys = set()

    if not country or not name:
        return keys

    words = name.split()

    # -----------------------------------------------------
    # Rule 1: First meaningful word prefix
    # -----------------------------------------------------

    if words:

        first_word = words[0]

        if len(first_word) >= 4:
            keys.add(
                f"country_nameprefix4:{country}:{first_word[:4]}"
            )

        if len(first_word) >= 5:
            keys.add(
                f"country_nameprefix5:{country}:{first_word[:5]}"
            )


    # -----------------------------------------------------
    # Rule 2: First word + second word prefix
    # -----------------------------------------------------

    if len(words) >= 2:

        first_word = words[0]
        second_word = words[1]

        if len(first_word) >= 3 and len(second_word) >= 3:

            keys.add(
                f"country_firstwords:{country}:"
                f"{first_word[:3]}_{second_word[:3]}"
            )


    # -----------------------------------------------------
    # Rule 3: Only FIRST name trigram
    # -----------------------------------------------------

    first_word = words[0]

    if len(first_word) >= 3:

        keys.add(
            f"country_firstgram:{country}:{first_word[:3]}"
        )


    # -----------------------------------------------------
    # Rule 4: Address number
    # -----------------------------------------------------

    if address:

        address_tokens = address.split()

        for token in address_tokens:

            if token.isdigit() and len(token) >= 2:

                keys.add(
                    f"country_addressnumber:{country}:{token}"
                )

                break


    return keys


# ---------------------------------------------------------
# Load Source 1
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
    f"Source 1 sample: "
    f"{len(source1_ids):,}"
)


# ---------------------------------------------------------
# Load ground truth
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

    required_reference_ids.update(matches)


print(
    f"Ground-truth S1 records: "
    f"{len(ground_truth):,}"
)

print(
    f"Required reference IDs: "
    f"{len(required_reference_ids):,}"
)


# ---------------------------------------------------------
# Find actual matching records in S2/S3
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


reference_records = (
    source2_records +
    source3_records
)


print(
    f"\nReference records loaded: "
    f"{len(reference_records):,}"
)


# ---------------------------------------------------------
# Build V2 index
# ---------------------------------------------------------

print("\nBuilding V2 blocking index...")

# We cannot use the old build_block_index because
# V2 has different blocking keys.

v2_index = defaultdict(set)

for record in reference_records:

    keys = make_v2_blocking_keys(record)

    for key in keys:

        v2_index[key].add(
            record["entity_id"]
        )


print(
    f"V2 blocking keys: "
    f"{len(v2_index):,}"
)


# ---------------------------------------------------------
# Normalize Source 1
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
# Evaluate V2
# ---------------------------------------------------------

total_true_matches = 0
total_found_matches = 0

s1_with_matches = 0
s1_all_matches_found = 0

candidate_counts = []


for record in source1_records:

    source1_id = record["entity_id"]

    true_matches = ground_truth.get(
        source1_id,
        set()
    )

    keys = make_v2_blocking_keys(record)

    candidates = set()

    for key in keys:

        candidates.update(
            v2_index.get(key, set())
        )

    candidate_counts.append(
        len(candidates)
    )

    if not true_matches:
        continue

    s1_with_matches += 1

    total_true_matches += len(
        true_matches
    )

    found_matches = (
        true_matches.intersection(
            candidates
        )
    )

    total_found_matches += len(
        found_matches
    )

    if found_matches == true_matches:

        s1_all_matches_found += 1


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

if total_true_matches > 0:

    recall = (
        total_found_matches /
        total_true_matches
    )

else:

    recall = 0


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


print("\n")
print("=" * 70)
print("V2 BLOCKING RESULTS")
print("=" * 70)

print(
    f"Source 1 records tested:       "
    f"{len(source1_records):,}"
)

print(
    f"Source 1 records with matches: "
    f"{s1_with_matches:,}"
)

print()

print(
    f"True matches:                   "
    f"{total_true_matches:,}"
)

print(
    f"True matches found:             "
    f"{total_found_matches:,}"
)

print()

print(
    f"V2 candidate recall:             "
    f"{recall:.4%}"
)

print(
    f"S1 with ALL matches found:      "
    f"{s1_all_matches_found:,} / "
    f"{s1_with_matches:,}"
)

print()

print(
    f"Average candidates per S1:      "
    f"{average_candidates:.2f}"
)

print(
    f"Maximum candidates for one S1:  "
    f"{max_candidates:,}"
)

print("=" * 70)

print("\nDONE! V2 experiment finished.")