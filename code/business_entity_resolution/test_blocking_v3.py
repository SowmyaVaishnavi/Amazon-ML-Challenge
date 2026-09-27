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
# V3 BLOCKING
# ---------------------------------------------------------

def make_v3_blocking_keys(record):

    country = record.get("country", "")
    name = record.get("business_name", "")
    address = record.get("business_address", "")

    keys = set()

    # -----------------------------------------------------
    # 1. Keep the original name-prefix rules
    # -----------------------------------------------------

    if country and name:

        words = name.split()

        if words:

            first_word = words[0]

            for length in (2, 3, 4):

                if len(first_word) >= length:

                    keys.add(
                        f"country_nameprefix:{country}:"
                        f"{first_word[:length]}"
                    )


    # -----------------------------------------------------
    # 2. Use ONLY selected name trigrams
    #
    # Instead of every trigram, use:
    # - first 3 characters of first word
    # - first 3 characters of second word
    # - one trigram from the longest word
    # -----------------------------------------------------

    if country and name:

        words = name.split()

        if words:

            first_word = words[0]

            if len(first_word) >= 3:

                keys.add(
                    f"country_namegram:{country}:"
                    f"{first_word[:3]}"
                )

            if len(words) >= 2:

                second_word = words[1]

                if len(second_word) >= 3:

                    keys.add(
                        f"country_namegram:{country}:"
                        f"{second_word[:3]}"
                    )

            longest_word = max(
                words,
                key=len
            )

            if len(longest_word) >= 4:

                middle = max(
                    0,
                    (len(longest_word) // 2) - 1
                )

                gram = longest_word[
                    middle:middle + 3
                ]

                if len(gram) == 3:

                    keys.add(
                        f"country_namegram:{country}:"
                        f"{gram}"
                    )


    # -----------------------------------------------------
    # 3. Keep phonetic blocking
    # -----------------------------------------------------

    if country and name:

        words = name.split()

        signatures = []

        for word in words:

            if not word:
                continue

            first = word[0]

            remainder = "".join(
                char
                for char in word[1:]
                if char not in "aeiou"
            )

            signature = (
                first +
                remainder[:4]
            )

            signatures.append(signature)

        phonetic = " ".join(signatures)

        if phonetic:

            keys.add(
                f"country_phonetic:{country}:"
                f"{phonetic}"
            )


    # -----------------------------------------------------
    # 4. Address number
    # -----------------------------------------------------

    if country and address:

        address_tokens = address.split()

        for token in address_tokens:

            if token.isdigit() and len(token) >= 2:

                keys.add(
                    f"country_addressnumber:{country}:"
                    f"{token}"
                )

                break


    return keys


# ---------------------------------------------------------
# Load Source 1 sample
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
    f"Source 1 sample: {len(source1_ids):,}"
)


# ---------------------------------------------------------
# Ground truth
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
# Find true reference records
# ---------------------------------------------------------

def load_required_records(
    file_path,
    required_ids,
    source_name
):

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
# Build V3 index
# ---------------------------------------------------------

print("\nBuilding V3 blocking index...")

v3_index = defaultdict(set)

for record in reference_records:

    keys = make_v3_blocking_keys(record)

    for key in keys:

        v3_index[key].add(
            record["entity_id"]
        )


print(
    f"V3 blocking keys: "
    f"{len(v3_index):,}"
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
# Evaluate
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

    keys = make_v3_blocking_keys(record)

    candidates = set()

    for key in keys:

        candidates.update(
            v3_index.get(key, set())
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

recall = (
    total_found_matches /
    total_true_matches
    if total_true_matches > 0
    else 0
)

average_candidates = (
    sum(candidate_counts) /
    len(candidate_counts)
    if candidate_counts
    else 0
)

max_candidates = (
    max(candidate_counts)
    if candidate_counts
    else 0
)


print("\n")
print("=" * 70)
print("V3 BLOCKING RESULTS")
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
    f"V3 candidate recall:             "
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

print("\nDONE! V3 experiment finished.")