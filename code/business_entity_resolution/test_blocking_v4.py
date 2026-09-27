import os
import sys
import pandas as pd

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
# V4 BLOCKING KEYS
# ---------------------------------------------------------

def make_v4_blocking_keys(record):

    country = record.get("country", "")
    name = record.get("business_name", "")
    address = record.get("business_address", "")

    keys = set()

    # -----------------------------------------------------
    # 1. NAME PREFIX
    # -----------------------------------------------------

    if country and name:

        words = name.split()

        if words:

            first_word = words[0]

            for length in (2, 3, 4):

                if len(first_word) >= length:

                    keys.add(
                        f"country_nameprefix:{country}:{first_word[:length]}"
                    )

        # First two words
        if len(words) >= 2:

            first_word = words[0]
            second_word = words[1]

            if len(first_word) >= 3 and len(second_word) >= 3:

                keys.add(
                    f"country_twoprefix:{country}:"
                    f"{first_word[:3]}:{second_word[:3]}"
                )

    # -----------------------------------------------------
    # 2. TARGETED NAME TRIGRAMS
    # -----------------------------------------------------

    if country and name:

        words = name.split()

        if words:

            # First word beginning
            first_word = words[0]

            if len(first_word) >= 3:

                keys.add(
                    f"country_firstgram:{country}:{first_word[:3]}"
                )

            # Second word beginning
            if len(words) >= 2:

                second_word = words[1]

                if len(second_word) >= 3:

                    keys.add(
                        f"country_secondgram:{country}:{second_word[:3]}"
                    )

            # Longest word beginning
            longest_word = max(words, key=len)

            if len(longest_word) >= 3:

                keys.add(
                    f"country_longestgram:{country}:{longest_word[:3]}"
                )

    # -----------------------------------------------------
    # 3. PHONETIC
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

            signature = first + remainder[:4]

            signatures.append(signature)

        phonetic = " ".join(signatures)

        if phonetic:

            keys.add(
                f"country_phonetic:{country}:{phonetic}"
            )

    # -----------------------------------------------------
    # 4. TARGETED ADDRESS BLOCKING
    # -----------------------------------------------------

    if country and address:

        address_tokens = address.split()

        # Keep only useful-looking address tokens.
        #
        # We intentionally DO NOT use every address word,
        # because that caused too many candidates in V1.

        for token in address_tokens:

            # Numbers / alphanumeric numbers
            if any(char.isdigit() for char in token):

                if len(token) >= 3:

                    keys.add(
                        f"country_address_id:{country}:{token}"
                    )

            # Longer distinctive words
            elif len(token) >= 7:

                keys.add(
                    f"country_address_word:{country}:{token}"
                )

    return keys


# ---------------------------------------------------------
# LOAD SOURCE 1
# ---------------------------------------------------------

print("\nLoading Source 1 sample...")

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
    f"Source 1 records: {len(source1_ids):,}"
)


# ---------------------------------------------------------
# LOAD GROUND TRUTH
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


print(
    f"Required reference records: "
    f"{len(required_reference_ids):,}"
)


# ---------------------------------------------------------
# LOAD REQUIRED S2/S3 RECORDS
# ---------------------------------------------------------

def load_required_records(
    file_path,
    required_ids,
    source_name
):

    records = []

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

            records.append(
                normalize_record(
                    row["entity_id"],
                    row["business_name"],
                    row["business_address"],
                    row["country"],
                )
            )

        print(
            f"  Scanned {total_rows:,} rows..."
            f" Found {len(records):,}"
        )

        if len(records) >= len(required_ids):
            break

    return records


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


# ---------------------------------------------------------
# BUILD V4 INDEX
# ---------------------------------------------------------

print("\nBuilding V4 blocking index...")

v4_index = {}

for record in reference_records:

    entity_id = record["entity_id"]

    keys = make_v4_blocking_keys(record)

    for key in keys:

        if key not in v4_index:

            v4_index[key] = set()

        v4_index[key].add(entity_id)


print(
    f"V4 blocking keys: "
    f"{len(v4_index):,}"
)


# ---------------------------------------------------------
# NORMALIZE SOURCE 1
# ---------------------------------------------------------

print("\nNormalizing Source 1...")

source1_records = []

for _, row in source1_df.iterrows():

    source1_records.append(
        normalize_record(
            row["entity_id"],
            row["business_name"],
            row["business_address"],
            row["country"],
        )
    )


# ---------------------------------------------------------
# EVALUATE V4
# ---------------------------------------------------------

print("\nEvaluating V4...")

total_true_matches = 0
total_found_matches = 0

source1_with_matches = 0
source1_with_all_matches_found = 0

candidate_counts = []


for record in source1_records:

    source1_id = record["entity_id"]

    true_matches = ground_truth.get(
        source1_id,
        set()
    )

    candidates = set()

    keys = make_v4_blocking_keys(record)

    for key in keys:

        candidates.update(
            v4_index.get(key, set())
        )

    candidate_counts.append(
        len(candidates)
    )

    if not true_matches:
        continue

    source1_with_matches += 1

    found_matches = (
        true_matches.intersection(candidates)
    )

    total_true_matches += len(true_matches)

    total_found_matches += len(found_matches)

    if found_matches == true_matches:

        source1_with_all_matches_found += 1


# ---------------------------------------------------------
# METRICS
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

    maximum_candidates = max(
        candidate_counts
    )

else:

    average_candidates = 0
    maximum_candidates = 0


reference_count = len(reference_records)

if reference_count > 0:

    reduction_ratio = (
        1 -
        average_candidates /
        reference_count
    )

else:

    reduction_ratio = 0


print("\n")
print("=" * 70)
print("V4 BLOCKING RESULTS")
print("=" * 70)

print(
    f"Source 1 records tested:       "
    f"{len(source1_records):,}"
)

print(
    f"Source 1 records with matches: "
    f"{source1_with_matches:,}"
)

print()

print(
    f"True matches:                  "
    f"{total_true_matches:,}"
)

print(
    f"True matches found:            "
    f"{total_found_matches:,}"
)

print(
    f"True matches missed:           "
    f"{total_true_matches - total_found_matches:,}"
)

print()

print(
    f"V4 candidate recall:            "
    f"{recall:.4%}"
)

print(
    f"S1 with ALL matches found:     "
    f"{source1_with_all_matches_found:,} "
    f"/ {source1_with_matches:,}"
)

print()

print(
    f"Average candidates per S1:     "
    f"{average_candidates:.2f}"
)

print(
    f"Maximum candidates for one S1: "
    f"{maximum_candidates:,}"
)

print()

print(
    f"Reduction ratio:               "
    f"{reduction_ratio:.4%}"
)

print("=" * 70)