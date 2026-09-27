import os
import sys
from collections import Counter

import pandas as pd

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
# V4 BLOCKING KEYS
# ---------------------------------------------------------

def make_v4_blocking_keys(record):

    country = record.get("country", "")
    name = record.get("business_name", "")
    address = record.get("business_address", "")

    keys = set()

    # NAME PREFIX
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

        # Targeted trigrams
        if words:

            first_word = words[0]

            if len(first_word) >= 3:
                keys.add(
                    f"country_firstgram:{country}:{first_word[:3]}"
                )

            if len(words) >= 2:

                second_word = words[1]

                if len(second_word) >= 3:
                    keys.add(
                        f"country_secondgram:{country}:{second_word[:3]}"
                    )

            longest_word = max(words, key=len)

            if len(longest_word) >= 3:
                keys.add(
                    f"country_longestgram:{country}:{longest_word[:3]}"
                )

    # PHONETIC
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

            signatures.append(
                first + remainder[:4]
            )

        phonetic = " ".join(signatures)

        if phonetic:

            keys.add(
                f"country_phonetic:{country}:{phonetic}"
            )

    # TARGETED ADDRESS
    if country and address:

        address_tokens = address.split()

        for token in address_tokens:

            # Numeric / alphanumeric address identifiers
            if any(char.isdigit() for char in token):

                if len(token) >= 3:

                    keys.add(
                        f"country_address_id:{country}:{token}"
                    )

            # Longer distinctive address words
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
# LOAD REQUIRED REFERENCE RECORDS
# ---------------------------------------------------------

def load_required_records(file_path, required_ids, source_name):

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

reference_records = source2_records + source3_records


# ---------------------------------------------------------
# BUILD V4 INDEX
# ---------------------------------------------------------

print("\nBuilding V4 index...")

v4_index = {}

for record in reference_records:

    entity_id = record["entity_id"]

    for key in make_v4_blocking_keys(record):

        if key not in v4_index:
            v4_index[key] = set()

        v4_index[key].add(entity_id)


print(
    f"V4 blocking keys: {len(v4_index):,}"
)


# ---------------------------------------------------------
# BUILD V1 INDEX
# ---------------------------------------------------------
# V1 is used only as a diagnostic.
# We know V1 recovered all true matches in this sample.

print("\nBuilding V1 diagnostic index...")

v1_index = build_block_index(
    reference_records
)


# ---------------------------------------------------------
# NORMALIZE SOURCE 1
# ---------------------------------------------------------

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
# FIND V4 MISSES
# ---------------------------------------------------------

print("\nFinding V4 missed matches...")

missed_rows = []

reason_counter = Counter()

total_true_matches = 0
total_found_matches = 0


for record in source1_records:

    source1_id = record["entity_id"]

    true_matches = ground_truth.get(
        source1_id,
        set()
    )

    if not true_matches:
        continue

    # Generate V4 candidates
    v4_candidates = set()

    for key in make_v4_blocking_keys(record):

        v4_candidates.update(
            v4_index.get(key, set())
        )

    total_true_matches += len(true_matches)

    found_matches = (
        true_matches.intersection(
            v4_candidates
        )
    )

    total_found_matches += len(found_matches)

    missed_matches = (
        true_matches - v4_candidates
    )

    for missed_id in missed_matches:

        # Find reference record
        reference_record = next(
            (
                r for r in reference_records
                if r["entity_id"] == missed_id
            ),
            None
        )

        if reference_record is None:
            continue

        # Check how V1 recovered this match
        v1_shared_keys = []

        for key in make_blocking_keys(record):

            if missed_id in v1_index.get(key, set()):

                v1_shared_keys.append(key)

        # Classify why V4 missed it
        categories = set()

        for key in v1_shared_keys:

            if key.startswith("country_namegram:"):
                categories.add("namegram")

            elif key.startswith("country_address:"):
                categories.add("address")

            elif key.startswith("country_nameprefix:"):
                categories.add("nameprefix")

            elif key.startswith("country_phonetic:"):
                categories.add("phonetic")

        if categories:

            reason = ",".join(
                sorted(categories)
            )

            for category in categories:
                reason_counter[category] += 1

        else:

            reason = "unknown"
            reason_counter["unknown"] += 1

        missed_rows.append({

            "source1_entity_id":
                source1_id,

            "missed_reference_entity_id":
                missed_id,

            "source1_name":
                record["business_name"],

            "reference_name":
                reference_record["business_name"],

            "source1_address":
                record["business_address"],

            "reference_address":
                reference_record["business_address"],

            "country":
                record["country"],

            "v1_recovery_type":
                reason,

            "v1_shared_keys":
                " || ".join(
                    v1_shared_keys
                ),

            "v4_source1_keys":
                " || ".join(
                    sorted(
                        make_v4_blocking_keys(record)
                    )
                ),

            "v4_reference_keys":
                " || ".join(
                    sorted(
                        make_v4_blocking_keys(
                            reference_record
                        )
                    )
                ),
        })


# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

missed_count = (
    total_true_matches -
    total_found_matches
)

recall = (
    total_found_matches /
    total_true_matches
    if total_true_matches
    else 0
)

print("\n")
print("=" * 70)
print("V4 MISS ANALYSIS")
print("=" * 70)

print(
    f"Total true matches:       "
    f"{total_true_matches:,}"
)

print(
    f"True matches found:       "
    f"{total_found_matches:,}"
)

print(
    f"True matches missed:      "
    f"{missed_count:,}"
)

print(
    f"V4 recall:                "
    f"{recall:.4%}"
)

print("\nWhy V4 misses matches:")
print("-" * 70)

for reason, count in reason_counter.most_common():

    print(
        f"{reason:15s}: {count:,}"
    )


# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

output_dir = os.path.join(
    PROJECT_ROOT,
    "output"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "v4_miss_analysis.tsv"
)

miss_df = pd.DataFrame(
    missed_rows
)

miss_df.to_csv(
    output_path,
    sep="\t",
    index=False
)

print("\nDetailed report saved to:")
print(output_path)


# ---------------------------------------------------------
# SHOW MISSES
# ---------------------------------------------------------

print("\n")
print("=" * 70)
print("V4 MISSED MATCHES")
print("=" * 70)

if len(miss_df) == 0:

    print("🎉 V4 missed ZERO matches!")

else:

    for _, row in miss_df.iterrows():

        print("\nS1:", row["source1_entity_id"])

        print(
            "S1 name:",
            row["source1_name"]
        )

        print(
            "S1 address:",
            row["source1_address"]
        )

        print(
            "TRUE MATCH:",
            row["missed_reference_entity_id"]
        )

        print(
            "Reference name:",
            row["reference_name"]
        )

        print(
            "Reference address:",
            row["reference_address"]
        )

        print(
            "Country:",
            row["country"]
        )

        print(
            "V1 recovery type:",
            row["v1_recovery_type"]
        )

        print(
            "V1 shared keys:",
            row["v1_shared_keys"]
        )

        print("-" * 70)


print("\nDONE!")