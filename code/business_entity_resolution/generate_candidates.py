from pathlib import Path
from collections import defaultdict
import csv
import sys

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make the project root visible to Python
sys.path.append(str(PROJECT_ROOT))

from src.normalization import normalize_record


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"
TEST_DIR = PROJECT_ROOT / "datasets" / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEST FILES
# ============================================================

TEST_SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = TEST_DIR / "test_source3.tsv"

OUTPUT_PATH = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# CHUNK SIZE
# ============================================================

CHUNK_SIZE = 200_000


# ============================================================
# V5.1 BLOCKING
# ============================================================

def make_blocking_keys(record):

    name = record["business_name"]
    address = record["business_address"]
    country = record["country"]

    keys = set()

    # ========================================================
    # BUSINESS NAME
    # ========================================================

    name_tokens = name.split()

    # --------------------------------------------------------
    # First-word prefixes
    # --------------------------------------------------------

    if name_tokens:

        first_word = name_tokens[0]

        if len(first_word) >= 2:
            keys.add(
                f"nameprefix:{country}:{first_word[:2]}"
            )

        if len(first_word) >= 3:
            keys.add(
                f"nameprefix:{country}:{first_word[:3]}"
            )

        if len(first_word) >= 4:
            keys.add(
                f"nameprefix:{country}:{first_word[:4]}"
            )

    # --------------------------------------------------------
    # Two-word prefix
    # --------------------------------------------------------

    if len(name_tokens) >= 2:

        first_two = (
            name_tokens[0][:3]
            + "_"
            + name_tokens[1][:3]
        )

        keys.add(
            f"name2prefix:{country}:{first_two}"
        )

    # --------------------------------------------------------
    # Name trigrams
    # --------------------------------------------------------

    for token in name_tokens:

        if len(token) >= 3:

            for i in range(len(token) - 2):

                trigram = token[i:i + 3]

                keys.add(
                    f"namegram:{country}:{trigram}"
                )

    # --------------------------------------------------------
    # Per-word prefixes
    # --------------------------------------------------------

    for token in name_tokens:

        if len(token) >= 4:

            keys.add(
                f"nameword:{country}:{token[:4]}"
            )

    # --------------------------------------------------------
    # V5.1 SHORT NAME BLOCK
    # --------------------------------------------------------

    for token in name_tokens:

        if 1 <= len(token) <= 3:

            keys.add(
                f"shortname:{country}:{token}"
            )

    # ========================================================
    # ADDRESS
    # ========================================================

    address_tokens = address.split()

    # --------------------------------------------------------
    # Full normalized address
    # --------------------------------------------------------

    if address:

        keys.add(
            f"address:{country}:{address}"
        )

    # --------------------------------------------------------
    # Long address words
    # --------------------------------------------------------

    for token in address_tokens:

        if len(token) >= 5:

            keys.add(
                f"addressword:{country}:{token[:5]}"
            )

        if len(token) >= 4:

            keys.add(
                f"addressword4:{country}:{token[:4]}"
            )

    # --------------------------------------------------------
    # Short address words
    # --------------------------------------------------------

    for token in address_tokens:

        if len(token) >= 3:

            keys.add(
                f"addressshort:{country}:{token}"
            )

    return keys


# ============================================================
# BUILD TEST REFERENCE INDEX
# ============================================================

def build_reference_index():

    print()
    print("=" * 70)
    print("BUILDING TEST REFERENCE INDEX")
    print("=" * 70)

    index = defaultdict(list)

    source_files = [
        ("S2", TEST_SOURCE2_PATH),
        ("S3", TEST_SOURCE3_PATH),
    ]

    total_records = 0

    for source_name, source_path in source_files:

        print()
        print(f"Reading {source_name}: {source_path}")

        chunk_number = 0

        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE
        ):

            chunk = chunk.fillna("")

            chunk_number += 1

            for row in chunk.itertuples(index=False):

                record = normalize_record(
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                    row.country
                )

                keys = make_blocking_keys(record)

                for key in keys:

                    index[key].append(
                        (
                            source_name,
                            record["entity_id"]
                        )
                    )

                total_records += 1

            print(
                f"  {source_name} chunks processed: "
                f"{chunk_number} | "
                f"records indexed: "
                f"{total_records:,}"
            )

    print()
    print(
        f"Total S2/S3 records indexed: "
        f"{total_records:,}"
    )

    print(
        f"Total blocking keys: "
        f"{len(index):,}"
    )

    return index


# ============================================================
# GENERATE TEST CANDIDATES
# ============================================================

def generate_candidates(index):

    print()
    print("=" * 70)
    print("GENERATING FULL TEST CANDIDATES")
    print("=" * 70)

    total_s1 = 0
    total_candidate_pairs = 0
    max_candidates = 0

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as output_file:

        writer = csv.writer(
            output_file,
            delimiter="\t"
        )

        # ====================================================
        # OFFICIAL OUTPUT FORMAT
        # ====================================================

        writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids"
            ]
        )

        chunk_number = 0

        for chunk in pd.read_csv(
            TEST_SOURCE1_PATH,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE
        ):

            chunk = chunk.fillna("")

            chunk_number += 1

            for row in chunk.itertuples(index=False):

                total_s1 += 1

                record = normalize_record(
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                    row.country
                )

                keys = make_blocking_keys(record)

                # ------------------------------------------------
                # Collect unique candidate IDs
                # ------------------------------------------------

                candidates = set()

                for key in keys:

                    matches = index.get(
                        key,
                        []
                    )

                    for source_name, entity_id in matches:

                        candidates.add(entity_id)

                candidate_count = len(candidates)

                total_candidate_pairs += candidate_count

                if candidate_count > max_candidates:
                    max_candidates = candidate_count

                # ------------------------------------------------
                # Sort for reproducibility
                # ------------------------------------------------

                sorted_candidates = sorted(
                    candidates
                )

                candidate_string = ",".join(
                    sorted_candidates
                )

                # ------------------------------------------------
                # EXACTLY ONE ROW PER S1
                # ------------------------------------------------

                writer.writerow(
                    [
                        record["entity_id"],
                        candidate_string
                    ]
                )

            print()
            print(
                f"Source 1 chunks processed: "
                f"{chunk_number}"
            )

            print(
                f"Source 1 records processed: "
                f"{total_s1:,}"
            )

            print(
                f"Candidate IDs generated: "
                f"{total_candidate_pairs:,}"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    average_candidates = (
        total_candidate_pairs / total_s1
        if total_s1 > 0
        else 0
    )

    print()
    print("=" * 70)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Source 1 records: "
        f"{total_s1:,}"
    )

    print(
        f"Candidate IDs: "
        f"{total_candidate_pairs:,}"
    )

    print(
        f"Average candidates per S1: "
        f"{average_candidates:.2f}"
    )

    print(
        f"Maximum candidates for one S1: "
        f"{max_candidates:,}"
    )

    print()
    print(
        f"Output file: "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("🚀 AMAZON ML CHALLENGE")
    print("🚀 V5.1 FULL TEST CANDIDATE GENERATION")
    print()

    index = build_reference_index()

    generate_candidates(index)

    print()
    print("✅ DONE.")