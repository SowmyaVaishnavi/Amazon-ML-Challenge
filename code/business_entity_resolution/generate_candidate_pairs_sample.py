import sys
import csv
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import normalize_record


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_S1_SIZE = 1000
CHUNK_SIZE = 100_000

TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"
OUTPUT_DIR = PROJECT_ROOT / "output"

SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

OUTPUT_PATH = OUTPUT_DIR / "candidate_pairs_sample.tsv"


# ============================================================
# ADDRESS STOPWORDS
# ============================================================

ADDRESS_STOPWORDS = {
    "road", "street", "st", "rd",
    "avenue", "ave",
    "boulevard", "blvd",
    "drive", "dr",
    "lane", "ln",
    "highway", "hwy",
    "unit", "apt", "apartment",
    "suite", "ste",
    "floor",
    "near",
    "plot", "block",
    "number", "no",
    "village", "post",
    "district", "sector",
    "phase",
    "main",
    "east", "west",
    "north", "south",
    "central",
    "city", "state", "county",
}


# ============================================================
# PHONETIC KEY
# ============================================================

def make_phonetic_key(name):

    if not name:
        return ""

    words = name.split()
    signatures = []

    for word in words:

        if not word:
            continue

        word = word.lower()

        first = word[0]

        remainder = "".join(
            char
            for char in word[1:]
            if char not in "aeiou"
        )

        signature = first + remainder[:4]

        signatures.append(signature)

    return " ".join(signatures)


# ============================================================
# V5.1 BLOCKING
# ============================================================

def make_blocking_keys(record):

    country = record.get("country", "")
    name = record.get("business_name", "")
    address = record.get("business_address", "")

    keys = set()

    # --------------------------------------------------------
    # 1. FIRST WORD PREFIX
    # --------------------------------------------------------

    if country and name:

        words = name.split()

        if words:

            first_word = words[0]

            for length in (2, 3, 4):

                if len(first_word) >= length:

                    keys.add(
                        f"country_nameprefix:"
                        f"{country}:"
                        f"{first_word[:length]}"
                    )

    # --------------------------------------------------------
    # 2. FIRST TWO WORD PREFIX
    # --------------------------------------------------------

    if country and name:

        words = name.split()

        if len(words) >= 2:

            first = words[0]
            second = words[1]

            for length in (2, 3, 4):

                if (
                    len(first) >= length
                    and len(second) >= length
                ):

                    keys.add(
                        f"country_name2prefix:"
                        f"{country}:"
                        f"{first[:length]}_"
                        f"{second[:length]}"
                    )

    # --------------------------------------------------------
    # 3. SELECTED NAME TRIGRAMS
    # --------------------------------------------------------

    if country and name:

        words = name.split()

        if words and len(words[0]) >= 3:

            keys.add(
                f"country_namegram:"
                f"{country}:"
                f"{words[0][:3]}"
            )

        if len(words) >= 2 and len(words[1]) >= 3:

            keys.add(
                f"country_namegram:"
                f"{country}:"
                f"{words[1][:3]}"
            )

        if words:

            longest_word = max(words, key=len)

            if len(longest_word) >= 3:

                keys.add(
                    f"country_namegram:"
                    f"{country}:"
                    f"{longest_word[:3]}"
                )

    # --------------------------------------------------------
    # 4. PREFIX OF EVERY NAME WORD
    # --------------------------------------------------------

    if country and name:

        for word in name.split():

            if len(word) >= 3:

                keys.add(
                    f"country_namewordprefix3:"
                    f"{country}:"
                    f"{word[:3]}"
                )

            if len(word) >= 4:

                keys.add(
                    f"country_namewordprefix4:"
                    f"{country}:"
                    f"{word[:4]}"
                )

    # --------------------------------------------------------
    # 5. PHONETIC
    # --------------------------------------------------------

    if country and name:

        phonetic = make_phonetic_key(name)

        if phonetic:

            keys.add(
                f"country_phonetic:"
                f"{country}:"
                f"{phonetic}"
            )

    # --------------------------------------------------------
    # 6. ADDRESS NUMBER / ALPHANUMERIC
    # --------------------------------------------------------

    if country and address:

        for token in address.split():

            if (
                len(token) >= 3
                and any(char.isdigit() for char in token)
            ):

                keys.add(
                    f"country_address_id:"
                    f"{country}:"
                    f"{token}"
                )

    # --------------------------------------------------------
    # 7. LONG ADDRESS WORDS
    # --------------------------------------------------------

    if country and address:

        for token in address.split():

            if (
                len(token) >= 7
                and token.isalpha()
                and token not in ADDRESS_STOPWORDS
            ):

                keys.add(
                    f"country_address_word:"
                    f"{country}:"
                    f"{token}"
                )

    # --------------------------------------------------------
    # 8. SHORT ADDRESS WORDS
    # --------------------------------------------------------

    if country and address:

        for token in address.split():

            if (
                4 <= len(token) <= 6
                and token.isalpha()
                and token not in ADDRESS_STOPWORDS
            ):

                keys.add(
                    f"country_address_short:"
                    f"{country}:"
                    f"{token}"
                )

    # --------------------------------------------------------
    # 9. V5.1 SHORT NAME BLOCK
    # --------------------------------------------------------

    if country and name:

        for word in name.split():

            if 1 <= len(word) <= 3:

                keys.add(
                    f"country_shortname:"
                    f"{country}:"
                    f"{word}"
                )

    return keys


# ============================================================
# LOAD FIRST 1000 SOURCE-1 RECORDS
# ============================================================

def load_sample_source1():

    s1 = pd.read_csv(
        SOURCE1_PATH,
        sep="\t",
        dtype=str,
        nrows=SAMPLE_S1_SIZE
    ).fillna("")

    return s1


# ============================================================
# LOAD GROUND TRUTH FOR SAMPLE
# ============================================================

def load_ground_truth(sample_s1_ids):

    gt = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str
    ).fillna("")

    gt = gt[
        gt["source1_entity_id"].isin(sample_s1_ids)
    ].copy()

    return gt


# ============================================================
# GET TRUE REFERENCE IDS
# ============================================================

def get_true_reference_ids(gt):

    true_ids = set()

    for value in gt["matched_entity_ids"]:

        if not value:
            continue

        # Ground truth contains comma-separated IDs.
        for entity_id in str(value).split(","):

            entity_id = entity_id.strip()

            if entity_id:

                true_ids.add(entity_id)

    return true_ids


# ============================================================
# LOAD ONLY TRUE REFERENCE RECORDS
# ============================================================

def load_reference_records(
    true_reference_ids
):

    records = []

    for source_name, path in [
        ("S2", SOURCE2_PATH),
        ("S3", SOURCE3_PATH)
    ]:

        print()
        print(
            f"Scanning {source_name}..."
        )

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE
        ):

            chunk = chunk.fillna("")

            matched = chunk[
                chunk["entity_id"].isin(
                    true_reference_ids
                )
            ]

            for row in matched.itertuples(
                index=False
            ):

                record = normalize_record(
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                    row.country
                )

                records.append(
                    (
                        source_name,
                        record
                    )
                )

    return records


# ============================================================
# BUILD SMALL REFERENCE INDEX
# ============================================================

def build_reference_index(
    reference_records
):

    index = defaultdict(list)

    for source_name, record in reference_records:

        keys = make_blocking_keys(
            record
        )

        for key in keys:

            index[key].append(
                (
                    source_name,
                    record["entity_id"]
                )
            )

    return index


# ============================================================
# GENERATE SAMPLE CANDIDATES
# ============================================================

def generate_candidates(
    sample_s1,
    index
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    total_candidates = 0
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
        # CORRECT TWO-COLUMN FORMAT
        # ====================================================

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_ids"
        ])

        for row in sample_s1.itertuples(
            index=False
        ):

            record = normalize_record(
                row.entity_id,
                row.business_name,
                row.business_address,
                row.country
            )

            candidates = set()

            keys = make_blocking_keys(
                record
            )

            for key in keys:

                matches = index.get(
                    key,
                    []
                )

                for source_name, entity_id in matches:

                    candidates.add(
                        entity_id
                    )

            candidates = sorted(
                candidates
            )

            total_candidates += len(
                candidates
            )

            max_candidates = max(
                max_candidates,
                len(candidates)
            )

            # ONE ROW PER S1
            writer.writerow([
                record["entity_id"],
                ",".join(candidates)
            ])

    print()
    print("=" * 60)
    print("SAMPLE CANDIDATE GENERATION COMPLETE")
    print("=" * 60)

    print(
        f"Source-1 sample size: {len(sample_s1):,}"
    )

    print(
        f"Total candidates: {total_candidates:,}"
    )

    print(
        f"Average candidates/S1: "
        f"{total_candidates / len(sample_s1):.2f}"
    )

    print(
        f"Maximum candidates/S1: {max_candidates:,}"
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("AMAZON ML CHALLENGE")
    print("V5.1 SMALL SAMPLE FOR PERSON B")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. First 1000 S1 records
    # --------------------------------------------------------

    sample_s1 = load_sample_source1()

    sample_s1_ids = set(
        sample_s1["entity_id"]
    )

    print(
        f"\nLoaded {len(sample_s1):,} "
        f"Source-1 records."
    )

    # --------------------------------------------------------
    # 2. Ground truth
    # --------------------------------------------------------

    gt = load_ground_truth(
        sample_s1_ids
    )

    # --------------------------------------------------------
    # 3. True reference IDs
    # --------------------------------------------------------

    true_reference_ids = get_true_reference_ids(
        gt
    )

    print(
        f"True reference IDs: "
        f"{len(true_reference_ids):,}"
    )

    # --------------------------------------------------------
    # 4. Load true S2/S3 records
    # --------------------------------------------------------

    reference_records = load_reference_records(
        true_reference_ids
    )

    print(
        f"Reference records loaded: "
        f"{len(reference_records):,}"
    )

    # --------------------------------------------------------
    # 5. Build V5.1 index
    # --------------------------------------------------------

    index = build_reference_index(
        reference_records
    )

    print(
        f"Unique blocking keys: "
        f"{len(index):,}"
    )

    # --------------------------------------------------------
    # 6. Generate sample candidate file
    # --------------------------------------------------------

    generate_candidates(
        sample_s1,
        index
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)