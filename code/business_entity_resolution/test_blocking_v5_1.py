import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import normalize_record


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 1000
CHUNK_SIZE = 200_000


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
# NAME HELPERS
# ============================================================

def make_name_ngrams(name, n=3):

    if not name:
        return set()

    text = f"  {name}  "

    if len(text) < n:
        return {text}

    return {
        text[i:i+n]
        for i in range(len(text) - n + 1)
    }


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

def make_v5_1_blocking_keys(record):

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

        words = name.split()

        for word in words:

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


    # ========================================================
    # 9. V5.1 TARGETED MICRO-BLOCK
    #
    # Only activates for VERY short / unusual business names.
    #
    # This targets cases like:
    #
    # S1: "b biotechnologies"
    # S3: "b llc services"
    #
    # We use the first short name token together with country.
    #
    # This is deliberately narrow so normal records do not
    # suddenly receive huge candidate sets.
    # ========================================================

    if country and name:

        words = name.split()

        short_words = [
            word
            for word in words
            if 1 <= len(word) <= 3
        ]

        if short_words:

            for word in short_words:

                keys.add(
                    f"country_shortname:"
                    f"{country}:"
                    f"{word}"
                )


    return keys


# ============================================================
# BUILD INDEX
# ============================================================

def build_block_index(records):

    index = defaultdict(set)

    for record in records:

        entity_id = record["entity_id"]

        keys = make_v5_1_blocking_keys(record)

        for key in keys:

            index[key].add(entity_id)

    return index


# ============================================================
# GET CANDIDATES
# ============================================================

def get_candidates(record, block_index):

    candidates = set()

    keys = make_v5_1_blocking_keys(record)

    for key in keys:

        candidates.update(
            block_index.get(key, set())
        )

    return candidates


# ============================================================
# LOAD DATA
# ============================================================

train_dir = PROJECT_ROOT / "datasets" / "train"

source1_path = train_dir / "train_source1.tsv"
source2_path = train_dir / "train_source2.tsv"
source3_path = train_dir / "train_source3.tsv"
ground_truth_path = train_dir / "train_ground_truth.tsv"


print("\nLoading Source 1 sample...")

source1 = pd.read_csv(
    source1_path,
    sep="\t",
    nrows=SAMPLE_SIZE,
    dtype=str
).fillna("")


print("Loading ground truth...")

ground_truth = pd.read_csv(
    ground_truth_path,
    sep="\t",
    dtype=str
).fillna("")


ground_truth = ground_truth[
    ground_truth["source1_entity_id"].isin(
        source1["entity_id"]
    )
]


# ============================================================
# GET TRUE IDS
# ============================================================

print("\nFinding true reference records...")

true_ids = set()

for value in ground_truth["matched_entity_ids"]:

    if not value:
        continue

    for entity_id in str(value).split(","):

        entity_id = entity_id.strip()

        if entity_id:

            true_ids.add(entity_id)


print(
    "True reference IDs:",
    len(true_ids)
)


# ============================================================
# LOAD TRUE RECORDS
# ============================================================

def load_true_records(path, true_ids):

    records = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        chunk = chunk.fillna("")

        matched = chunk[
            chunk["entity_id"].isin(true_ids)
        ]

        for row in matched.itertuples(index=False):

            records.append(
                normalize_record(
                    row.entity_id,
                    row.business_name,
                    row.business_address,
                    row.country
                )
            )

    return records


print("\nLoading true Source 2 records...")

source2_records = load_true_records(
    source2_path,
    true_ids
)

print(
    "Loaded Source 2:",
    len(source2_records)
)


print("\nLoading true Source 3 records...")

source3_records = load_true_records(
    source3_path,
    true_ids
)

print(
    "Loaded Source 3:",
    len(source3_records)
)


# ============================================================
# BUILD INDEX
# ============================================================

print("\nBuilding V5.1 block index...")

reference_records = (
    source2_records +
    source3_records
)

block_index = build_block_index(
    reference_records
)

print(
    "Number of blocking keys:",
    len(block_index)
)


# ============================================================
# EVALUATE
# ============================================================

print("\nEvaluating V5.1...\n")

total_true_matches = 0
total_found_matches = 0

s1_with_matches = 0
s1_all_found = 0

candidate_counts = []

missed_examples = []


for row in source1.itertuples(index=False):

    s1_id = row.entity_id

    gt_row = ground_truth[
        ground_truth["source1_entity_id"] == s1_id
    ]

    if gt_row.empty:
        continue

    matched_ids_text = gt_row.iloc[0][
        "matched_entity_ids"
    ]

    if not matched_ids_text:
        continue

    true_match_ids = {
        x.strip()
        for x in str(matched_ids_text).split(",")
        if x.strip()
    }

    if not true_match_ids:
        continue

    s1_with_matches += 1

    total_true_matches += len(true_match_ids)

    record = normalize_record(
        row.entity_id,
        row.business_name,
        row.business_address,
        row.country
    )

    candidates = get_candidates(
        record,
        block_index
    )

    candidate_counts.append(
        len(candidates)
    )

    found = true_match_ids.intersection(
        candidates
    )

    total_found_matches += len(found)

    if found == true_match_ids:

        s1_all_found += 1

    else:

        missed = true_match_ids - candidates

        if len(missed_examples) < 20:

            missed_examples.append(
                (
                    s1_id,
                    missed,
                    record
                )
            )


# ============================================================
# RESULTS
# ============================================================

recall = (
    total_found_matches / total_true_matches
    if total_true_matches
    else 0
)

avg_candidates = (
    sum(candidate_counts) / len(candidate_counts)
    if candidate_counts
    else 0
)

max_candidates = (
    max(candidate_counts)
    if candidate_counts
    else 0
)


print("=" * 60)

print("V5.1 BLOCKING RESULTS")

print("=" * 60)

print(
    "S1 records with matches:",
    s1_with_matches
)

print(
    "Total true matches:",
    total_true_matches
)

print(
    "True matches found:",
    total_found_matches
)

print(
    "Missed true matches:",
    total_true_matches - total_found_matches
)

print(
    "Recall:",
    f"{recall * 100:.4f}%"
)

print(
    "S1 with ALL matches found:",
    s1_all_found,
    "/",
    s1_with_matches
)

print(
    "Average candidates:",
    f"{avg_candidates:.2f}"
)

print(
    "Maximum candidates:",
    max_candidates
)

print("=" * 60)


# ============================================================
# MISSES
# ============================================================

if missed_examples:

    print("\nMISSED EXAMPLES:")

    for s1_id, missed, record in missed_examples:

        print("\nS1:", s1_id)

        print(
            "Name:",
            record["business_name"]
        )

        print(
            "Address:",
            record["business_address"]
        )

        print(
            "Country:",
            record["country"]
        )

        print(
            "Missed IDs:",
            list(missed)
        )

else:

    print(
        "\nNO MISSED MATCHES IN SAMPLE! 🔥🔥🔥"
    )


print("\nDONE.")