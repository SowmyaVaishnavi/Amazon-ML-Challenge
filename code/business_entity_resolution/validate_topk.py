from pathlib import Path
from collections import defaultdict
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"


# ============================================================
# VALIDATION SETTINGS
# ============================================================

# IMPORTANT:
# We are testing ONLY the same 1000 S1 records
# that we have been using since morning.

SAMPLE_SIZE = 1000

# Top-K values to test
K_VALUES = [10, 15, 20, 30, 40, 50]

# Used only for reading the large S2/S3 files
CHUNK_SIZE = 200_000


# ============================================================
# V5.1 BLOCKING KEYS
# ============================================================

def make_blocking_keys(record):

    name = record["business_name"]
    address = record["business_address"]
    country = record["country"]

    keys = set()

    # --------------------------------------------------------
    # NAME TOKENS
    # --------------------------------------------------------

    name_tokens = name.split()

    # --------------------------------------------------------
    # FIRST WORD PREFIX
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
    # FIRST TWO WORD PREFIX
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
    # NAME TRIGRAMS
    # --------------------------------------------------------

    for token in name_tokens:

        if len(token) >= 3:

            for i in range(len(token) - 2):

                trigram = token[i:i + 3]

                keys.add(
                    f"namegram:{country}:{trigram}"
                )

    # --------------------------------------------------------
    # NAME WORD PREFIX
    # --------------------------------------------------------

    for token in name_tokens:

        if len(token) >= 4:

            keys.add(
                f"nameword:{country}:{token[:4]}"
            )

    # --------------------------------------------------------
    # SHORT NAME TOKENS
    # --------------------------------------------------------

    for token in name_tokens:

        if 1 <= len(token) <= 3:

            keys.add(
                f"shortname:{country}:{token}"
            )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    address_tokens = address.split()

    # Full normalized address
    if address:

        keys.add(
            f"address:{country}:{address}"
        )

    # --------------------------------------------------------
    # ADDRESS WORD PREFIXES
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
    # ADDRESS SHORT WORDS
    # --------------------------------------------------------

    for token in address_tokens:

        if len(token) >= 3:

            keys.add(
                f"addressshort:{country}:{token}"
            )

    return keys


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    return str(text).strip().lower()


def normalize_record(row):

    return {
        "entity_id": str(row["entity_id"]),
        "business_name": normalize_text(
            row["business_name"]
        ),
        "business_address": normalize_text(
            row["business_address"]
        ),
        "country": normalize_text(
            row["country"]
        ),
    }


# ============================================================
# START
# ============================================================

print("=" * 75)
print("V5.1 TOP-K VALIDATION WITH FULL TRAIN REFERENCE POPULATION")
print("=" * 75)

print(
    f"\nIMPORTANT: Using ONLY the first "
    f"{SAMPLE_SIZE} Source 1 records."
)


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("\nLoading Source 1...")

s1_df = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str
).fillna("")

# EXACTLY the same 1000 S1 records
s1_df = s1_df.head(SAMPLE_SIZE).copy()

print(
    f"S1 records loaded: "
    f"{len(s1_df):,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt_df = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
).fillna("")

sample_s1_ids = set(
    s1_df["entity_id"].astype(str)
)

gt_sample = gt_df[
    gt_df["source1_entity_id"]
    .astype(str)
    .isin(sample_s1_ids)
].copy()

print(
    f"Ground-truth rows for sample: "
    f"{len(gt_sample):,}"
)


# ============================================================
# BUILD GROUND TRUTH DICTIONARY
# ============================================================

ground_truth = {}

for _, row in gt_sample.iterrows():

    s1_id = str(
        row["source1_entity_id"]
    )

    matched_ids = set()

    value = row["matched_entity_ids"]

    if value:

        for entity_id in str(value).split(","):

            entity_id = entity_id.strip()

            if entity_id:

                matched_ids.add(entity_id)

    ground_truth[s1_id] = matched_ids


# Total true matches
total_true_matches = sum(
    len(match_set)
    for match_set in ground_truth.values()
)


# Unique true reference IDs
all_true_reference_ids = set()

for match_set in ground_truth.values():

    all_true_reference_ids.update(
        match_set
    )


print(
    f"Unique true S2/S3 reference records: "
    f"{len(all_true_reference_ids):,}"
)

print(
    f"Total true matches: "
    f"{total_true_matches:,}"
)


# ============================================================
# BUILD FULL S2 + S3 BLOCKING INDEX
# ============================================================

print("\n" + "=" * 75)
print("BUILDING FULL S2 + S3 BLOCKING INDEX")
print("=" * 75)

blocking_index = defaultdict(list)

total_reference_records = 0


for source_name, path in [
    ("S2", S2_PATH),
    ("S3", S3_PATH)
]:

    print(
        f"\nScanning FULL {source_name}..."
    )

    chunk_number = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        chunk = chunk.fillna("")

        chunk_number += 1

        # ----------------------------------------------------
        # IMPORTANT:
        # Use iterrows here because normalize_record()
        # expects row["column_name"] access.
        # ----------------------------------------------------

        for _, row in chunk.iterrows():

            record = normalize_record(
                row
            )

            entity_id = record[
                "entity_id"
            ]

            keys = make_blocking_keys(
                record
            )

            for key in keys:

                blocking_index[key].append(
                    entity_id
                )

            total_reference_records += 1

        print(
            f"{source_name} chunk "
            f"{chunk_number} processed | "
            f"reference records indexed: "
            f"{total_reference_records:,}"
        )


print(
    f"\nFULL S2 + S3 reference records indexed: "
    f"{total_reference_records:,}"
)

print(
    f"Total blocking keys: "
    f"{len(blocking_index):,}"
)


# ============================================================
# GENERATE CANDIDATES FOR ONLY 1000 S1
# ============================================================

print("\n" + "=" * 75)
print("GENERATING V5.1 CANDIDATES")
print("=" * 75)

results = []

pre_k_total_candidates = 0
pre_k_true_found = 0
pre_k_all_found = 0


for counter, (_, row) in enumerate(
    s1_df.iterrows(),
    start=1
):

    record = normalize_record(
        row
    )

    s1_id = record[
        "entity_id"
    ]

    # --------------------------------------------------------
    # Generate S1 blocking keys
    # --------------------------------------------------------

    query_keys = make_blocking_keys(
        record
    )

    # --------------------------------------------------------
    # Blocking-hit score
    #
    # score = number of blocking keys shared
    # between S1 and candidate
    # --------------------------------------------------------

    candidate_scores = defaultdict(int)

    for key in query_keys:

        matching_entities = (
            blocking_index.get(
                key,
                []
            )
        )

        for candidate_id in matching_entities:

            candidate_scores[
                candidate_id
            ] += 1

    # --------------------------------------------------------
    # Rank candidates
    #
    # 1. Higher blocking-hit score first
    # 2. Entity ID as deterministic tie-break
    # --------------------------------------------------------

    ranked_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: (-x[1], x[0])
    )

    candidate_ids = [
        candidate_id
        for candidate_id, score
        in ranked_candidates
    ]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    true_ids = ground_truth.get(
        s1_id,
        set()
    )

    # --------------------------------------------------------
    # Check how many true matches are found
    # BEFORE Top-K
    # --------------------------------------------------------

    found = true_ids.intersection(
        candidate_ids
    )

    pre_k_total_candidates += len(
        candidate_ids
    )

    pre_k_true_found += len(
        found
    )

    if len(found) == len(true_ids):

        pre_k_all_found += 1

    results.append({
        "s1_id": s1_id,
        "ranked_candidates": ranked_candidates,
        "true_ids": true_ids
    })

    # Progress
    if counter % 100 == 0:

        print(
            f"Processed S1: "
            f"{counter:,}/{SAMPLE_SIZE:,}"
        )


# ============================================================
# PRE-TOP-K RESULTS
# ============================================================

pre_k_average = (
    pre_k_total_candidates
    / SAMPLE_SIZE
)

pre_k_recall = (
    pre_k_true_found
    / total_true_matches
    if total_true_matches > 0
    else 1.0
)


print("\n" + "=" * 75)
print("V5.1 BEFORE TOP-K")
print("=" * 75)

print(
    f"Average candidates/entity: "
    f"{pre_k_average:.2f}"
)

print(
    f"True matches found: "
    f"{pre_k_true_found:,}/"
    f"{total_true_matches:,}"
)

print(
    f"Recall: "
    f"{pre_k_recall * 100:.4f}%"
)

print(
    f"S1 with all true matches found: "
    f"{pre_k_all_found:,}/"
    f"{SAMPLE_SIZE:,}"
)


# ============================================================
# TOP-K SWEEP
# ============================================================

print("\n" + "=" * 75)
print("TOP-K EXPERIMENT")
print("=" * 75)

print(
    "\nK\tAvg Candidates\tRecall\t\t"
    "True Found\tS1 All Found"
)

print("-" * 75)


for k in K_VALUES:

    total_kept = 0

    true_found = 0

    all_found_s1 = 0

    # --------------------------------------------------------
    # Evaluate every one of the 1000 S1 records
    # --------------------------------------------------------

    for result in results:

        ranked_candidates = (
            result["ranked_candidates"]
        )

        true_ids = result["true_ids"]

        # Keep only top K
        top_k_candidates = {
            candidate_id
            for candidate_id, score
            in ranked_candidates[:k]
        }

        total_kept += len(
            top_k_candidates
        )

        # True matches surviving Top-K
        found = true_ids.intersection(
            top_k_candidates
        )

        true_found += len(
            found
        )

        # Did this S1 retain ALL its true matches?
        if len(found) == len(true_ids):

            all_found_s1 += 1

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    average_candidates = (
        total_kept
        / SAMPLE_SIZE
    )

    recall = (
        true_found
        / total_true_matches
        if total_true_matches > 0
        else 1.0
    )

    print(
        f"{k}\t"
        f"{average_candidates:.2f}\t\t"
        f"{recall * 100:.4f}%\t\t"
        f"{true_found:,}/"
        f"{total_true_matches:,}\t"
        f"{all_found_s1:,}/"
        f"{SAMPLE_SIZE:,}"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)

print(
    "\nValidation scope:"
)

print(
    f"- Source 1: ONLY first {SAMPLE_SIZE:,} records"
)

print(
    "- Source 2: FULL TRAIN"
)

print(
    "- Source 3: FULL TRAIN"
)

print(
    "- Ground truth: ONLY those 1000 S1 records"
)

print(
    "- V5.1 blocking: unchanged"
)

print(
    "- Candidate ranking: blocking-hit score"
)

print(
    "- K values: 10, 15, 20, 30, 40, 50"
)

print(
    "\nThe full 2.2M Source 1 dataset was NOT evaluated."
)

print(
    "Do NOT run full TEST candidate generation yet."
)