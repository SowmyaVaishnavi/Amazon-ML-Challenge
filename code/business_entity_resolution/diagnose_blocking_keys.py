from pathlib import Path
from collections import defaultdict
import pandas as pd
import sys


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"

S1_PATH = TRAIN_DIR / "train_source1.tsv"
S2_PATH = TRAIN_DIR / "train_source2.tsv"
S3_PATH = TRAIN_DIR / "train_source3.tsv"
GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

sys.path.insert(0, str(PROJECT_ROOT))

from src.normalization import normalize_record


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 1000


# ============================================================
# V5.2 BLOCKING KEYS BY TYPE
#
# Changes vs V5.1, based on measured % of the 10.32M reference pool
# each block type was matching:
#
#   CUT ENTIRELY:
#     - namegram   (was 81.61% of the whole pool -- ~no filtering value)
#     - addrshort  (was 59.87% -- same problem)
#
#   TIGHTENED (thresholds raised so they discriminate more):
#     - addrword4/5 (was 45.31% / 34.12%) -> addrword6/7
#     - nameword4/5 (was 18.03% / 16.94%) -> nameword5/6
#     - nameprefix2 (was 16.82%) -> DROPPED, kept only nameprefix3/4
#
#   KEPT UNCHANGED (already tight):
#     - name2prefix (0.03%), fulladdress (0.00%), nameshort (7.88%),
#       nameprefix3 (3.97%), nameprefix4 (2.02%)
# ============================================================

def make_blocking_keys_by_type(record):

    name = record["business_name"]
    address = record["business_address"]
    country = record["country"]

    blocks = defaultdict(set)

    name_tokens = name.split()

    # --------------------------------------------------------
    # NAME PREFIX 3 / 4   (dropped n=2)
    # --------------------------------------------------------

    if country and name_tokens:

        first_word = name_tokens[0]

        for n in (3, 4):

            if len(first_word) >= n:

                blocks[f"nameprefix{n}"].add(
                    f"country|{country}|nameprefix{n}|"
                    f"{first_word[:n]}"
                )

    # --------------------------------------------------------
    # TWO WORD PREFIX
    # --------------------------------------------------------

    if country and len(name_tokens) >= 2:

        w1 = name_tokens[0]
        w2 = name_tokens[1]

        if len(w1) >= 3 and len(w2) >= 3:

            blocks["name2prefix"].add(
                f"country|{country}|name2prefix|"
                f"{w1[:3]}|{w2[:3]}"
            )

    # --------------------------------------------------------
    # NAME WORD PREFIX 5 / 6   (raised from 4/5)
    # --------------------------------------------------------

    if country:

        for word in name_tokens:

            if len(word) >= 5:

                for n in (5, 6):

                    if len(word) >= n:

                        blocks[f"nameword{n}"].add(
                            f"country|{country}|nameword{n}|"
                            f"{word[:n]}"
                        )

    # --------------------------------------------------------
    # SHORT NAME
    # --------------------------------------------------------

    if country:

        for word in name_tokens:

            if 1 <= len(word) <= 3:

                blocks["nameshort"].add(
                    f"country|{country}|nameshort|"
                    f"{word}"
                )

    # --------------------------------------------------------
    # FULL ADDRESS
    # --------------------------------------------------------

    if country and address:

        blocks["fulladdress"].add(
            f"country|{country}|address|"
            f"{address}"
        )

    # --------------------------------------------------------
    # ADDRESS WORD 6 / 7   (raised from 4/5)
    # --------------------------------------------------------

    if country:

        for word in address.split():

            if len(word) >= 6:

                blocks["addrword6"].add(
                    f"country|{country}|addrword6|"
                    f"{word[:6]}"
                )

            if len(word) >= 7:

                blocks["addrword7"].add(
                    f"country|{country}|addrword7|"
                    f"{word[:7]}"
                )

    # REMOVED: namegram, addrshort

    return blocks


# ============================================================
# LOAD S1
# ============================================================

print("=" * 80)
print("TRUE-MATCH BLOCK COVERAGE DIAGNOSTIC (V5.2)")
print("=" * 80)

print()
print(f"Using FIRST {SAMPLE_SIZE} S1 records")
print()


s1_df = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    nrows=SAMPLE_SIZE
).fillna("")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

gt_df = pd.read_csv(
    GT_PATH,
    sep="\t",
    dtype=str
).fillna("")


gt_df = gt_df[
    gt_df["source1_entity_id"].isin(
        s1_df["entity_id"]
    )
]


# ============================================================
# BUILD TRUE MATCH SET
# ============================================================

true_matches = {}

for row in gt_df.itertuples(index=False):

    s1_id = row.source1_entity_id

    matched = row.matched_entity_ids

    if not matched or matched == "nan":

        true_matches[s1_id] = set()

        continue

    ids = {
        x.strip()
        for x in matched.split(",")
        if x.strip()
    }

    true_matches[s1_id] = ids


# ============================================================
# CREATE S1 BLOCKS
# ============================================================

s1_blocks = {}

for row in s1_df.itertuples(index=False):

    record = normalize_record(
        row.entity_id,
        row.business_name,
        row.business_address,
        row.country
    )

    s1_blocks[
        row.entity_id
    ] = make_blocking_keys_by_type(record)


# ============================================================
# LOAD ONLY TRUE REFERENCE RECORDS
# ============================================================

all_true_ids = set()

for ids in true_matches.values():

    all_true_ids.update(ids)


print(
    f"Total unique true reference IDs: "
    f"{len(all_true_ids):,}"
)

print()


# ============================================================
# COVERAGE COUNTERS
# ============================================================

total_true_pairs = 0

covered_by_type = defaultdict(int)

covered_by_any = 0

not_covered = []


# ============================================================
# SCAN S2 + S3
# ============================================================

def scan_reference(path, source_name):

    global total_true_pairs
    global covered_by_any

    print(
        f"Scanning {source_name}..."
    )

    df_iter = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=200_000
    )

    for chunk in df_iter:

        chunk = chunk.fillna("")

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            if entity_id not in all_true_ids:

                continue

            record = normalize_record(
                row.entity_id,
                row.business_name,
                row.business_address,
                row.country
            )

            reference_blocks = (
                make_blocking_keys_by_type(
                    record
                )
            )

            # ------------------------------------------------
            # Find which S1 owns this true match
            # ------------------------------------------------

            owning_s1s = []

            for s1_id, ids in true_matches.items():

                if entity_id in ids:

                    owning_s1s.append(
                        s1_id
                    )

            for s1_id in owning_s1s:

                total_true_pairs += 1

                matched_any = False

                for block_type, ref_keys in (
                    reference_blocks.items()
                ):

                    s1_keys = s1_blocks[
                        s1_id
                    ].get(
                        block_type,
                        set()
                    )

                    if ref_keys & s1_keys:

                        covered_by_type[
                            block_type
                        ] += 1

                        matched_any = True

                if matched_any:

                    covered_by_any += 1

                else:

                    not_covered.append(
                        (
                            s1_id,
                            entity_id
                        )
                    )


# ============================================================
# RUN
# ============================================================

scan_reference(
    S2_PATH,
    "S2"
)

scan_reference(
    S3_PATH,
    "S3"
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 80)
print("TRUE-MATCH BLOCK COVERAGE")
print("=" * 80)

print()

print(
    f"Total true pairs checked: "
    f"{total_true_pairs:,}"
)

print(
    f"Covered by at least one V5.2 block: "
    f"{covered_by_any:,}"
)

if total_true_pairs:

    overall_recall = (
        covered_by_any
        / total_true_pairs
        * 100
    )

    print(
        f"Overall block recall: "
        f"{overall_recall:.4f}%"
    )


print()

print(
    f"{'BLOCK TYPE':<20}"
    f"{'TRUE PAIRS':>15}"
    f"{'COVERAGE %':>15}"
)

print("-" * 55)


for block_type in sorted(
    covered_by_type.keys()
):

    count = covered_by_type[
        block_type
    ]

    percentage = (
        count
        / total_true_pairs
        * 100
        if total_true_pairs
        else 0
    )

    print(
        f"{block_type:<20}"
        f"{count:>15,}"
        f"{percentage:>14.2f}%"
    )


# ============================================================
# MISSES
# ============================================================

print()
print("=" * 80)
print("TRUE MATCHES MISSED BY ALL BLOCKS")
print("=" * 80)

if not not_covered:

    print(
        "NONE -- every checked true pair "
        "is covered by at least one block."
    )

else:

    print(
        f"Missed true pairs: "
        f"{len(not_covered):,}"
    )

    print()

    for s1_id, entity_id in (
        not_covered[:20]
    ):

        print(
            f"{s1_id} -> {entity_id}"
        )


print()
print("=" * 80)
print("DONE")
print("=" * 80)