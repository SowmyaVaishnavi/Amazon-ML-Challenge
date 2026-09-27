import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.normalization import normalize_record
from src.blocking import (
    make_name_ngrams,
    make_phonetic_key,
    make_blocking_keys,
    build_block_index,
    get_candidates,
)


records = [
    normalize_record(
        "S1-TEST-1",
        "Prime Money Corp.",
        "17560 Ellis Road, Tahlequah, OK",
        "US",
    ),
    normalize_record(
        "S2-TEST-1",
        "Prime Money Corporation",
        "17560 Ellis Rd, Tahlequah, OK",
        "US",
    ),
    normalize_record(
        "S2-TEST-2",
        "Completely Different Business",
        "100 Main Street, Dallas, TX",
        "US",
    ),
]


print("=" * 70)
print("BLOCKING TEST")
print("=" * 70)

for record in records:
    print("\nENTITY:", record["entity_id"])
    print("NAME:", record["business_name"])

    print("\nName n-grams:")
    print(make_name_ngrams(record["business_name"]))

    print("\nPhonetic key:")
    print(make_phonetic_key(record["business_name"]))

    print("\nBlocking keys:")
    keys = make_blocking_keys(record)

    for key in sorted(keys):
        print(" ", key)


print("\n" + "=" * 70)
print("BUILDING BLOCK INDEX")
print("=" * 70)

index = build_block_index(records)

print("Number of blocking keys:", len(index))

print("\n" + "=" * 70)
print("CANDIDATES FOR S1-TEST-1")
print("=" * 70)

candidates = get_candidates(records[0], index)

print(candidates)