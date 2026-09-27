import os
import sys

# Allow imports from src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import normalize_record
from src.blocking import build_block_index, get_candidates


# ---------------------------------------------------------
# Small fake dataset
# ---------------------------------------------------------

source2_records = [
    normalize_record(
        "S2-001",
        "Prime Money Corp.",
        "17560 Ellis Road, Tahlequah, OK",
        "US"
    ),
    normalize_record(
        "S2-002",
        "Green Valley Foods",
        "123 Main Street, Dallas, TX",
        "US"
    ),
]

source3_records = [
    normalize_record(
        "S3-001",
        "Prime Money Corporation",
        "17560 Ellis Rd, Tahlequah, OK",
        "US"
    ),
    normalize_record(
        "S3-002",
        "Blue Star Cafe",
        "55 Park Road, Austin, TX",
        "US"
    ),
]


source1_records = [
    normalize_record(
        "S1-001",
        "Prime Money",
        "17560 Ellis Rd, Tahlequah, OK",
        "US"
    )
]


# ---------------------------------------------------------
# Build index using ONLY Source 2 + Source 3
# ---------------------------------------------------------

reference_records = source2_records + source3_records

print("Building block index...")

block_index = build_block_index(reference_records)

print(f"Number of blocking keys: {len(block_index)}")


# ---------------------------------------------------------
# Generate candidates for Source 1
# ---------------------------------------------------------

print("\nFinding candidates for S1-001...")

candidates = get_candidates(
    source1_records[0],
    block_index
)


# ---------------------------------------------------------
# Show results
# ---------------------------------------------------------

print("\nCandidates found:")

for candidate in sorted(candidates):
    print(candidate)


# ---------------------------------------------------------
# Basic checks
# ---------------------------------------------------------

assert "S2-001" in candidates
assert "S3-001" in candidates

print("\n✅ S2-001 found")
print("✅ S3-001 found")
print("✅ Candidate generation test PASSED!")