import os
import sys
import pandas as pd

# Allow imports from src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.normalization import normalize_record
from src.blocking import build_block_index, get_candidates


def load_source_file(path):
    """
    Load a TSV source file and normalize every record.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)

    records = []

    for _, row in df.iterrows():
        record = normalize_record(
            row["entity_id"],
            row["business_name"],
            row["business_address"],
            row["country"],
        )
        records.append(record)

    return records


def build_candidate_index(source2_records, source3_records):
    """
    Build blocking index using ONLY Source 2 and Source 3.

    Source 1 records must never be inserted into this index.
    """
    reference_records = source2_records + source3_records

    print(f"Building block index for {len(reference_records):,} records...")

    block_index = build_block_index(reference_records)

    print(f"Number of blocking keys: {len(block_index):,}")

    return block_index


def generate_candidates(source1_records, block_index):
    """
    Generate candidate entity IDs for every Source 1 record.
    """
    results = []

    total = len(source1_records)

    for i, record in enumerate(source1_records, start=1):

        candidates = get_candidates(record, block_index)

        results.append({
            "source1_entity_id": record["entity_id"],
            "candidate_entity_ids": ",".join(sorted(candidates)),
        })

        if i % 1000 == 0 or i == total:
            print(f"Processed {i:,}/{total:,} Source 1 records")

    return pd.DataFrame(results)


def main():

    # ---------------------------------------------------------
    # File locations
    # ---------------------------------------------------------

    train_dir = os.path.join(PROJECT_ROOT, "datasets", "train")

    source1_path = os.path.join(train_dir, "train_source1.tsv")
    source2_path = os.path.join(train_dir, "train_source2.tsv")
    source3_path = os.path.join(train_dir, "train_source3.tsv")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("\nLoading Source 1...")
    source1_records = load_source_file(source1_path)

    print("Loading Source 2...")
    source2_records = load_source_file(source2_path)

    print("Loading Source 3...")
    source3_records = load_source_file(source3_path)

    print("\nData loaded successfully!")
    print(f"Source 1: {len(source1_records):,}")
    print(f"Source 2: {len(source2_records):,}")
    print(f"Source 3: {len(source3_records):,}")

    # ---------------------------------------------------------
    # Build blocking index
    # ---------------------------------------------------------

    block_index = build_candidate_index(
        source2_records,
        source3_records
    )

    # ---------------------------------------------------------
    # Generate candidates
    # ---------------------------------------------------------

    print("\nGenerating candidates...")

    candidate_df = generate_candidates(
        source1_records,
        block_index
    )

    # ---------------------------------------------------------
    # Save output
    # ---------------------------------------------------------

    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "candidate_pairs_sample.tsv"
    )

    candidate_df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    print("\nDONE!")
    print(f"Saved candidate file to:")
    print(output_path)


if __name__ == "__main__":
    main()