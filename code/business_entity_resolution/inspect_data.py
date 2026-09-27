import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

files = [
    BASE / "datasets" / "train" / "train_source1.tsv",
    BASE / "datasets" / "train" / "train_source2.tsv",
    BASE / "datasets" / "train" / "train_source3.tsv",
    BASE / "datasets" / "train" / "train_ground_truth.tsv",
]

for file in files:
    print("\n" + "=" * 70)
    print(f"FILE: {file.name}")
    print("=" * 70)

    df = pd.read_csv(file, sep="\t")

    print("Columns:")
    print(list(df.columns))

    print(f"\nRows: {len(df)}")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nMissing values:")
    print(df.isna().sum())
    print("\n" + "#" * 70)
print("TEST FILE CHECK")
print("#" * 70)

test_files = [
    BASE / "datasets" / "test" / "test_source1.tsv",
    BASE / "datasets" / "test" / "test_source2.tsv",
    BASE / "datasets" / "test" / "test_source3.tsv",
]

for file in test_files:
    print("\n" + "=" * 70)
    print(f"FILE: {file.name}")
    print("=" * 70)

    df = pd.read_csv(file, sep="\t", nrows=3)

    print("Columns:")
    print(list(df.columns))

    print("\nFirst 3 rows:")
    print(df.to_string(index=False))

    print("\nData types:")
    print(df.dtypes)