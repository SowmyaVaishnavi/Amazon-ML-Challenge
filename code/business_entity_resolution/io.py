"""
Shared I/O helpers for the Business Entity Resolution challenge.

Every workstream (A/B/C/D) reads the same four column names off the same
tab-separated files, so that logic lives here once instead of being
copy-pasted (and drifting) across four people's scripts.

All readers force dtype=str: entity IDs like "S1-00001" must never be
coerced to int/float by pandas' type inference, and business_address can
be legitimately empty (NaN) which we want to keep as "".
"""
from __future__ import annotations

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_source(path: str) -> pd.DataFrame:
    """Read a *_source{1,2,3}.tsv file. Always tab-separated, always str dtype."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def read_ground_truth(path: str) -> pd.DataFrame:
    """Read train_ground_truth.tsv (or a val-split ground truth file)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(GT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}")
    return df


def parse_id_list(cell: str) -> list[str]:
    """
    'matched_entity_ids' / 'candidate_entity_ids' cells are comma-separated,
    empty string means "no matches" (singleton). Never returns [''].
    """
    if cell is None or cell == "" or (isinstance(cell, float)):
        return []
    cell = str(cell).strip()
    if cell == "":
        return []
    return [x for x in cell.split(",") if x != ""]


def format_id_list(ids) -> str:
    """Inverse of parse_id_list. Dedupes, preserves first-seen order."""
    seen = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return ",".join(seen)
