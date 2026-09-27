"""
Turn whatever shape your predictions are in into a correctly formatted
matching_results.tsv or candidate_pairs.tsv - so nobody needs to hand-roll
pandas.to_csv(sep="\\t", ...) calls and risk a subtle formatting rejection.

Accepts THREE input shapes, auto-detected:

  1. dict[str, list[str]]           {"S1-001": ["S2-047", "S3-812"], ...}
  2. pandas.DataFrame (long)        columns: source1_entity_id, matched_id
                                     (one row per pair; will be grouped)
  3. pandas.DataFrame (wide)        columns: source1_entity_id, matched_entity_ids
                                     (already one row per S1 entity, ids may be
                                     a list OR an already-comma-joined string)

In every case:
  - every s1_entity_id in `all_s1_ids` gets exactly one output row, even if it
    has zero predictions (empty string - required so singletons aren't
    silently dropped)
  - IDs are deduped per row (order-preserving)
  - column names/order match the brief exactly
  - written tab-separated, no index, no quoting surprises

Works identically for matching_results.tsv (id_col="matched_entity_ids") and
candidate_pairs.tsv (id_col="candidate_entity_ids") - just pass the right
id_col and out_path.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, Mapping, Union

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.io import format_id_list  # noqa: E402

PredictionsInput = Union[
    Mapping[str, Iterable[str]],
    pd.DataFrame,
]


def _normalize_to_dict(predictions: PredictionsInput, id_col: str) -> dict[str, list[str]]:
    if isinstance(predictions, Mapping):
        return {str(k): list(v) for k, v in predictions.items()}

    if isinstance(predictions, pd.DataFrame):
        cols = set(predictions.columns)

        if "source1_entity_id" not in cols:
            raise ValueError(
                f"DataFrame input must have a 'source1_entity_id' column, got {list(predictions.columns)}"
            )

        if id_col in cols:
            out = {}
            for _, row in predictions.iterrows():
                val = row[id_col]
                if isinstance(val, list):
                    ids = val
                elif pd.isna(val) or val == "":
                    ids = []
                else:
                    ids = [x for x in str(val).split(",") if x != ""]
                out[row["source1_entity_id"]] = ids
            return out

        # long format: one (s1_id, matched_id) pair per row
        id_value_col = None
        for candidate_col in ("matched_id", "candidate_id", "entity_id", "id"):
            if candidate_col in cols:
                id_value_col = candidate_col
                break
        if id_value_col is None:
            raise ValueError(
                f"Long-format DataFrame needs a per-pair ID column "
                f"(one of matched_id/candidate_id/entity_id/id), got {list(predictions.columns)}"
            )
        grouped = predictions.groupby("source1_entity_id")[id_value_col].apply(list)
        return grouped.to_dict()

    raise TypeError(f"Unsupported predictions type: {type(predictions)}")


def write_submission_file(
    predictions: PredictionsInput,
    all_s1_ids: Iterable[str],
    out_path: str,
    id_col: str = "matched_entity_ids",
) -> pd.DataFrame:
    """
    predictions : dict or DataFrame, see module docstring for accepted shapes
    all_s1_ids  : every Source 1 entity_id that MUST appear in the output
                  (pass test_source1["entity_id"], not just the ones you
                  found predictions for - this is how singletons get their
                  required empty row)
    out_path    : where to write the .tsv
    id_col      : "matched_entity_ids" for matching_results.tsv,
                  "candidate_entity_ids" for candidate_pairs.tsv
    """
    pred_dict = _normalize_to_dict(predictions, id_col)

    rows = []
    for s1_id in all_s1_ids:
        ids = pred_dict.get(s1_id, [])
        rows.append({"source1_entity_id": s1_id, id_col: format_id_list(ids)})

    out_df = pd.DataFrame(rows, columns=["source1_entity_id", id_col])

    if out_df["source1_entity_id"].duplicated().any():
        raise ValueError(
            "all_s1_ids contained duplicates - each Source 1 entity must "
            "produce exactly one output row."
        )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)
    return out_df


def write_matching_results(predictions: PredictionsInput, all_s1_ids: Iterable[str], out_path: str) -> pd.DataFrame:
    return write_submission_file(predictions, all_s1_ids, out_path, id_col="matched_entity_ids")


def write_candidate_pairs(predictions: PredictionsInput, all_s1_ids: Iterable[str], out_path: str) -> pd.DataFrame:
    return write_submission_file(predictions, all_s1_ids, out_path, id_col="candidate_entity_ids")


if __name__ == "__main__":
    # Tiny smoke test / usage example
    demo_preds = {
        "S1-00001": ["S2-00047", "S2-00193", "S3-00812"],
        "S1-00002": ["S3-00004"],
        # S1-00003 deliberately absent -> should still get an empty row
    }
    demo_all_ids = ["S1-00001", "S1-00002", "S1-00003"]
    df = write_submission_file(demo_preds, demo_all_ids, "/tmp/demo_matching_results.tsv")
    print(df.to_string(index=False))
