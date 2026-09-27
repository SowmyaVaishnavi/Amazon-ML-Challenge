"""
Validate matching_results.tsv and candidate_pairs.tsv against every rule
in the challenge brief, before spending a real leaderboard submission on
a format rejection.

Checked rules:
  matching_results.tsv:
    1.  Exactly one row per Source 1 entity in the given test/val directory.
    2.  No duplicate source1_entity_id rows.
    3.  matched_entity_ids only reference IDs that exist in source2/source3
        for this test/val set (no self-matches to Source 1, no unknown IDs).
    4.  No duplicate entity IDs within a single row's ID list.
    5.  Tab-separated, correct column names/order.

  candidate_pairs.tsv (when provided):
    6.  Same structural rules (1, 2, 4, 5) applied to candidate_entity_ids.
    7.  Every ID that appears in matching_results.tsv's matched list for a
        given Source 1 entity must also appear in that same entity's
        candidate list (a matched ID that was never a candidate = pipeline
        bug per the brief).

Usage mirrors the brief's own utils/validate_submission.py invocation:

    python3 validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

--test-dir must contain test_source1.tsv / test_source2.tsv / test_source3.tsv
(or val_source1.tsv / val_source2.tsv / val_source3.tsv - both prefixes are
auto-detected, so the exact same script validates against the real test set
or a local val split).

Prints PASS (exit 0) or a numbered list of issues (exit 1). Never computes a
score - that's scorer.py's job.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.io import parse_id_list  # noqa: E402


def _find_source_file(test_dir: str, source_num: int) -> str:
    for prefix in ("test", "val", "train"):
        candidates = glob.glob(os.path.join(test_dir, f"{prefix}_source{source_num}.tsv"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(
        f"Could not find a *_source{source_num}.tsv file under {test_dir} "
        f"(looked for test_/val_/train_ prefixes)"
    )


def _load_valid_ids(test_dir: str) -> tuple[set[str], set[str], set[str]]:
    s1_path = _find_source_file(test_dir, 1)
    s2_path = _find_source_file(test_dir, 2)
    s3_path = _find_source_file(test_dir, 3)
    s1_ids = set(pd.read_csv(s1_path, sep="\t", dtype=str, usecols=["entity_id"])["entity_id"])
    s2_ids = set(pd.read_csv(s2_path, sep="\t", dtype=str, usecols=["entity_id"])["entity_id"])
    s3_ids = set(pd.read_csv(s3_path, sep="\t", dtype=str, usecols=["entity_id"])["entity_id"])
    return s1_ids, s2_ids, s3_ids


def _validate_one_file(
    path: str,
    id_col: str,
    s1_ids: set[str],
    s23_ids: set[str],
    file_label: str,
) -> tuple[list[str], dict[str, list[str]]]:
    issues = []
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        return [f"[{file_label}] could not read file as TSV: {e}"], {}

    expected_cols = ["source1_entity_id", id_col]
    if list(df.columns) != expected_cols:
        issues.append(
            f"[{file_label}] column mismatch: expected {expected_cols}, got {list(df.columns)}"
        )
        if not {"source1_entity_id", id_col}.issubset(df.columns):
            return issues, {}

    present_ids = set(df["source1_entity_id"])
    missing = s1_ids - present_ids
    if missing:
        sample = list(missing)[:5]
        issues.append(
            f"[{file_label}] {len(missing):,} Source 1 test entities have NO row "
            f"(every test entity must appear). Sample missing: {sample}"
        )
    unknown = present_ids - s1_ids
    if unknown:
        sample = list(unknown)[:5]
        issues.append(
            f"[{file_label}] {len(unknown):,} rows reference source1_entity_id values "
            f"not present in the test set. Sample: {sample}"
        )

    dup_rows = df["source1_entity_id"][df["source1_entity_id"].duplicated()]
    if len(dup_rows):
        issues.append(
            f"[{file_label}] {len(dup_rows):,} duplicate source1_entity_id rows. "
            f"Sample: {dup_rows.tolist()[:5]}"
        )

    id_lists: dict[str, list[str]] = {}
    n_dup_within_row = 0
    n_bad_ref = 0
    bad_ref_samples = []
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        raw = parse_id_list(row[id_col])
        id_lists[s1_id] = raw
        if len(raw) != len(set(raw)):
            n_dup_within_row += 1
        bad_refs = [i for i in raw if i not in s23_ids]
        if bad_refs:
            n_bad_ref += 1
            if len(bad_ref_samples) < 5:
                bad_ref_samples.append((s1_id, bad_refs[:3]))

    if n_dup_within_row:
        issues.append(
            f"[{file_label}] {n_dup_within_row:,} rows contain duplicate entity IDs "
            f"within their own {id_col} list."
        )
    if n_bad_ref:
        issues.append(
            f"[{file_label}] {n_bad_ref:,} rows reference IDs that are not valid "
            f"Source 2/3 test entities (or are self-references to Source 1). "
            f"Sample (s1_id, bad_ids): {bad_ref_samples}"
        )

    return issues, id_lists


def validate(matching_path: str, test_dir: str, candidate_path: str | None = None) -> tuple[bool, list[str]]:
    s1_ids, s2_ids, s3_ids = _load_valid_ids(test_dir)
    s23_ids = s2_ids | s3_ids

    all_issues, match_lists = _validate_one_file(
        matching_path, "matched_entity_ids", s1_ids, s23_ids, "matching_results.tsv"
    )

    if candidate_path:
        cand_issues, cand_lists = _validate_one_file(
            candidate_path, "candidate_entity_ids", s1_ids, s23_ids, "candidate_pairs.tsv"
        )
        all_issues.extend(cand_issues)

        n_not_in_candidates = 0
        sample = []
        for s1_id, matched in match_lists.items():
            candidates = set(cand_lists.get(s1_id, []))
            leaked = [m for m in matched if m not in candidates]
            if leaked:
                n_not_in_candidates += 1
                if len(sample) < 5:
                    sample.append((s1_id, leaked[:3]))
        if n_not_in_candidates:
            all_issues.append(
                f"[cross-file] {n_not_in_candidates:,} Source 1 entities have a matched "
                f"ID that never appeared in their own candidate_pairs.tsv row - this "
                f"signals a pipeline bug (final matches must be a subset of candidates). "
                f"Sample (s1_id, leaked_ids): {sample}"
            )

    return len(all_issues) == 0, all_issues


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    ap.add_argument("--candidate", default=None, help="Path to candidate_pairs.tsv (optional but recommended)")
    ap.add_argument("--test-dir", required=True, help="Directory with *_source{1,2,3}.tsv for this split")
    args = ap.parse_args()

    ok, issues = validate(args.matching, args.test_dir, args.candidate)

    if ok:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL - {len(issues)} issue(s):")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
