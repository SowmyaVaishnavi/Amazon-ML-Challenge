"""
One-command check: validate format, THEN score if valid.

    python3 check_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --ground-truth dataset/val_split/val_ground_truth.tsv \
        --test-dir dataset/val_split

Behavior:
  - Always runs format validation first (same checks as validate_submission.py).
  - If format validation FAILS: prints the issues, exits 1, does NOT attempt to
    score (a malformed file can't be scored meaningfully, and this mirrors the
    real leaderboard - a failed validation never gets a SCORED status).
  - If format validation PASSES: also runs the F0.5 scorer against
    --ground-truth (only possible on a local val split - the real test set's
    ground truth isn't available to you), prints the report, exits 0.
  - --ground-truth is optional: omit it when checking a real test-set
    submission you can't score locally, and this becomes a pure format check.
"""
from __future__ import annotations

import argparse
import sys

from validate_submission import validate
from scorer import score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    ap.add_argument("--candidate", default=None, help="Path to candidate_pairs.tsv (optional but recommended)")
    ap.add_argument("--test-dir", required=True, help="Directory with *_source{1,2,3}.tsv for this split")
    ap.add_argument("--ground-truth", default=None,
                     help="Path to ground truth for local scoring (only available for val splits)")
    args = ap.parse_args()

    print(">>> Step 1/2: format validation")
    ok, issues = validate(args.matching, args.test_dir, args.candidate)
    if not ok:
        print(f"FAIL - {len(issues)} issue(s):")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        print("\n>>> Step 2/2: SKIPPED (fix format issues first)")
        sys.exit(1)
    print("PASS\n")

    if not args.ground_truth:
        print(">>> Step 2/2: SKIPPED (no --ground-truth given - format-only check)")
        sys.exit(0)

    print(">>> Step 2/2: scoring")
    result = score(args.matching, args.ground_truth)
    print("=" * 50)
    print(f"Entities scored:         {result['n_entities']:,}")
    if result["n_missing_from_predictions"]:
        print(f"  WARNING: {result['n_missing_from_predictions']:,} entities missing from predictions")
    print(f"Macro Precision:         {result['macro_precision']:.4f}")
    print(f"Macro Recall:            {result['macro_recall']:.4f}")
    print(f"Macro F0.5:              {result['macro_f0.5']:.4f}")
    print(f"Singletons in GT:        {result['n_singletons']:,}")
    print(f"Singleton accuracy:      {result['singleton_accuracy']:.4f}")
    print("=" * 50)
    sys.exit(0)


if __name__ == "__main__":
    main()
