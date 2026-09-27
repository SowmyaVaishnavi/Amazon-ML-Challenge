"""
03_predict.py

Person C's inference step: score EVERY candidate pair in B's features.tsv
with the saved model, and hand off a RAW predictions file to Person D.
This script does NOT build matching_results.tsv or candidate_pairs.tsv --
that reshaping into the final submission schema is D's job. This script
only scores and sanity-checks; it does not filter, reshape, or add
singleton rows.

Loads:
    lgbm_matcher.txt     -- saved lgb.Booster (via booster_.save_model)
    best_threshold.tsv   -- tuned threshold from 02_train_matcher.py
    features.tsv         -- B's full candidate pairs (ALL pairs, not just
                            the held-out val split -- inference scores
                            everything)

Writes:
    predictions.tsv -- columns: source1_entity_id, candidate_entity_id,
                        probability, predicted_label
                        One row per INPUT candidate pair. If an S1 entity
                        has candidates but none score >= threshold, its
                        rows still appear here with predicted_label = 0 --
                        this script never drops or omits rows. Whether a
                        "zero positive rows" entity becomes an explicit
                        empty-list singleton in the final output is D's
                        formatting decision, not something decided here.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

DATA_DIR = r"C:\Users\LENOVO\Desktop\AWA hack\dataset"
OUT_DIR = r"C:\Users\LENOVO\Desktop\AWA hack\outputs"

FEATURES_PATH = r"C:\Users\LENOVO\Desktop\AWA hack\features.tsv"   # <-- B's full candidate features (train version for now)
MODEL_PATH = f"{OUT_DIR}/lgbm_matcher.txt"
THRESHOLD_PATH = f"{OUT_DIR}/best_threshold.tsv"
GT_PATH = f"{DATA_DIR}/train/train_ground_truth.tsv"          # only used for the optional sanity spot-check
S1_PATH = f"{DATA_DIR}/train/train_source1.tsv"               # only used for the optional sanity spot-check
S2_PATH = f"{DATA_DIR}/train/train_source2.tsv"
S3_PATH = f"{DATA_DIR}/train/train_source3.tsv"

FEATURE_COLS = [
    "lev_ratio_name", "lev_ratio_address", "jw_name", "jw_address",
    "jaccard_name", "jaccard_address", "tfidf_cosine_name",
    "tfidf_cosine_address", "tfidf_cosine_combined", "country_match",
    "len_diff_name", "len_diff_address",
]


def sanity_checks(feat_df, orig_s1_ids):
    print("\n--- SANITY CHECKS ---")

    # 1. Every original S1 entity still present (no silent drops)
    scored_s1_ids = set(feat_df["source1_entity_id"].unique())
    missing_s1 = orig_s1_ids - scored_s1_ids
    print(f"1. S1 entities in input: {len(orig_s1_ids)} | "
          f"present in output: {len(scored_s1_ids)} | "
          f"missing: {len(missing_s1)}")
    if missing_s1:
        print(f"   WARNING: {len(missing_s1)} S1 entities vanished from the output. "
              f"Example missing IDs: {list(missing_s1)[:5]}")
    else:
        print("   OK -- no entity silently dropped.")

    # 2. Positive rate sanity check against training-time expectation (~2.2-2.5%)
    pos_rate = feat_df["predicted_label"].mean()
    print(f"2. Predicted-positive rate on full inference: {pos_rate:.4%}")
    if not (0.005 <= pos_rate <= 0.10):
        print("   WARNING: this is well outside the ~2.2-2.5% range seen in train/val. "
              "Possible distribution shift or a bug -- investigate before handing off.")
    else:
        print("   OK -- roughly consistent with training/val positive rate.")

    # 3. Spot-check a few high- and low-probability pairs against real text,
    #    if the raw source files are reachable. Streams through files
    #    looking only for the handful of IDs we need (not a full 5M-row load).
    try:
        top = feat_df.nlargest(3, "probability")
        bottom = feat_df.nsmallest(3, "probability")
        spot_rows = pd.concat([top, bottom])

        needed_ids = set(spot_rows["source1_entity_id"]) | set(spot_rows["candidate_entity_id"])
        found = {}

        def scan_for_ids(path, ids_wanted, chunksize=200_000):
            remaining = set(ids_wanted)
            for chunk in pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str,
                                      chunksize=chunksize):
                if not remaining:
                    break
                hits = chunk[chunk["entity_id"].isin(remaining)]
                for _, row in hits.iterrows():
                    found[row["entity_id"]] = (row["business_name"], row["business_address"])
                    remaining.discard(row["entity_id"])

        s1_ids_needed = {i for i in needed_ids if i.startswith("S1-")}
        s2_ids_needed = {i for i in needed_ids if i.startswith("S2-")}
        s3_ids_needed = {i for i in needed_ids if i.startswith("S3-")}
        if s1_ids_needed:
            scan_for_ids(S1_PATH, s1_ids_needed)
        if s2_ids_needed:
            scan_for_ids(S2_PATH, s2_ids_needed)
        if s3_ids_needed:
            scan_for_ids(S3_PATH, s3_ids_needed)

        def fmt(eid):
            if eid not in found:
                return f"{eid}: (not found)"
            name, addr = found[eid]
            return f"{eid}: '{name}' / '{addr}'"

        print("\n3. Spot-check -- top 3 HIGHEST probability pairs:")
        for _, r in top.iterrows():
            print(f"   prob={r.probability:.4f}")
            print(f"     S1: {fmt(r.source1_entity_id)}")
            print(f"     Candidate: {fmt(r.candidate_entity_id)}")

        print("\n   Spot-check -- bottom 3 LOWEST probability pairs:")
        for _, r in bottom.iterrows():
            print(f"   prob={r.probability:.4f}")
            print(f"     S1: {fmt(r.source1_entity_id)}")
            print(f"     Candidate: {fmt(r.candidate_entity_id)}")
    except FileNotFoundError:
        print("\n3. Spot-check skipped -- raw source files not found at the configured paths.")


def main():
    print("Loading model and threshold...")
    booster = lgb.Booster(model_file=MODEL_PATH)
    threshold = pd.read_csv(THRESHOLD_PATH, sep="\t")["best_threshold"].iloc[0]
    print(f"  Threshold: {threshold}")

    print("Loading full features.tsv (ALL pairs, not just val split)...")
    feat_df = pd.read_csv(FEATURES_PATH, sep="\t")
    orig_s1_ids = set(feat_df["source1_entity_id"].unique())
    print(f"  {feat_df.shape[0]} pairs, {len(orig_s1_ids)} unique Source1 entities")

    print("Scoring...")
    feat_df["probability"] = booster.predict(feat_df[FEATURE_COLS])
    feat_df["predicted_label"] = (feat_df["probability"] >= threshold).astype(int)

    out_df = feat_df[["source1_entity_id", "candidate_entity_id", "probability", "predicted_label"]]
    out_path = f"{OUT_DIR}/predictions.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}: {len(out_df)} rows (one per input candidate pair, nothing filtered/reshaped)")

    # Convenience wide-format export, alongside the raw pair-level file above
    # (not a replacement -- the raw file stays the source of truth for
    # auditing individual pair probabilities; this is just to save D a
    # manual conversion step every run).
    wide_rows = []
    for s1_id, grp in out_df.groupby("source1_entity_id"):
        matched = list(dict.fromkeys(grp.loc[grp.predicted_label == 1, "candidate_entity_id"]))
        wide_rows.append((s1_id, ",".join(matched)))
    wide_df = pd.DataFrame(wide_rows, columns=["source1_entity_id", "matched_entity_ids"])
    wide_path = f"{OUT_DIR}/predictions_wide.tsv"
    wide_df.to_csv(wide_path, sep="\t", index=False)
    print(f"Wrote {wide_path}: {len(wide_df)} rows (wide format: one row per S1 entity, "
          f"empty string if no candidate crossed the threshold)")

    sanity_checks(out_df, orig_s1_ids)

    print("\n--- HANDOFF NOTE FOR PERSON D ---")
    print("This file has one row per (source1_entity_id, candidate_entity_id) INPUT pair.")
    print("An S1 entity with candidates but predicted_label=0 for all of them still has")
    print("rows in this file -- it is NOT collapsed to an empty-list singleton row here.")
    print("Confirm with D whether they want:")
    print("  (a) this raw shape, and they derive singletons themselves by grouping and")
    print("      checking for zero positive rows per source1_entity_id, or")
    print("  (b) an explicit empty row per singleton entity added on C's side instead.")
    print("Whichever D expects, make sure both sides agree -- don't assume.")


if __name__ == "__main__":
    main()