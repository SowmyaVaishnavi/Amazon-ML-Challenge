"""
03_predict_test.py

Scores every (source1_entity_id, candidate_entity_id) pair from
candidate_pairs_test.tsv with the saved LightGBM model + tuned threshold,
and writes the two files required by the challenge:

    output/matching_results.tsv   (only IDs scoring >= threshold)
    output/candidate_pairs.tsv    (the full candidate set fed to the model)

Both follow the exact spec: one row per Source-1 test entity, empty string
for no matches, no duplicate IDs, comma-separated, no quoting.
"""

import pandas as pd
import lightgbm as lgb
from common_features import compute_pair_features, FEATURE_COLS

DATA_DIR = "/mnt/user-data/uploads"
OUT_DIR = "/mnt/user-data/outputs"

CANDIDATES_PATH = f"{OUT_DIR}/candidate_pairs_test.tsv"   # <-- swap for B's real test candidates later
S1_PATH = f"{DATA_DIR}/test_source1.tsv"
S2_PATH = f"{DATA_DIR}/test_source2.tsv"
S3_PATH = f"{DATA_DIR}/test_source3.tsv"
MODEL_PATH = f"{OUT_DIR}/lgbm_matcher.txt"
THRESHOLD_PATH = f"{OUT_DIR}/best_threshold.tsv"


def load_records(path):
    df = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    return df.set_index("entity_id")[["business_name", "business_address", "country"]]


def main():
    threshold = pd.read_csv(THRESHOLD_PATH, sep="\t")["best_threshold"].iloc[0]
    print(f"Using threshold: {threshold}")

    booster = lgb.Booster(model_file=MODEL_PATH)

    print("Loading test source records...")
    s1_recs = load_records(S1_PATH)
    s2_recs = load_records(S2_PATH)
    s3_recs = load_records(S3_PATH)

    cand = pd.read_csv(CANDIDATES_PATH, sep="\t", keep_default_na=False, dtype=str)

    match_rows = []
    cand_rows = []  # mirrors candidate_pairs.tsv exactly (the set fed to the model)

    for s1_id, cand_str in zip(cand["source1_entity_id"], cand["candidate_entity_ids"]):
        cand_ids = cand_str.split(",") if cand_str else []
        cand_rows.append((s1_id, ",".join(cand_ids)))  # last-stage candidate set = same set we score

        if not cand_ids or s1_id not in s1_recs.index:
            match_rows.append((s1_id, ""))
            continue

        r1 = s1_recs.loc[s1_id]
        feats = []
        valid_ids = []
        for cid in cand_ids:
            recs = s2_recs if cid.startswith("S2-") else s3_recs
            if cid not in recs.index:
                continue
            r2 = recs.loc[cid]
            feats.append(compute_pair_features(
                r1["business_name"], r1["business_address"], r1["country"],
                r2["business_name"], r2["business_address"], r2["country"],
            ))
            valid_ids.append(cid)

        if not feats:
            match_rows.append((s1_id, ""))
            continue

        feat_df = pd.DataFrame(feats)[FEATURE_COLS]
        scores = booster.predict(feat_df)
        matched = [cid for cid, s in zip(valid_ids, scores) if s >= threshold]

        # de-dup while preserving order (safety net)
        seen = set()
        matched_uniq = [m for m in matched if not (m in seen or seen.add(m))]
        match_rows.append((s1_id, ",".join(matched_uniq)))

    match_df = pd.DataFrame(match_rows, columns=["source1_entity_id", "matched_entity_ids"])
    cand_out_df = pd.DataFrame(cand_rows, columns=["source1_entity_id", "candidate_entity_ids"])

    match_df.to_csv(f"{OUT_DIR}/matching_results.tsv", sep="\t", index=False)
    cand_out_df.to_csv(f"{OUT_DIR}/candidate_pairs.tsv", sep="\t", index=False)

    n_matched = (match_df["matched_entity_ids"] != "").sum()
    print(f"Wrote matching_results.tsv: {len(match_df)} rows, "
          f"{n_matched} entities with >=1 match ({n_matched/len(match_df):.2%})")
    print(f"Wrote candidate_pairs.tsv: {len(cand_out_df)} rows")


if __name__ == "__main__":
    main()