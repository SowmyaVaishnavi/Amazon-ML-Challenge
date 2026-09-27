"""
02_train_matcher.py

Trains on B's real precomputed features.tsv (columns: source1_entity_id,
candidate_entity_id, lev_ratio_name, lev_ratio_address, jw_name, jw_address,
jaccard_name, jaccard_address, tfidf_cosine_name, tfidf_cosine_address,
tfidf_cosine_combined, country_match, len_diff_name, len_diff_address).

B's file has NO label column -- labels come from train_ground_truth.tsv,
joined in here. Does an entity-level train/val split, negative-samples,
trains LightGBM, and reports TWO threshold sweeps side by side:

  1. Macro-averaged per-Source1-entity F_0.5 -- the ACTUAL competition
     metric (singletons included, correct-empty = 1.0, any false positive
     on a singleton = 0.0). The saved threshold is chosen by THIS sweep,
     since it's what the leaderboard actually scores.
  2. Plain pair-level positive-class-only precision/recall/F_0.5 -- shown
     alongside for visibility, since macro-averaging across many easy
     true-negative entities can mask a mediocre positive class. If the two
     sweeps pick very different thresholds, that's worth a closer look.

Also prints LightGBM feature importances so you can see which of B's
features are actually pulling weight (and confirm dead features, e.g.
country_match when blocking already enforces same-country candidates).

Input contract: when B sends MORE/UPDATED features.tsv (more entities, or
a refreshed full-scale run), just repoint FEATURES_PATH below -- nothing
else in this script needs to change, since it reads whatever columns are
in FEATURE_COLS.

Outputs:
    lgbm_matcher.txt      -- saved LightGBM model
    best_threshold.tsv    -- tuned decision threshold (chosen by macro F_0.5)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = r"C:\Users\LENOVO\Desktop\AWA hack\dataset"
OUT_DIR = r"C:\Users\LENOVO\Desktop\AWA hack\outputs"

FEATURES_PATH = r"C:\Users\LENOVO\Desktop\AWA hack\features.tsv"   # <-- B's real train features file. Move it here or fix this path.
GT_PATH = f"{DATA_DIR}/train/train_ground_truth.tsv"

FEATURE_COLS = [
    "lev_ratio_name", "lev_ratio_address", "jw_name", "jw_address",
    "jaccard_name", "jaccard_address", "tfidf_cosine_name",
    "tfidf_cosine_address", "tfidf_cosine_combined", "country_match",
    "len_diff_name", "len_diff_address",
]

NEG_PER_POS = 5
RANDOM_STATE = 42
THRESHOLDS = np.arange(0.05, 0.96, 0.05)


def load_labels(gt_path):
    gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False, dtype=str)
    label_map = {}
    for s1_id, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        if matched:
            for cid in matched.split(","):
                label_map[(s1_id, cid)] = 1
    return label_map


def entity_level_split(df, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, val_idx = next(gss.split(df, groups=df["source1_entity_id"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def negative_sample(df, neg_per_pos=NEG_PER_POS):
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    n_keep = min(len(neg), max(len(pos) * neg_per_pos, 1))
    neg_sampled = neg.sample(n=n_keep, random_state=RANDOM_STATE) if n_keep > 0 else neg
    return pd.concat([pos, neg_sampled]).sample(frac=1, random_state=RANDOM_STATE)


def macro_f_beta(val_df, scores, threshold, beta=0.5):
    """Competition metric: per-Source1-entity F_beta, macro-averaged."""
    val_df = val_df.copy()
    val_df["pred"] = (scores >= threshold).astype(int)
    f_scores = []
    for s1_id, grp in val_df.groupby("source1_entity_id"):
        true_pos = set(grp.loc[grp.label == 1, "candidate_entity_id"])
        pred_pos = set(grp.loc[grp.pred == 1, "candidate_entity_id"])
        if len(true_pos) == 0:
            f_scores.append(1.0 if len(pred_pos) == 0 else 0.0)
            continue
        tp = len(true_pos & pred_pos)
        fp = len(pred_pos - true_pos)
        fn = len(true_pos - pred_pos)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision == 0 and recall == 0:
            f_scores.append(0.0)
        else:
            b2 = beta ** 2
            f_scores.append((1 + b2) * precision * recall / (b2 * precision + recall))
    return float(np.mean(f_scores))


def pair_level_pos_metrics(labels, scores, threshold, beta=0.5):
    """Plain pair-level precision/recall/F_beta for the POSITIVE class only
    (not macro-averaged across entities). Reveals whether the model is
    actually good at finding matches, since macro F_0.5 across many easy
    true-negative entities can mask a mediocre positive class."""
    preds = (scores >= threshold).astype(int)
    labels = np.asarray(labels)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta ** 2
    f = (1 + b2) * precision * recall / (b2 * precision + recall) if (precision + recall) > 0 else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f0.5": f, "tp": tp, "fp": fp, "fn": fn}


def main():
    print("Loading B's features...")
    feat_df = pd.read_csv(FEATURES_PATH, sep="\t")
    print(f"  {feat_df.shape[0]} pairs, {feat_df.source1_entity_id.nunique()} unique Source1 entities")

    print("Loading ground truth labels...")
    label_map = load_labels(GT_PATH)
    feat_df["label"] = [
        label_map.get((s1, c), 0)
        for s1, c in zip(feat_df.source1_entity_id, feat_df.candidate_entity_id)
    ]
    print(f"  Positive rate: {feat_df.label.mean():.4f}  ({int(feat_df.label.sum())} positives)")

    # Blocking recall check: are there S1 entities in this feature file whose
    # true match never appears anywhere in it (i.e. B's blocking missed it)?
    gt_df = pd.read_csv(GT_PATH, sep="\t", keep_default_na=False, dtype=str)
    s1_in_features = set(feat_df.source1_entity_id.unique())
    gt_sub = gt_df[gt_df.source1_entity_id.isin(s1_in_features)]
    all_true_ids = set()
    for m in gt_sub.matched_entity_ids:
        if m:
            all_true_ids.update(m.split(","))
    found_true_ids = set(feat_df.loc[feat_df.label == 1, "candidate_entity_id"])
    missing = all_true_ids - found_true_ids
    print(f"  Blocking recall gap (true matches missing from B's candidates): "
          f"{len(missing)} / {len(all_true_ids)} ({len(missing)/max(len(all_true_ids),1):.2%})")

    train_df, val_df = entity_level_split(feat_df)
    print(f"Train pairs: {len(train_df)} | Val pairs: {len(val_df)}")

    train_bal = negative_sample(train_df)
    print(f"After negative sampling: {len(train_bal)} "
          f"(pos={int(train_bal.label.sum())}, neg={int((train_bal.label==0).sum())})")

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        objective="binary", random_state=RANDOM_STATE,
    )
    model.fit(train_bal[FEATURE_COLS], train_bal["label"])

    scores = model.predict_proba(val_df[FEATURE_COLS])[:, 1]
    labels = val_df["label"].values

    # ---- Sweep 1: macro F_0.5 (the real competition metric) ----
    macro_results = sorted(
        ((t, macro_f_beta(val_df, scores, t)) for t in THRESHOLDS),
        key=lambda x: -x[1],
    )
    print("\n[1] Macro per-entity F_0.5 sweep (competition metric, best first):")
    for t, f05 in macro_results[:10]:
        print(f"    threshold={t:.2f}  macro F_0.5={f05:.4f}")
    best_threshold, best_macro_f05 = macro_results[0]
    print(f"  -> Best by macro F_0.5: threshold={best_threshold:.2f}, macro F_0.5={best_macro_f05:.4f}")

    # ---- Sweep 2: plain positive-class precision/recall/F_0.5 ----
    pos_results = [pair_level_pos_metrics(labels, scores, t) for t in THRESHOLDS]
    pos_results_sorted = sorted(pos_results, key=lambda r: -r["f0.5"])
    print("\n[2] Positive-class-only pair-level sweep (diagnostic, best first):")
    for r in pos_results_sorted[:10]:
        print(f"    threshold={r['threshold']:.2f}  precision={r['precision']:.4f}  "
              f"recall={r['recall']:.4f}  F_0.5={r['f0.5']:.4f}  "
              f"(tp={r['tp']}, fp={r['fp']}, fn={r['fn']})")
    best_pos = pos_results_sorted[0]
    print(f"  -> Best by positive-class F_0.5: threshold={best_pos['threshold']:.2f}, "
          f"F_0.5={best_pos['f0.5']:.4f} (precision={best_pos['precision']:.4f}, recall={best_pos['recall']:.4f})")

    if abs(best_threshold - best_pos["threshold"]) > 1e-9:
        print(f"\n  NOTE: the two sweeps disagree on the best threshold "
              f"({best_threshold:.2f} vs {best_pos['threshold']:.2f}). "
              f"Saving the macro-F_0.5 choice since that's what the leaderboard scores.")

    # Report positive-class metrics AT the threshold we're actually saving,
    # so you always see both numbers for the same decision point.
    chosen_pos_metrics = pair_level_pos_metrics(labels, scores, best_threshold)
    print(f"\nAt saved threshold={best_threshold:.2f}: "
          f"macro F_0.5={best_macro_f05:.4f} | "
          f"positive-class precision={chosen_pos_metrics['precision']:.4f}, "
          f"recall={chosen_pos_metrics['recall']:.4f}, F_0.5={chosen_pos_metrics['f0.5']:.4f}")
    print(f"Val set size: {len(val_df)} pairs, positive rate: {val_df.label.mean():.4f}")

    # ---- Feature importance ----
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\nFeature importances (gain-based, from LightGBM):")
    for feat, imp in importances.items():
        print(f"    {feat:<25s} {imp}")

    # ---- Save outputs ----
    model.booster_.save_model(f"{OUT_DIR}/lgbm_matcher.txt")
    threshold_df = pd.DataFrame([{
        "best_threshold": best_threshold,
        "macro_f0.5": best_macro_f05,
        "pos_precision": chosen_pos_metrics["precision"],
        "pos_recall": chosen_pos_metrics["recall"],
        "pos_f0.5": chosen_pos_metrics["f0.5"],
    }])
    threshold_df.to_csv(f"{OUT_DIR}/best_threshold.tsv", sep="\t", index=False)
    print(f"\nSaved model -> {OUT_DIR}/lgbm_matcher.txt")
    print(f"Saved threshold -> {OUT_DIR}/best_threshold.tsv")


if __name__ == "__main__":
    main()