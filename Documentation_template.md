
## 1. Project Overview

This document consolidates the methodologies for the complete Business Entity Resolution pipeline: normalization and blocking, feature engineering, matching-model training and prediction, and evaluation/submission packaging.

The overall pipeline is designed to avoid all-pairs comparison, retain true candidate matches, convert candidate pairs into numerical similarity features, classify candidate pairs as matches/non-matches, and produce a validated submission in the required schema.

The challenge contains Source 1 reference entities, Source 2 and Source 3 candidate entities, training ground truth, and test data without ground truth. Each record contains `entity_id`, `business_name`, `business_address`, and `country`. fileciteturn0file2L11-L20

---

## 2. End-to-End Pipeline

```text
Source 1 + Source 2 + Source 3
              |
              v
     Normalization
              |
              v
   Blocking / Candidate Generation
              |
              v
      Candidate Pairs
              |
              v
     Feature Engineering
              |
              v
       Similarity Features
              |
              v
       Ground-Truth Label Join
              |
              v
     Entity-Level Train/Val Split
              |
              v
        Negative Sampling
              |
              v
     LightGBM Binary Classifier
              |
              v
       Threshold Tuning
              |
              v
          Predictions
              |
              v
   Formatting + Validation + Scoring
              |
              v
       Final Submission
```

The matching-model implementation follows this sequence from candidate pairs through features, labels, entity-level splitting, negative sampling, classification, threshold tuning, and prediction generation. fileciteturn0file1L15-L36

---

# 3. Workstream A — Normalization, Blocking, and Candidate Generation

## 3.1 Objective

The objective is to avoid comparing every Source 1 entity against every Source 2/3 entity. Instead, a substantially smaller set of plausible candidates is generated while preserving true matches. fileciteturn0file2L18-L20

## 3.2 Normalization

Records are normalized so that spelling, punctuation, formatting, and legal-suffix variations become more comparable.

### General text normalization

1. Unicode NFKC normalization.
2. Transliteration using `unidecode`.
3. Lowercasing.
4. Replacing non-alphanumeric characters with spaces.
5. Collapsing repeated whitespace.
6. Converting missing, empty, and `nan` values to empty strings.

### Business names

Common legal/company suffixes are repeatedly removed from the end of names, including:

`inc`, `incorporated`, `corp`, `corporation`, `co`, `company`, `llc`, `ltd`, `limited`, `plc`, `pvt`, `private`, `llp`, `sarl`, `sas`, `gmbh`, `ag`, `sa`.

### Addresses

Common address terms are abbreviated:

- `street` → `st`
- `road` → `rd`
- `avenue` → `ave`
- `boulevard` → `blvd`
- `drive` → `dr`
- `lane` → `ln`
- `highway` → `hwy`
- `apartment` → `apt`
- `suite` → `ste`

### Country

Country is lowercased and treated as an open set rather than being restricted to countries observed during development.

These normalization rules are defined in the Person A methodology. fileciteturn0file2L22-L57

## 3.3 Blocking Strategy

Blocking uses multiple complementary signals, generally combined with normalized country.

### Name blocks

- `nameprefix2`
- `nameprefix3`
- `nameprefix4`
- `name2prefix`
- `nameword4`
- `nameword5`
- `nameshort`
- `namegram`

### Address blocks

- `fulladdress`
- `addrword4`
- `addrword5`
- `addrshort`

Multiple block families are used for recall: a true match only needs to share at least one useful blocking key. fileciteturn0file2L59-L83

## 3.4 Candidate Generation

For every Source 1 entity:

1. Generate all applicable blocking keys.
2. Look up each key in the Source 2/3 index.
3. Union returned entity IDs.
4. Remove duplicates.
5. Sort deterministically.
6. Write one output row.

Required format:

```text
source1_entity_id    candidate_entity_ids
```

Every TEST Source 1 entity must have exactly one row, including entities with empty candidate lists. Candidate IDs must exist in TEST Source 2 or Source 3. fileciteturn0file2L85-L104

## 3.5 Candidate Validation

Development validation uses training ground truth because TEST has no labels.

The main development subset contains the first 1,000 Source 1 entities. True-match coverage asks whether at least one blocking key retains every known true match.

For V5.1:
- 1,000 Source 1 entities were checked.
- 3,517 known true matches were checked.
- 3,517 / 3,517 were covered.
- True-match block recall was 100%.
- 946 / 1,000 Source 1 entities had all known true matches covered.

This is a recall diagnostic only and does not measure false positives. fileciteturn0file2L106-L128

## 3.6 Blocking Refinement

V5.1.1 was created as a tighter experiment by removing:
- `namegram`
- `nameshort`
- `addrshort`

The retained blocks were `nameprefix2`, `nameprefix3`, `nameprefix4`, `name2prefix`, `nameword4`, `nameword5`, `addrword4`, `addrword5`, and `fulladdress`.

V5.1.1 must be evaluated using the true-match coverage diagnostic before being accepted as a replacement. fileciteturn0file2L152-L172

## 3.7 Full-Population Decoy Validation

True-match-only validation is insufficient for candidate-size claims because it excludes unrelated records.

Therefore, candidate generation should also be evaluated against the complete Source 2 + Source 3 population to measure:
- candidate-set growth
- broad-block effects
- decoy volume
- recall versus candidate-reduction trade-off

Production reduction-ratio claims must be based on the full population. fileciteturn0file2L174-L187

## 3.8 Candidate Output

Production candidate file:

```text
output/candidate_pairs.tsv
```

Required properties:
- exactly one row per TEST Source 1 entity
- no duplicate candidate IDs
- every candidate ID exists in TEST Source 2/3
- empty candidate lists allowed
- deterministic output
- no external business databases/APIs/geocoding
- final downstream matches must be a subset of candidate IDs. fileciteturn0file2L228-L250

---

# 4. Workstream B — Feature Engineering

## 4.1 Objective

For each candidate pair, compute a fixed set of similarity features that quantify how likely the two records are to represent the same real-world business.

Feature engineering receives raw `(business_name, business_address, country)` triples and does not depend on shared upstream normalization.

## 4.2 Feature-Specific Cleanup

The feature module performs:
- lowercasing
- whitespace collapsing
- empty-field handling

This keeps feature generation independently testable and prevents dependency on the normalization/blocking implementation.

## 4.3 Feature Set

Twelve features are computed:

| Feature | Description |
|---|---|
| `lev_ratio_name` | Levenshtein similarity on business name |
| `lev_ratio_address` | Levenshtein similarity on business address |
| `jw_name` | Jaro-Winkler similarity on business name |
| `jw_address` | Jaro-Winkler similarity on business address |
| `jaccard_name` | Token-level Jaccard overlap on business name |
| `jaccard_address` | Token-level Jaccard overlap on business address |
| `tfidf_cosine_name` | Character n-gram TF-IDF cosine similarity on name |
| `tfidf_cosine_address` | Character n-gram TF-IDF cosine similarity on address |
| `tfidf_cosine_combined` | Character n-gram TF-IDF cosine on name + address |
| `country_match` | Exact country-match binary flag |
| `len_diff_name` | Absolute cleaned-name length difference |
| `len_diff_address` | Absolute cleaned-address length difference |

The matching-model methodology confirms that these twelve columns are present in the delivered feature set. fileciteturn0file1L58-L76

## 4.4 Feature Rationale

### Character-level similarity
Levenshtein and Jaro-Winkler capture typos, spelling variations, and transliteration noise.

### Token-level similarity
Token Jaccard captures word overlap while being less sensitive to word order.

### Character n-gram TF-IDF
Character n-gram TF-IDF cosine provides an additional similarity signal that emphasizes distinctive character patterns. The implementation derives IDF from the two documents being compared rather than relying on a global corpus.

### Structural features
Country equality and length differences provide inexpensive additional signals.

No feature uses external data lookups, APIs, or government registries. fileciteturn0file1L78-L80

## 4.5 Performance Optimization

The initial pure-Python implementation processed approximately 1,750 candidate pairs/second.

Approximate cost distribution:

| Component | Share |
|---|---:|
| Levenshtein ×2 | ~53% |
| TF-IDF cosine ×3 | ~37% |
| Jaro-Winkler ×2 | ~8% |
| Token Jaccard ×2 | ~2% |

`rapidfuzz` replaced the pure-Python Levenshtein and Jaro-Winkler implementations where available, with automatic fallback.

Measured speedups:
- Levenshtein: ~345×
- Jaro-Winkler: ~55×
- Overall per-pair cost: reduced by roughly 61%

Because candidate-pair feature computations are independent, the stage can also be parallelized across CPU cores.

## 4.6 Correctness and Testing

The optimized implementations were compared with the original implementations across exact matches, empty strings, single characters, and high/low similarity cases.

Levenshtein produced exact agreement with maximum observed difference `0.000000`.

The Jaro-Winkler implementation exposed an original bug: the Winkler prefix boost was applied unconditionally. It was corrected so that the boost is applied only when the base Jaro score exceeds the conventional `0.7` threshold.

Independent testing covered 25 hand-written pairs including exact matches, case/whitespace differences, legal suffix differences, typos, abbreviations, transliteration, word reordering, truncated addresses, deliberate non-matches, branch differences, country mismatches, empty fields, and single-character names.

## 4.7 Feature Output

One row is produced for every upstream candidate pair:

```text
source1_entity_id
candidate_entity_id
lev_ratio_name
lev_ratio_address
jw_name
jw_address
jaccard_name
jaccard_address
tfidf_cosine_name
tfidf_cosine_address
tfidf_cosine_combined
country_match
len_diff_name
len_diff_address
```

No match/no-match label is generated at this stage.

## 4.8 Feature Limitations

Acronym/full-name pairs can receive low scores because literal overlap is limited. A future acronym-expansion or initialism feature could address this.

`country_match` may be redundant if blocking already guarantees same-country candidates. In the delivered model, this was observed: the feature had zero gain-based importance because the blocking key enforced country equality. fileciteturn0file1L82-L87

---

# 5. Workstream C — Matching Model

## 5.1 Objective

The matching-model stage receives candidate pairs and similarity features, joins training labels, trains a binary classifier, tunes a decision threshold, and produces scored predictions. fileciteturn0file1L5-L10

## 5.2 Label Generation

The feature file contains no labels.

Labels are joined from `train_ground_truth.tsv`:
- `1` if the candidate ID appears in the Source 1 entity's `matched_entity_ids`
- `0` otherwise

For the delivered 141,830-pair file:
- 3,517 positives
- 2.48% positive rate. fileciteturn0file1L89-L97

## 5.3 Train/Validation Split

The split is performed by `source1_entity_id` using `GroupShuffleSplit` with an 80/20 split.

This prevents pairs belonging to the same Source 1 entity from appearing in both train and validation and avoids information leakage. fileciteturn0file1L99-L106

The resulting split was:
- 112,114 train pairs
- 29,716 validation pairs

## 5.4 Negative Sampling

The pair-level class imbalance is substantial.

Training keeps all positive pairs and samples negatives at a 5:1 negative-to-positive ratio. Validation retains the natural prevalence.

Resulting balanced training set:
- 17,142 pairs
- 2,857 positive
- 14,285 negative. fileciteturn0file1L108-L119

## 5.5 Classifier

The model is a LightGBM binary classifier:

```text
n_estimators = 300
learning_rate = 0.05
num_leaves = 31
random_state = 42
```

The model is a gradient-boosted tree ensemble rather than a neural network or LLM.

It is saved as a LightGBM Booster text file and loaded at inference time. fileciteturn0file1L121-L132

## 5.6 Threshold Tuning

Two validation views are evaluated:

1. Macro per-entity F₀.₅ — the competition metric.
2. Positive-class pair-level precision, recall, and F₀.₅.

Macro F₀.₅ is computed per Source 1 entity and then averaged. Singleton entities are explicitly handled.

At threshold `0.80`:
- Macro F₀.₅ = 0.9625
- Positive precision = 0.9529
- Positive recall = 0.9803
- Positive F₀.₅ = 0.9582
- TP = 647
- FP = 32
- FN = 13. fileciteturn0file1L134-L153

## 5.7 Singleton Analysis

On the full 1,000-entity feature set:

| Threshold | Singleton Accuracy | Matched-Entity Coverage |
|---:|---:|---:|
| 0.80 | 87.04% | 100.00% |
| 0.85 | 88.89% | 99.89% |
| 0.90 | 88.89% | 99.89% |
| 0.95 | 90.74% | 99.89% |
| 0.99 | 94.44% | 99.89% |

Because F₀.₅ weights precision more strongly, the threshold was manually raised from 0.80 to 0.90. fileciteturn0file1L156-L173

## 5.8 False-Merge Analysis

Six false-merge cases were investigated. Their scores ranged from 0.91 to 0.9995, so they were not simply borderline threshold cases.

Several had near-zero name and address similarity while still receiving near-certain match probabilities.

The training data contained zero true-positive examples with near-zero name-and-address similarity. The leading hypothesis was a small-training-data artifact rather than a fundamental model-architecture issue.

The recommended remedy was to increase training coverage rather than change the model architecture or threshold. fileciteturn0file1L175-L207

## 5.9 Prediction Outputs

Two prediction artifacts are generated:

### `predictions.tsv`

Raw long-format predictions containing:
- `source1_entity_id`
- `candidate_entity_id`
- `probability`
- `predicted_label`

Every input entity is represented, including entities with no predicted matches.

### `predictions_wide.tsv`

Grouped representation:

```text
source1_entity_id    matched_entity_ids
```

The raw predictions remain the source of truth for pair-level auditing. fileciteturn0file1L209-L220

---

# 6. Workstream D — Evaluation Harness and Packaging

## 6.1 Objective

The evaluation workstream provides:
- reproducible train/validation splitting
- a local F₀.₅ scorer
- submission-format validation
- output formatting
- final packaging checks

Its purpose is to enable local development without touching the real test set, catch formatting errors before submission, and convert predictions into the exact required schema. fileciteturn0file0L5-L14

## 6.2 Train/Validation Split

Only Source 1 entities are split.

The development split uses:
- 85% train
- 15% validation
- seed `42`
- stratification by country

Source 2 and Source 3 remain at full size in both splits so that blocking recall and candidate-reduction measurements remain representative of the real matching problem. fileciteturn0file0L18-L40

A known limitation is that France appears in the real test set but is absent from training data, so the split cannot reproduce that domain shift. fileciteturn0file0L43-L48

## 6.3 F₀.₅ Scoring

The competition metric is:

```text
F_0.5 = (1.25 × Precision × Recall) /
        (0.25 × Precision + Recall)
```

The score is computed per Source 1 entity and then macro-averaged, with singletons included. fileciteturn0file0L50-L59

Explicit edge cases:

| True Matches | Predicted Matches | Score |
|---|---|---:|
| none | none | 1.0 |
| none | any | 0.0 |
| some | none | 0.0 |
| some | some, no overlap | 0.0 |

A precision=recall=0 division case is explicitly guarded and defined as 0.0. fileciteturn0file0L61-L74

## 6.4 Scorer Verification

Two baseline checks were performed:

- Ground truth used as predictions → macro F₀.₅ = `1.0000`
- Empty predictions → macro F₀.₅ equal to singleton fraction: `18,557 / 331,023 = 0.0561`

These checks were treated as a prerequisite before using the scorer. fileciteturn0file0L76-L87

## 6.5 Submission Validation

The validator checks:

1. Every test Source 1 entity has exactly one row.
2. No duplicate `source1_entity_id`.
3. All referenced IDs are valid Source 2/3 test entities.
4. No duplicate IDs inside a match list.
5. Correct column names/order and tab-separated format.
6. When candidate pairs are supplied, every predicted match is contained in the corresponding candidate list.

The validator can operate on test or local train/validation directories. fileciteturn0file0L89-L108

## 6.6 Output Formatting

The formatter accepts predictions in:
- dictionary form
- long-format DataFrame
- wide-format DataFrame

It writes the schema-correct TSV and guarantees one output row for every entity in `all_s1_ids`, including entities with zero predictions. fileciteturn0file0L115-L131

## 6.7 End-to-End Validation

On an initial 1,000-entity batch:

- Person A candidate pairs and Person C predictions were processed.
- Long-format predictions were converted into the required wide format.
- Submission validation passed.
- Macro F₀.₅ = `0.9764`
- Precision = `0.9751`
- Recall = `0.9889`
- Singleton accuracy = `88.89%` (48/54). fileciteturn0file0L133-L145

A threshold sweep appeared to improve F₀.₅ up to roughly `0.998`, but this was not adopted because only 84 of 141,830 pairs fell into the 0.90–0.998 interval, making the observed gain potentially sample-specific. Threshold tuning was therefore deferred until the full validation split. fileciteturn0file0L147-L156

---

# 7. Integrated Data Contracts

## 7.1 Input Record Schema

```text
entity_id
business_name
business_address
country
```

## 7.2 Candidate Pair Schema

```text
source1_entity_id
candidate_entity_id
```

## 7.3 Feature Schema

```text
source1_entity_id
candidate_entity_id
lev_ratio_name
lev_ratio_address
jw_name
jw_address
jaccard_name
jaccard_address
tfidf_cosine_name
tfidf_cosine_address
tfidf_cosine_combined
country_match
len_diff_name
len_diff_address
```

## 7.4 Raw Prediction Schema

```text
source1_entity_id
candidate_entity_id
probability
predicted_label
```

## 7.5 Final Submission Schema

```text
source1_entity_id
matched_entity_ids
```

The candidate-generation methodology fixes the candidate-file contract, while the evaluation harness ensures the final output follows the required submission schema and contains valid candidate references. fileciteturn0file2L228-L250 fileciteturn0file0L89-L103

---

# 8. Reproducibility

The model pipeline uses a fixed random seed of `42` for data splitting, negative sampling, and model training. Dependencies include:

```text
pandas
numpy
lightgbm
scikit-learn
```

The Person A workstream uses the normalization and candidate-generation scripts under:

```text
src/normalization.py
scripts/generate_candidates.py
scripts/diagnose_true_match_blocks.py
scripts/diagnose_blocking_keys.py
scripts/test_candidate_recall.py
scripts/generate_candidate_pairs_sample.py
```

The model workflow is:

```text
python 02_train_matcher.py
python 03_predict.py
```

The first trains the model, tunes the threshold, and saves the model/threshold artifacts; the second generates raw and grouped predictions. fileciteturn0file1L233-L242

---

# 9. Known Limitations and Risks

## Candidate Generation
- True-match recall alone does not measure false positives.
- Candidate-size/reduction measurements based only on known true references are not production reduction measurements.
- V5.1.1 requires further recall validation before replacement.
- Final production reduction claims require full Source 2 + Source 3 population validation. fileciteturn0file2L211-L226

## Feature Engineering
- Acronym/full-name pairs can score poorly.
- `country_match` can become redundant when blocking already enforces country equality.
- Character and token features depend on literal overlap and therefore may miss semantic equivalence.

## Matching Model
- The model was trained on only 1,000 Source 1 entities in the delivered development feature set.
- False-merge analysis suggests that more training coverage is needed for better generalization.
- Test-side features had not yet been received in the documented model state. fileciteturn0file1L222-L231

## Evaluation
- The local validation score is not the final leaderboard score.
- The real test set contains France, which is absent from training data.
- Threshold tuning on small samples can overfit borderline cases. fileciteturn0file0L159-L175

---

# 10. Final End-to-End Workflow

1. Load Source 1, Source 2, Source 3, and training ground truth.
2. Normalize names, addresses, and countries.
3. Generate complementary blocking keys.
4. Validate true-match coverage on the development subset.
5. Diagnose broad blocking families.
6. Evaluate tighter blocking variants.
7. Validate candidate generation against the full reference population.
8. Freeze the blocking strategy using the recall/decoy trade-off.
9. Generate the complete TEST candidate file.
10. Compute the twelve similarity features for every candidate pair.
11. Join ground-truth labels for training data.
12. Split by Source 1 entity to avoid leakage.
13. Apply negative sampling to the training split only.
14. Train the LightGBM binary classifier.
15. Tune the decision threshold using validation data.
16. Generate raw pair-level predictions.
17. Group predictions into the final submission format.
18. Validate IDs, row uniqueness, candidate membership, and schema.
19. Run the local F₀.₅ scorer where ground truth is available.
20. Produce the final submission only after all validation checks pass.

---

# 11. Current Project Status

### Person A
- V5.1 demonstrated 100% true-match block recall on the 1,000-entity development diagnostic.
- V5.1.1 exists as a tighter experimental variant.
- Full-population validation remains required before final candidate-reduction claims. fileciteturn0file2L293-L299

### Person B
- Feature engineering provides twelve fixed similarity/structural features.
- Optimized Levenshtein and Jaro-Winkler implementations were verified against the original implementations.
- Feature output is unlabeled and passed downstream to the model.

### Person C
- LightGBM matching model was trained and evaluated on the delivered candidate/feature set.
- Threshold was raised from 0.80 to 0.90 following singleton analysis.
- False-merge analysis indicates that additional training data is the main recommended improvement. fileciteturn0file1L156-L173

### Person D
- Train/validation splitting, F₀.₅ scoring, submission validation, and output formatting are implemented.
- End-to-end validation passed on the initial 1,000-entity batch.
- Final full-test execution remains dependent on complete candidate generation, feature generation, and prediction. fileciteturn0file0L133-L145
