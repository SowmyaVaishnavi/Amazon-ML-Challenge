"""
common_features.py
Shared blocking-key normalization and pairwise feature computation.
Used by both 02_train_matcher.py and 03_predict_test.py so train/test
features are computed IDENTICALLY.
"""

import re
from rapidfuzz import fuzz

LEGAL_SUFFIXES = [
    "private limited", "pvt ltd", "pvt limited", "private ltd",
    "limited", "ltd", "llc", "inc", "incorporated", "corp",
    "corporation", "co", "company", "sarl", "sas", "llp",
]

_punct_re = re.compile(r"[^\w\s]", re.UNICODE)
_space_re = re.compile(r"\s+")


def normalize_text(s):
    """Lowercase, strip punctuation, collapse whitespace. Unicode-safe
    (Hindi/Devanagari business names in Source 2/3 must survive this)."""
    if not isinstance(s, str) or s == "":
        return ""
    s = s.lower()
    s = _punct_re.sub(" ", s)
    s = _space_re.sub(" ", s).strip()
    return s


def strip_legal_suffix(name_norm):
    """Remove common legal-entity suffixes AFTER normalize_text."""
    tokens = name_norm.split()
    # drop trailing tokens that match known suffixes (longest match first)
    changed = True
    while changed and tokens:
        changed = False
        for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
            suf_tokens = suf.split()
            n = len(suf_tokens)
            if n and tokens[-n:] == suf_tokens:
                tokens = tokens[:-n]
                changed = True
                break
    return " ".join(tokens)


def blocking_key(name, country):
    """
    Blocking key = country + first 4 chars of (suffix-stripped) normalized
    name with spaces removed. This is a lightweight stand-in blocking
    strategy -- Person B's real blocking/candidate-generation will replace
    this; the model script below does not care how candidates were produced.
    """
    norm = strip_legal_suffix(normalize_text(name))
    norm_nospace = norm.replace(" ", "")
    prefix = norm_nospace[:4] if norm_nospace else "____"
    country_norm = normalize_text(country) or "unk"
    return f"{country_norm}|{prefix}"


def compute_block_keys_vectorized(names, countries):
    """
    Same logic as blocking_key(), but operates on whole pandas Series at
    once using vectorized string ops instead of a Python loop per row.
    This is 10-20x faster on multi-million-row files -- use this in
    01_build_candidates.py instead of calling blocking_key() per row.
    """
    s = names.fillna("").astype(str).str.lower()
    s = s.str.replace(r"[^\w\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    # Strip trailing legal suffixes. Two passes catches stacked suffixes
    # (e.g. "xyz pvt ltd" -> strip "ltd" -> strip "pvt") without needing a
    # per-row while loop.
    for _ in range(2):
        for suf in sorted(LEGAL_SUFFIXES, key=len, reverse=True):
            pattern = r"(^|\s)" + re.escape(suf) + r"$"
            s = s.str.replace(pattern, "", regex=True).str.strip()

    nospace = s.str.replace(" ", "", regex=False)
    prefix = nospace.str[:4]
    prefix = prefix.mask(prefix.isna() | (prefix == ""), "____")

    country_norm = countries.fillna("").astype(str).str.lower().str.strip()
    country_norm = country_norm.mask(country_norm == "", "unk")

    return country_norm + "|" + prefix


def token_set(s):
    return set(normalize_text(s).split())


def jaccard(a_tokens, b_tokens):
    if not a_tokens and not b_tokens:
        return 1.0
    union = a_tokens | b_tokens
    if not union:
        return 0.0
    return len(a_tokens & b_tokens) / len(union)


def compute_pair_features(name1, addr1, country1, name2, addr2, country2):
    """Compute the real similarity feature vector for one (S1, candidate) pair."""
    n1, n2 = normalize_text(name1), normalize_text(name2)
    a1, a2 = normalize_text(addr1), normalize_text(addr2)

    n1_tok, n2_tok = set(n1.split()), set(n2.split())
    a1_tok, a2_tok = set(a1.split()), set(a2.split())

    return {
        "name_ratio": fuzz.ratio(n1, n2) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(n1, n2) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(n1, n2) / 100.0,
        "name_jaccard": jaccard(n1_tok, n2_tok),
        "address_token_sort_ratio": fuzz.token_sort_ratio(a1, a2) / 100.0,
        "address_jaccard": jaccard(a1_tok, a2_tok),
        "name_len_diff": abs(len(n1) - len(n2)),
        "address_len_diff": abs(len(a1) - len(a2)),
        "country_match": int(normalize_text(country1) == normalize_text(country2)),
        "name1_empty": int(n1 == ""),
        "name2_empty": int(n2 == ""),
        "addr1_empty": int(a1 == ""),
        "addr2_empty": int(a2 == ""),
    }


FEATURE_COLS = [
    "name_ratio", "name_token_sort_ratio", "name_partial_ratio",
    "name_jaccard", "address_token_sort_ratio", "address_jaccard",
    "name_len_diff", "address_len_diff", "country_match",
    "name1_empty", "name2_empty", "addr1_empty", "addr2_empty",
]