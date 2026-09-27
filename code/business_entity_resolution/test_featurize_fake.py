"""
scripts/test_featurize_fake.py

Person B's independent smoke test. 25 hand-made (row1, row2, expected_label)
triples -- some obvious matches, some obvious non-matches, some tricky
near-miss cases (typos, suffix differences, transliteration variants,
country mismatch, truncated addresses, empty fields).

Run this immediately -- it needs nothing from Person A, no real
candidates.tsv, no source files. Just:

    python scripts\\test_featurize_fake.py

It prints a table of every feature per pair plus the expected label, so
you can eyeball whether matches score high and non-matches score low
before A's real file ever lands.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import featurize


FAKE_PAIRS = [
    # (label, row1, row2)
    ("match_exact",
     {"business_name": "Tata Consultancy Services", "business_address": "Mumbai Maharashtra 400001", "country": "IN"},
     {"business_name": "Tata Consultancy Services", "business_address": "Mumbai Maharashtra 400001", "country": "IN"}),

    ("match_case_diff",
     {"business_name": "INFOSYS LIMITED", "business_address": "Electronics City Bangalore", "country": "IN"},
     {"business_name": "infosys limited", "business_address": "electronics city bangalore", "country": "IN"}),

    ("match_suffix_diff",
     {"business_name": "Wipro Ltd", "business_address": "Sarjapur Road Bangalore", "country": "IN"},
     {"business_name": "Wipro Limited", "business_address": "Sarjapur Road Bangalore", "country": "IN"}),

    ("match_typo",
     {"business_name": "Reliance Industries", "business_address": "Nariman Point Mumbai", "country": "IN"},
     {"business_name": "Relaince Indutries", "business_address": "Nariman Point Mumbai", "country": "IN"}),

    ("match_abbrev_address",
     {"business_name": "HCL Technologies", "business_address": "Sector 126 Noida Uttar Pradesh", "country": "IN"},
     {"business_name": "HCL Technologies", "business_address": "Sec 126 Noida UP", "country": "IN"}),

    ("match_transliteration",
     {"business_name": "Bajaj Auto", "business_address": "Pune Maharashtra", "country": "IN"},
     {"business_name": "Bajaaj Auto", "business_address": "Pune Maharashtra", "country": "IN"}),

    ("match_word_order",
     {"business_name": "Auto Bajaj Industries", "business_address": "Akurdi Pune", "country": "IN"},
     {"business_name": "Bajaj Auto Industries", "business_address": "Pune Akurdi", "country": "IN"}),

    ("near_miss_different_branch",
     {"business_name": "State Bank of India", "business_address": "Connaught Place Delhi", "country": "IN"},
     {"business_name": "State Bank of India", "business_address": "Andheri Mumbai", "country": "IN"}),

    ("near_miss_similar_name_diff_company",
     {"business_name": "Bharat Heavy Electricals", "business_address": "Trichy Tamil Nadu", "country": "IN"},
     {"business_name": "Bharat Heavy Industries", "business_address": "Trichy Tamil Nadu", "country": "IN"}),

    ("nonmatch_completely_different",
     {"business_name": "Zomato Limited", "business_address": "Gurugram Haryana", "country": "IN"},
     {"business_name": "Adani Ports SEZ", "business_address": "Ahmedabad Gujarat", "country": "IN"}),

    ("nonmatch_country_mismatch",
     {"business_name": "Tech Mahindra", "business_address": "Pune Maharashtra", "country": "IN"},
     {"business_name": "Tech Mahindra", "business_address": "Pune Maharashtra", "country": "US"}),

    ("nonmatch_empty_vs_filled",
     {"business_name": "", "business_address": "", "country": ""},
     {"business_name": "Larsen and Toubro", "business_address": "Powai Mumbai", "country": "IN"}),

    ("edge_both_empty",
     {"business_name": "", "business_address": "", "country": ""},
     {"business_name": "", "business_address": "", "country": ""}),

    ("edge_single_char_name",
     {"business_name": "M", "business_address": "Delhi", "country": "IN"},
     {"business_name": "M Corp", "business_address": "Delhi", "country": "IN"}),

    ("match_hindi_translit_variant1",
     {"business_name": "Shree Ganesh Traders", "business_address": "Ahmedabad", "country": "IN"},
     {"business_name": "Shri Ganesh Traders", "business_address": "Ahmedabad", "country": "IN"}),

    ("match_hindi_translit_variant2",
     {"business_name": "Laxmi Enterprises", "business_address": "Jaipur Rajasthan", "country": "IN"},
     {"business_name": "Lakshmi Enterprises", "business_address": "Jaipur Rajasthan", "country": "IN"}),

    ("nonmatch_subword_overlap_only",
     {"business_name": "National Insurance Company", "business_address": "Kolkata", "country": "IN"},
     {"business_name": "National Textile Corporation", "business_address": "Kolkata", "country": "IN"}),

    ("match_pvt_ltd_variants",
     {"business_name": "Suzlon Energy Pvt Ltd", "business_address": "Pune", "country": "IN"},
     {"business_name": "Suzlon Energy Private Limited", "business_address": "Pune", "country": "IN"}),

    ("match_address_truncated",
     {"business_name": "Godrej Industries", "business_address": "Vikhroli West Mumbai Maharashtra 400079", "country": "IN"},
     {"business_name": "Godrej Industries", "business_address": "Vikhroli West Mumbai", "country": "IN"}),

    ("nonmatch_anagram_style",
     {"business_name": "Star India Media", "business_address": "Mumbai", "country": "IN"},
     {"business_name": "India Star Metals", "business_address": "Mumbai", "country": "IN"}),

    ("match_numbers_in_address",
     {"business_name": "Cipla Limited", "business_address": "Plot 12 MIDC Kurla Mumbai", "country": "IN"},
     {"business_name": "Cipla Limited", "business_address": "Plot No 12 MIDC Kurla Mumbai", "country": "IN"}),

    ("nonmatch_swapped_digits_address",
     {"business_name": "Asian Paints", "business_address": "Plot 21 Turbhe Navi Mumbai", "country": "IN"},
     {"business_name": "Asian Paints", "business_address": "Plot 12 Turbhe Navi Mumbai", "country": "IN"}),

    ("match_short_acronym_name",
     {"business_name": "ONGC", "business_address": "Dehradun Uttarakhand", "country": "IN"},
     {"business_name": "Oil and Natural Gas Corporation", "business_address": "Dehradun Uttarakhand", "country": "IN"}),

    ("nonmatch_one_field_missing_address",
     {"business_name": "Britannia Industries", "business_address": "Bangalore", "country": "IN"},
     {"business_name": "Britannia Industries", "business_address": "", "country": "IN"}),

    ("match_extra_whitespace",
     {"business_name": "  Dabur   India   Ltd  ", "business_address": "Ghaziabad   UP", "country": "IN"},
     {"business_name": "Dabur India Ltd", "business_address": "Ghaziabad UP", "country": "IN"}),
]


def main():
    header = [
        "label", "lev_name", "lev_addr", "jw_name", "jw_addr",
        "jac_name", "jac_addr", "tfidf_name", "tfidf_addr",
        "tfidf_comb", "country_match", "len_diff_name", "len_diff_addr",
    ]
    print("\t".join(header))

    for label, row1, row2 in FAKE_PAIRS:
        feats = featurize(row1, row2)
        values = [
            label,
            f"{feats['lev_ratio_name']:.3f}",
            f"{feats['lev_ratio_address']:.3f}",
            f"{feats['jw_name']:.3f}",
            f"{feats['jw_address']:.3f}",
            f"{feats['jaccard_name']:.3f}",
            f"{feats['jaccard_address']:.3f}",
            f"{feats['tfidf_cosine_name']:.3f}",
            f"{feats['tfidf_cosine_address']:.3f}",
            f"{feats['tfidf_cosine_combined']:.3f}",
            str(feats['country_match']),
            str(feats['len_diff_name']),
            str(feats['len_diff_address']),
        ]
        print("\t".join(values))

    print()
    print(f"Ran featurize() on {len(FAKE_PAIRS)} hand-made pairs with no crashes.")
    print("Eyeball check: 'match_*' rows should generally score high on "
          "jw/tfidf/jaccard and low on len_diff; 'nonmatch_*' rows should "
          "score lower; 'edge_*' rows just confirm nothing crashes on "
          "empty/degenerate input.")


if __name__ == "__main__":
    main()