from collections import defaultdict


def make_name_ngrams(name, n=3):
    """
    Create character n-grams from a normalized business name.
    """
    if not name:
        return set()

    # Add boundaries so beginnings/endings carry information.
    text = f"  {name}  "

    if len(text) < n:
        return {text}

    return {
        text[i:i + n]
        for i in range(len(text) - n + 1)
    }


def make_phonetic_key(name):
    """
    Simple phonetic-style signature.

    This is intentionally lightweight and deterministic.
    It is NOT intended to replace the later similarity model.
    """
    if not name:
        return ""

    words = name.split()

    signatures = []

    for word in words:
        if not word:
            continue

        word = word.lower()

        # Keep first character.
        first = word[0]

        # Remove common vowels after the first character.
        remainder = "".join(
            char for char in word[1:]
            if char not in "aeiou"
        )

        # Keep a short signature.
        signature = first + remainder[:4]

        signatures.append(signature)

    return " ".join(signatures)


def make_blocking_keys(record):
    """
    Generate multiple blocking keys for one normalized record.

    The keys are designed to create candidate blocks rather than
    make the final entity-resolution decision.
    """

    country = record.get("country", "")
    name = record.get("business_name", "")
    address = record.get("business_address", "")

    keys = set()

    # ---------------------------------------------------------
    # 1. Country + first characters of business name
    # ---------------------------------------------------------
    if country and name:
        words = name.split()

        if words:
            first_word = words[0]

            for length in (2, 3, 4):
                if len(first_word) >= length:
                    keys.add(
                        f"country_nameprefix:{country}:{first_word[:length]}"
                    )

    # ---------------------------------------------------------
    # 2. Country + character n-grams
    # ---------------------------------------------------------
    if country and name:
        ngrams = make_name_ngrams(name, n=3)

        for gram in ngrams:
            keys.add(
                f"country_namegram:{country}:{gram}"
            )

    # ---------------------------------------------------------
    # 3. Country + phonetic signature
    # ---------------------------------------------------------
    if country and name:
        phonetic = make_phonetic_key(name)

        if phonetic:
            keys.add(
                f"country_phonetic:{country}:{phonetic}"
            )

    # ---------------------------------------------------------
    # 4. Country + address tokens
    # ---------------------------------------------------------
    if country and address:
        address_tokens = address.split()

        # Use informative address tokens.
        for token in address_tokens:
            if len(token) >= 4:
                keys.add(
                    f"country_address:{country}:{token}"
                )

    return keys


def build_block_index(records):
    """
    Build an inverted index:

        blocking_key -> set(entity_id)

    This lets us retrieve only records sharing at least one
    blocking key instead of comparing every record with every
    other record.
    """

    index = defaultdict(set)

    for record in records:
        entity_id = record["entity_id"]

        keys = make_blocking_keys(record)

        for key in keys:
            index[key].add(entity_id)

    return index


def get_candidates(record, block_index):
    """
    Retrieve candidate entity IDs for one source-1 record.
    """

    candidates = set()

    keys = make_blocking_keys(record)

    for key in keys:
        candidates.update(block_index.get(key, set()))

    return candidates