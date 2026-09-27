import re
import unicodedata
from unidecode import unidecode


# Common business/legal suffixes.
# These are removed only when they appear as separate words,
# mainly from the end of the business name.
LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "ltd",
    "limited",
    "plc",
    "pvt",
    "private",
    "llp",
    "sarl",
    "sas",
    "gmbh",
    "ag",
    "sa",
}


def clean_unicode(text):
    """
    Normalize Unicode characters and transliterate non-Latin scripts.
    Example:
        Hindi/Devanagari -> Latin representation
        accented characters -> normalized representation
    """
    if text is None:
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Transliterate non-Latin characters locally.
    text = unidecode(text)

    return text


def normalize_text(text):
    """
    General text normalization.

    Steps:
    1. Handle missing values.
    2. Unicode normalization.
    3. Transliteration.
    4. Lowercase.
    5. Replace punctuation with spaces.
    6. Collapse repeated whitespace.
    """
    if text is None:
        return ""

    text = str(text).strip()

    if not text or text.lower() == "nan":
        return ""

    text = clean_unicode(text)

    text = text.lower()

    # Replace punctuation/symbols with spaces.
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Remove repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_legal_suffixes(name):
    """
    Remove common legal/business suffixes from the END of a normalized name.
    """
    if not name:
        return ""

    tokens = name.split()

    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()

    return " ".join(tokens)


def normalize_business_name(name):
    """
    Normalize a business name for entity resolution.
    """
    name = normalize_text(name)

    if not name:
        return ""

    name = remove_legal_suffixes(name)

    return name


def normalize_address(address):
    """
    Normalize a business address.

    This does not try to understand the geographic meaning
    of the address. It only standardizes the text.
    """
    address = normalize_text(address)

    if not address:
        return ""

    # Common address abbreviations.
    replacements = {
        " street ": " st ",
        " road ": " rd ",
        " avenue ": " ave ",
        " boulevard ": " blvd ",
        " drive ": " dr ",
        " lane ": " ln ",
        " highway ": " hwy ",
        " apartment ": " apt ",
        " suite ": " ste ",
    }

    # Add spaces around the address so replacement works at boundaries.
    address = f" {address} "

    for old, new in replacements.items():
        address = address.replace(old, new)

    address = re.sub(r"\s+", " ", address).strip()

    return address


def normalize_country(country):
    """
    Normalize country values while preserving the country information.
    """
    if country is None:
        return ""

    country = str(country).strip().lower()

    if country == "nan":
        return ""

    return country


def normalize_record(entity_id, business_name, business_address, country):
    """
    Normalize one complete entity record.
    """
    return {
        "entity_id": str(entity_id),
        "business_name": normalize_business_name(business_name),
        "business_address": normalize_address(business_address),
        "country": normalize_country(country),
    }