import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.normalization import normalize_record


test_records = [
    {
        "entity_id": "S2-TEST-1",
        "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
        "business_address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "country": "India",
    },
    {
        "entity_id": "S2-TEST-2",
        "business_name": "Prime Money Corp.",
        "business_address": "17560 Ellis Road, Tahlequah, OK",
        "country": "US",
    },
    {
        "entity_id": "S3-TEST-3",
        "business_name": "Fractales Amis Groupe S.A.S",
        "business_address": "23 Rue Icmre, La Teste-de-buch, Gironde",
        "country": "France",
    },
    {
        "entity_id": "S3-TEST-4",
        "business_name": None,
        "business_address": None,
        "country": "India",
    },
]


for record in test_records:
    print("\n" + "=" * 70)

    print("ORIGINAL:")
    print(record)

    normalized = normalize_record(
        record["entity_id"],
        record["business_name"],
        record["business_address"],
        record["country"],
    )

    print("\nNORMALIZED:")
    print(normalized)