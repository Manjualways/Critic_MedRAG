ENTITY_NORMALIZATION = {

    # Diabetes normalization
    "type 2 diabetes": "Diabetes mellitus",
    "type ii diabetes": "Diabetes mellitus",

    # Common laboratory names
    "bloodglucose": "blood glucose",
    "blood glucose test": "Blood Glucose",

    # CBC normalization
    "complete blood count": "CBC",
    "complete blood count (cbc)": "CBC",

    # Thyroid tests
    "free t4": "Free T4",

    # Cholesterol
    "ldl cholesterol": "LDL Cholesterol",
}


def normalize_entity_name(name: str) -> str:

    cleaned = " ".join(
        name.split()
    ).strip()

    normalized = ENTITY_NORMALIZATION.get(
        cleaned.lower()
    )

    if normalized:
        return normalized

    return cleaned