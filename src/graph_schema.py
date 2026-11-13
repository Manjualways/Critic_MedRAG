NODE_TYPES = {
    "Disease", "Symptom", "Drug", "LabTest", "Doctor", "Patient"
}

RELATIONSHIP_TYPES = {
    "TREATED_BY", "INTERACTS_WITH", "COMORBID_WITH",
    "INDICATES", "REQUIRES_TEST"
}

def normalize_label(value: str) -> str:
    aliases = {
        "disease": "Disease", "condition": "Disease",
        "symptom": "Symptom",
        "drug": "Drug", "medication": "Drug",
        "labtest": "LabTest", "lab_test": "LabTest", "test": "LabTest",
        "doctor": "Doctor", "patient": "Patient",
    }
    return aliases.get(value.strip().lower(), value.strip())

def normalize_relation(value: str) -> str:
    value = value.strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "TREATS": "TREATED_BY",
        "TREATEDBY": "TREATED_BY",
        "INTERACTS": "INTERACTS_WITH",
        "INTERACTION": "INTERACTS_WITH",
        "COMORBID": "COMORBID_WITH",
        "REQUIRES": "REQUIRES_TEST",
    }
    return aliases.get(value, value)
