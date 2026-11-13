from src.llm_extractor import MedicalExtractor
from src.validate_extraction import validate_extraction


medical_text = """
Diabetes mellitus is associated with elevated blood glucose.
Metformin is commonly used for type 2 diabetes.
Metformin interacts with Warfarin.
HbA1c and blood glucose tests can be used to assess
glucose-related status.
"""


extractor = MedicalExtractor()

raw_result = extractor.extract(medical_text)

validated_result = validate_extraction(
    raw_result
)

print("\n==============================")
print("VALIDATED EXTRACTION")
print("==============================")

print(
    validated_result.model_dump_json(
        indent=2
    )
)