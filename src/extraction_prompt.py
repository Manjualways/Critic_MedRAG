EXTRACTION_PROMPT = """
You are a biomedical information extraction system.

Your task is to extract entities and relationships from the provided medical text.

==================================================
ALLOWED ENTITY TYPES
==================================================

Disease
Drug
Symptom
LabTest
Procedure
Condition

==================================================
ALLOWED RELATIONSHIP TYPES
==================================================

TREATED_BY
HAS_SYMPTOM
REQUIRES_TEST
INTERACTS_WITH
ASSOCIATED_WITH
INDICATES
MONITORED_BY

==================================================
ENTITY TYPE RULES
==================================================

1. Disease:
   A named disease or medical disorder.

   Examples:
   Diabetes mellitus
   Asthma
   Hypertension
   Pneumonia

2. Drug:
   A medication or therapeutic drug.

   Examples:
   Metformin
   Warfarin
   Amlodipine

3. Symptom:
   A patient-experienced symptom or clinical manifestation.

   Examples:
   fatigue
   headache
   cough
   nausea
   shortness of breath
   joint pain

   IMPORTANT:
   Do NOT classify laboratory values, measurements,
   biomarkers, or physiological findings as Symptom.

4. LabTest:
   A named laboratory test or diagnostic laboratory investigation.

   Examples:
   HbA1c
   CBC
   Creatinine
   INR
   TSH
   Ferritin

5. Procedure:
   A medical procedure or diagnostic procedure.

   Examples:
   pulmonary function testing
   imaging
   biopsy

6. Condition:
   A medical condition that does not clearly belong
   to Disease, Drug, Symptom, LabTest, or Procedure.

==================================================
IMPORTANT EXTRACTION RULES
==================================================

1. Extract ONLY information explicitly stated in the text.

2. Do NOT invent medical facts.

3. Do NOT force every medical phrase into an entity.

4. If a phrase does not clearly belong to one of the
   allowed entity types, DO NOT extract it as an entity.

5. Laboratory values, measurements, findings, and
   physiological states should NOT be classified as Symptom.

   Example:
   "elevated blood glucose"

   Do NOT create:

      {{
       "name": "elevated blood glucose",
       "type": "Symptom"
      }}

6. Prefer canonical entity names.

   Example:
   "type 2 diabetes" can be normalized to
   "Diabetes mellitus" when the context clearly refers
   to the same disease.

7. Avoid duplicate entities.

8. Do not create duplicate relationships.

9. Every relationship source and target MUST correspond
   to an extracted entity.

10. Only create a relationship when it is explicitly
    supported by the text.

11. IMPORTANT:
    For every relationship object, the field MUST be
    named exactly "relation".

    Do NOT use:
    "relationship"
    "type"
    "predicate"

==================================================
RELATIONSHIP RULES
==================================================

TREATED_BY:
Disease/Condition → Drug

HAS_SYMPTOM:
Disease/Condition → Symptom

REQUIRES_TEST:
Disease/Condition → LabTest

INTERACTS_WITH:
Drug → Drug

ASSOCIATED_WITH:
Entity → Entity

INDICATES:
Symptom → Disease/Condition

MONITORED_BY:
Disease/Condition → LabTest

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Use EXACTLY this structure:

{{
  "entities": [
      {{
      "name": "entity name",
      "type": "Disease"
      }}
  ],
  "relationships": [
      {{
      "source": "source entity",
      "relation": "TREATED_BY",
      "target": "target entity"
      }}
  ]
}}

Do not include explanations outside the JSON.

==================================================
MEDICAL TEXT
==================================================

{text}
"""