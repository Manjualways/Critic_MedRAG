import os
import json
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from src.extraction_prompt import EXTRACTION_PROMPT


load_dotenv()


class MedicalExtractor:

    def __init__(self):

        self.api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.5-flash-lite"
        )

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing from .env"
            )

        self.client = genai.Client(
            api_key=self.api_key
        )


    def extract(self, medical_text):

        prompt = EXTRACTION_PROMPT.format(
            text=medical_text
        )

        max_retries = 4

        for attempt in range(max_retries):

            try:

                print(
                    f"Gemini extraction attempt "
                    f"{attempt + 1}/{max_retries}..."
                )

                response = (
                    self.client.models.generate_content(
                        model=self.model,
                        contents=prompt
                    )
                )

                content = response.text.strip()

                # Remove Markdown JSON fences
                if content.startswith("```json"):

                    content = content[7:]

                elif content.startswith("```"):

                    content = content[3:]

                if content.endswith("```"):

                    content = content[:-3]

                content = content.strip()

                try:

                    return json.loads(content)

                except json.JSONDecodeError as e:

                    print(
                        "Gemini returned invalid JSON."
                    )

                    print(content)

                    raise ValueError(
                        f"Invalid JSON returned by Gemini: {e}"
                    )


            except errors.ServerError as e:

                print(
                    f"Gemini server error: {e}"
                )

                if attempt == max_retries - 1:

                    raise

                wait_time = 2 ** attempt

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )


        raise RuntimeError(
            "Gemini extraction failed after retries."
        )