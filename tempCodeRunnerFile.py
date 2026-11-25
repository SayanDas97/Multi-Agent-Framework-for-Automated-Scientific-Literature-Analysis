
import time
from google import genai


def generate_text(api_key, model_name, prompt):

    client = genai.Client(api_key=api_key)

    max_attempts = 3

    for attempt in range(max_attempts):

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response.text

        except Exception as exc:

            error_message = str(exc).lower()

            temporary_error = any(
                term in error_message
                for term in [
                    "503",
                    "unavailable",
                    "high demand",
                    "500",
                    "502",
                    "504",
                    "429",
                    "resource_exhausted"
                ]
            )

            if not temporary_error or attempt == max_attempts - 1:
                raise

            wait_time = 2 ** (attempt + 1)

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Gemini could not complete the request after retries."
    )