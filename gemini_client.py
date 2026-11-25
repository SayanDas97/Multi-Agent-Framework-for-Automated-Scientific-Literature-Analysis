
import time
from google import genai


def generate_text(api_key, model_name, prompt):
    """
    Generate text using the Gemini API.
    Retries temporary server and rate-limit errors.
    """

    client = genai.Client(api_key=api_key)

    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if not response or not response.text:
                raise ValueError("Gemini returned an empty response.")

            return response.text.strip()

        except Exception as exc:
            error_message = str(exc)
            error_lower = error_message.lower()

            temporary_error = (
                "503" in error_message
                or "unavailable" in error_lower
                or "429" in error_message
                or "resource_exhausted" in error_lower
                or "500" in error_message
                or "internal server error" in error_lower
            )

            if temporary_error and attempt < max_attempts - 1:
                wait_time = 2 ** (attempt + 1)
                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
                continue

            raise RuntimeError(
                f"Gemini API request failed: {error_message}"
            ) from exc

    raise RuntimeError("Gemini API request failed after all retry attempts.")
