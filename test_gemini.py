
from google import genai
from getpass import getpass

print("Gemini API connection test")

api_key = getpass("Enter your Gemini API key: ")

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with only the word: SUCCESS"
    )

    print("\nGemini response:")
    print(response.text)

except Exception as e:
    print("\nGemini API test failed:")
    print(type(e).__name__)
    print(str(e))
