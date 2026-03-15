import os
from google import genai
import sys

key = "AIzaSyCejk-stmvq2soStSONAxsIHcGJnUZZZXM"
try:
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Hello, how are you?"
    )
    print("SUCCESS: ", response.text)
except Exception as e:
    import traceback
    traceback.print_exc()
