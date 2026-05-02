import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the new client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Generate response
response = client.models.generate_content(
    model="gemini-flash-lite-latest",
    contents="Explain how AI works in urdu"
)

# Output the text and save to a file
print(response.text)
with open("ai_explanation_urdu.txt", "w", encoding="utf-8") as f:
    f.write(response.text)
print("\nExplanation saved to ai_explanation_urdu.txt")
