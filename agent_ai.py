import os
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Gemini client
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

client = genai.Client(api_key=api_key)

def start_chatbot():
    print("🤖 Agentic AI Chatbot (Complete Edition)")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 30)

    # Initialize chat history
    chat = client.chats.create(model="gemini-flash-lite-latest")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break
        
        if not user_input.strip():
            continue

        try:
            print("AI is thinking...", end="\r")
            response = chat.send_message(user_input)
            print(" " * 20, end="\r") # Clear the thinking line
            print(f"AI: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    start_chatbot()
