import os
from firebase_admin import credentials, firestore, initialize_app
from dotenv import load_dotenv

load_dotenv()

def seed_data():
    cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not cred_path or not os.path.exists(cred_path):
        print("❌ Error: FIREBASE_SERVICE_ACCOUNT environment variable not set or file not found.")
        return

    cred = credentials.Certificate(cred_path)
    initialize_app(cred)
    db = firestore.client()

    content = {
        "hero_title": "Intelligence <br><span class='gradient-text'>Redefined.</span>",
        "hero_subtitle": "Experience the convergence of Computer Vision, Natural Language Processing, and Advanced Analytics in one unified ecosystem.",
        "vision_title": "Vision AI",
        "vision_desc": "Neural network optimized for high-precision botanical classification and visual analysis.",
        "linguistic_title": "Linguistic AI",
        "linguistic_desc": "Context-aware sentiment engine designed to extract emotional intelligence from unstructured text.",
        "agentic_title": "Agentic AI",
        "agentic_desc": "Autonomous intelligence layer capable of complex reasoning and context-aware interactions."
    }

    print("🚀 Seeding landing page content to Firestore...")
    db.collection('app_config').document('landing_page').set(content)
    print("✅ Content seeded successfully!")

if __name__ == "__main__":
    seed_data()
