import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import pickle
import os
from transformers import pipeline

from google import genai
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- LOAD MODELS ---

# 1. Flower Classifier
FLOWER_MODEL_PATH = 'flower_classifier_model.h5'
flower_model = None
if os.path.exists(FLOWER_MODEL_PATH):
    flower_model = tf.keras.models.load_model(FLOWER_MODEL_PATH)
    print("✅ Flower Model Loaded.")

FLOWER_CLASSES = ['Bougainvillea', 'Daisies', 'Tulip']

# 2. Sentiment Analysis (Using YOUR local model)
sentiment_model = None
vectorizer = None
if os.path.exists('sentiment_model.pkl'):
    with open('sentiment_model.pkl', 'rb') as f:
        sentiment_model = pickle.load(f)
    with open('sentiment_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Local Sentiment Model Loaded.")

# 3. Agentic AI (Gemini)
agent_client = None
if os.getenv("GEMINI_API_KEY"):
    agent_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("✅ Agentic AI (Gemini) Connected.")

# 4. In-Memory Activity Logs
activity_logs = []

def log_activity(data):
    data['timestamp'] = datetime.utcnow().isoformat()
    data['id'] = f"log_{len(activity_logs)}_{datetime.utcnow().timestamp()}"
    activity_logs.insert(0, data)
    if len(activity_logs) > 10:
        activity_logs.pop()

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/content', methods=['GET'])
def get_content():
    # Return defaults
    default_content = {
        "nav_logo": "AI<span>HUB</span>",
        "nav_link_1": "Ecosystem",
        "nav_link_2": "Contact",
        "hero_title": "Intelligence <br><span class=\"gradient-text\">Redefined.</span>",
        "hero_subtitle": "Experience the convergence of Computer Vision, Natural Language Processing, and Advanced Analytics in one unified ecosystem.",
        "hero_btn_1": "Explore Models",
        "hero_btn_2": "Repository",
        "ecosystem_title": "Our AI Ecosystem",
        "ecosystem_subtitle": "Three powerful models working in harmony to solve complex problems.",
        "vision_title": "Vision AI",
        "vision_desc": "Deep learning neural network trained to classify 3 distinct flower classes with high precision.",
        "vision_btn": "Select Image",
        "linguistic_title": "Linguistic AI",
        "linguistic_desc": "Context-aware sentiment engine designed to extract emotional intelligence from unstructured text.",
        "linguistic_placeholder": "Paste analytical data or text here...",
        "linguistic_btn": "Run Analysis",
        "agentic_title": "Agentic AI",
        "agentic_desc": "Autonomous intelligence layer capable of complex reasoning and context-aware interactions.",
        "agentic_btn": "Deploy Assistant",
        "history_title": "Real-time Activity",
        "history_subtitle": "Live feed of server-side logs and model inferences.",
        "footer_text": "&copy; 2026 Ayesha Naveed. All rights reserved."
    }
    return jsonify(default_content)

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(activity_logs)

@app.route('/predict_flower', methods=['POST'])
def predict_flower():
    if not flower_model:
        return jsonify({'error': 'Flower model not trained yet.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = flower_model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx])
    
    result = {
        'class': FLOWER_CLASSES[class_idx],
        'confidence': f"{confidence*100:.2f}%"
    }

    # Log Activity
    log_activity({
        'type': 'flower_classification',
        'input': file.filename,
        'result': result['class'],
        'confidence': result['confidence']
    })

    return jsonify(result)

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    if not sentiment_model or not vectorizer:
        return jsonify({'error': 'Sentiment model not loaded.'}), 400
    
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Preprocess and Predict
    tfidf_text = vectorizer.transform([text])
    prediction = sentiment_model.predict(tfidf_text)[0]
    
    result = {
        'label': str(prediction),
        'score': "N/A"
    }

    # Log Activity
    log_activity({
        'type': 'sentiment_analysis',
        'input': text[:100] + '...' if len(text) > 100 else text,
        'result': result['label']
    })

    return jsonify(result)

# --- CHAT SESSIONS ---
chat_sessions = {}

@app.route('/ask_agent', methods=['POST'])
def ask_agent():
    if not agent_client:
        return jsonify({'error': 'Agentic AI API key not configured.'}), 400
    
    data = request.json
    prompt = data.get('prompt', '')
    session_id = data.get('session_id', 'default') # Simple session tracking
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Create or retrieve chat session
        if session_id not in chat_sessions:
            chat_sessions[session_id] = agent_client.chats.create(
                model="gemini-flash-lite-latest",
                config={"system_instruction": "You are a highly professional AI Consultant. Your responses must be structured with **bold headings**, bullet points where appropriate, and a sophisticated tone. Always aim for clarity and a premium feel in your writing."}
            )
        
        chat = chat_sessions[session_id]
        response = chat.send_message(prompt)
        
        # Log Activity
        log_activity({
            'type': 'agent_chat',
            'input': prompt[:100],
            'result': 'Response generated'
        })

        return jsonify({
            'response': response.text
        })
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in /ask_agent: {error_msg}")
        # If session fails (e.g. timeout/expired), reset it
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        
        if "503" in error_msg or "UNAVAILABLE" in error_msg or "high demand" in error_msg.lower():
            user_msg = "The AI model is currently experiencing high demand. Please try again in a few moments."
            return jsonify({'error': user_msg}), 503
            
        return jsonify({'error': "An internal error occurred. Please try again."}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"\n🚀 AI Portfolio running at: http://127.0.0.1:{port}")
    print(f"👉 Open this link in your browser to see the frontend!\n")
    app.run(debug=True, port=port)
