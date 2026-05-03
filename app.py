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

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

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

    return jsonify({
        'class': FLOWER_CLASSES[class_idx],
        'confidence': f"{confidence*100:.2f}%"
    })

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
    
    return jsonify({
        'label': str(prediction),
        'score': "N/A"
    })

@app.route('/ask_agent', methods=['POST'])
def ask_agent():
    if not agent_client:
        return jsonify({'error': 'Agentic AI API key not configured.'}), 400
    
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Using gemini-flash-lite-latest for optimal performance and compatibility
        response = agent_client.models.generate_content(
            model="gemini-flash-lite-latest", 
            contents=prompt
        )
        return jsonify({
            'response': response.text
        })
    except Exception as e:
        print(f"❌ Error in /ask_agent: {str(e)}") # Log to terminal for debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    print(f"\n🚀 AI Portfolio running at: http://127.0.0.1:{port}")
    print(f"👉 Open this link in your browser to see the frontend!\n")
    app.run(debug=True, port=port)
