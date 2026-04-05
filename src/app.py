from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv, find_dotenv
import sys
import os

# Discover the root directory of the project (one level up from src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the .env from the root directory
load_dotenv(find_dotenv(os.path.join(BASE_DIR, '.env')), override=True)

# Add the root directory/utils to sys.path
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from helpers import load_model_artifacts, chatbot_respond

# Explicitly tell Flask where the templates and static files are located
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# Load model artifacts with an absolute path ending in a slash
model, tfidf, le = load_model_artifacts(os.path.join(BASE_DIR, 'models') + os.sep)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_text = data.get('text', '')
    
    # This logic calls your ML model and returns (reply_text, emotion_label, confidence)
    reply, emotion, conf = chatbot_respond(user_text, model, tfidf, le)
    
    return jsonify({
        "reply": reply,
        "emotion": emotion,
        "confidence": f"{conf:.1%}"
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)