# ============================================================
# utils/helpers.py — Shared Functions for All Notebooks
# Emotion-Aware AI Chatbot Project
# ============================================================

import re
import random
import joblib
import os
try:
    from openai import OpenAI
    openai_client = None
except ImportError:
    pass  # We'll fail gracefully if openai is not installed yet
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources (safe to call multiple times)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ─────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────
def clean_text(text):
    """Full NLP preprocessing pipeline."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)             # Remove mentions/hashtags
    text = re.sub(r'[^a-z\s]', '', text)              # Remove punctuation/digits
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)


# ─────────────────────────────────────────
# EMOTION RESPONSE TEMPLATES
# ─────────────────────────────────────────
EMOTION_RESPONSES = {
    'joy': [
        "😊 That's wonderful! I'm so glad you're feeling joyful!",
        "🎉 Happiness looks great on you! What's making you smile?",
        "✨ Your positive energy is contagious! Keep shining!"
    ],
    'sadness': [
        "💙 I'm really sorry you're feeling this way. You're not alone.",
        "🤗 It's okay to feel sad. Would you like to talk about it?",
        "💛 Remember, tough days don't last. Things will get better."
    ],
    'anger': [
        "😤 I hear you — that sounds really frustrating. Take a deep breath.",
        "🔥 Your feelings are valid. Want to tell me more?",
        "💪 It's okay to feel angry. Let's work through this together."
    ],
    'fear': [
        "🫂 It sounds like you're scared. I'm here with you.",
        "🕊️ Fear can feel overwhelming, but you are braver than you think.",
        "💙 Take it one step at a time. What's worrying you?"
    ],
    'surprise': [
        "😲 Wow, that sounds unexpected! How are you feeling about it?",
        "🎊 Surprises can be exciting! Tell me more!",
        "🤩 Life is full of surprises! This one caught you off guard?"
    ],
    'love': [
        "💖 Aww, that's so sweet! Love makes everything brighter.",
        "❤️ What a beautiful feeling! Cherish those moments.",
        "🌹 Love is such a powerful emotion — I'm happy for you!"
    ],
    'neutral': [
        "🙂 Got it! Is there anything on your mind you'd like to share?",
        "💬 I'm here and listening. Tell me more!",
        "😊 Thanks for sharing. What would you like to talk about?"
    ]
}

EMOTION_EMOJIS = {
    'joy': '😊', 'sadness': '😢', 'anger': '😠',
    'fear': '😨', 'surprise': '😲', 'love': '❤️', 'neutral': '😐'
}

EMOTION_MAP = {
    0: 'sadness', 1: 'joy', 2: 'love',
    3: 'anger',   4: 'fear', 5: 'surprise'
}


# ─────────────────────────────────────────
# CHATBOT INFERENCE
# ─────────────────────────────────────────
def load_model_artifacts(model_dir='models/'):
    """Load saved model, vectorizer, and label encoder."""
    model = joblib.load(f'{model_dir}emotion_model.pkl')
    tfidf = joblib.load(f'{model_dir}tfidf_vectorizer.pkl')
    le    = joblib.load(f'{model_dir}label_encoder.pkl')
    return model, tfidf, le


def predict_emotion(text, model, tfidf, le):
    """Predict emotion label and confidence from raw text."""
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0] if hasattr(model, 'predict_proba') else None
    emotion = le.inverse_transform([pred])[0]
    confidence = proba.max() if proba is not None else None
    return emotion, confidence


def chatbot_respond(user_input, model, tfidf, le):
    """Full pipeline: detect emotion → pick empathetic response."""
    global openai_client
    if not user_input.strip():
        return "Please say something!", "unknown", None
    emotion, confidence = predict_emotion(user_input, model, tfidf, le)
    
    # Try calling OpenAI if API key is provided
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        return "⚠️ I cannot provide a ChatGPT-level response because an OpenAI API Key has not been added to your .env file! Please add it to activate ChatGPT responses.", emotion, confidence

    if 'OpenAI' in globals() and openai_client is None:
        if api_key.startswith("sk-or-"):
            openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
        else:
            openai_client = OpenAI(api_key=api_key)
        
    try:
        model_name = "openai/gpt-4o-mini" if api_key.startswith("sk-or-") else "gpt-4o-mini"
        completion = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a highly intelligent, empathetic AI chatbot. Respond naturally and thoughtfully, adjusting your tone to be appropriate for the user's detected emotion."},
                {"role": "user", "content": f"The user says: '{user_input}'. The detected emotion is: '{emotion}'. Your response:"}
            ],
            temperature=0.7,
            max_tokens=250
        )
        reply = completion.choices[0].message.content.strip()
        return reply, emotion, confidence
    except Exception as e:
        return f"⚠️ OpenAI API Error: {str(e)}", emotion, confidence


print("✅ helpers.py loaded successfully!")
