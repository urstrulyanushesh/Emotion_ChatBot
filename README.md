<div align="center">
  <h1>🧠 Emotion-Aware AI Chatbot</h1>
  <p>An intelligent, context-aware chatbot that detects emotional tone using a custom Machine Learning pipeline and replies empathetically via a Large Language Model (ChatGPT).</p>
</div>

---

## 🌟 Overview

The **Emotion-Aware AI Chatbot** is a hybrid AI implementation. Rather than relying purely on LLM prompting, it employs a **local Machine Learning model (Scikit-Learn/TF-IDF)** to explicitly classify the user's emotion (Joy, Sadness, Anger, Fear, Surprise, Love, Neutral). 

Once the emotional context is identified, the application leverages the **OpenAI / OpenRouter API** to generate a highly dynamic, context-aware, and empathetic response tailored to that exact feeling. 

## ⚙️ Key Technologies

- **Backend:** Python 3.12, Flask 
- **Machine Learning Engine:** Scikit-Learn (TF-IDF Vectorizer, Classification Model)
- **NLP Preprocessing:** NLTK (Stopwords removal, Lemmatization, Tokenization)
- **Generative AI:** `openai` Python SDK (Supports strictly OpenAI or OpenRouter routing)
- **Frontend UI:** HTML5, Vanilla CSS, JavaScript

---

## 📂 Project Structure

```text
emotion_chatbot/
├── data/                       # Raw and processed datasets
│   ├── emotion_data.csv        # Primary training data
│   ├── cleaned_data.csv        # Preprocessed data subsets
│   └── X_train.pkl, y_train.pkl# Serialized train/test splits
│
├── models/                     # Compiled Machine Learning artifacts
│   ├── emotion_model.pkl       # The trained classification model
│   ├── tfidf_vectorizer.pkl    # Vectorizer for text transformation
│   └── label_encoder.pkl       # Decoder for emotion integer labels
│
├── src/                        # Core application and notebooks
│   ├── app.py                  # The main Flask application entrypoint
│   ├── requirements.txt        # PIP dependency list
│   └── *.ipynb                 # Jupyter Notebooks documenting the full ML pipeline 
│       (01_data_collection, 02_preprocessing, 03_eda, etc.)
│
├── static/                     # Web assets
│   └── css/style.css           # Frontend application styling
│
├── templates/                  # HTML views
│   └── index.html              # The frontend chatbot interface
│
└── utils/                      # Helper modules
    └── helpers.py              # NLP cleaning algorithms and API request logic
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
Open your terminal and pull down the project repository:
```bash
git clone https://github.com/urstrulyanushesh/Emotion_ChatBot.git
cd Emotion_ChatBot
```

### 2. Create the Virtual Environment (Optional but Recommended)
For best practices, create an isolated Python environment for your dependencies:
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all the necessary backend packages required to run the Machine Learning and Flask server:
```bash
pip install -r src/requirements.txt
```
*(Note: Ensure you include `openai`, `python-dotenv`, `joblib`, `flask`, `nltk`, and `scikit-learn` in your environment.)*

### 4. Provide your API Key
You must securely provide an API key to enable the intelligent Chatbot responses. 
1. Create a hidden file named `.env` in the root directory.
2. Inside `.env`, paste your OpenAI (or OpenRouter) API key like this:

```env
OPENAI_API_KEY=sk-your_api_key_here
```
> [!NOTE]
> If your API key begins with `sk-or-v1-`, the system's `helpers.py` will automatically recognize it as an OpenRouter key, adjust the `base_url`, and route requests through OpenRouter's `gpt-4o-mini` API endpoint safely!

---

## 💻 Running the Application

Once your `.env` is set up and dependencies are installed, you are ready to boot up the backend:

1. Use your terminal (ensuring you are in the project's root folder) to run:
```bash
python src/app.py
```
2. The terminal will log that `helpers.py loaded successfully!` and Flask is active.
3. Open your favorite web browser and navigate to:
   **http://127.0.0.1:5000**

You can now interact with the Chatbot directly!

---

## 🧠 How the Architecture Works

1. **User Input:** The user types a message in the frontend chat UI.
2. **Preprocessing:** `app.py` passes the text into `utils/helpers.py`. The text is cleaned stripped of punctuation, lemmatized, and converted to lower case.
3. **Classification:** The raw string is fed into the local pre-trained TF-IDF vectorizer and evaluated by the Scikit-Learn `emotion_model.pkl` to yield an integer label.
4. **Decoding:** The label encoder maps the integer to a core emotion (e.g., `Joy`).
5. **Generation Pipeline:** The prompt is dynamically crafted containing both the user's string and the detected emotion. This is sent to OpenAI's ChatGPT.
6. **Delivery:** The chatbot intelligently outputs an empathetic, ChatGPT-level response tailored uniquely to the user's emotional state.
