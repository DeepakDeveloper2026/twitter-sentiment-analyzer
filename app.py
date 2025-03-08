from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import os

NLTK_DIR = "/opt/render/project/src/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)
nltk.download('punkt', download_dir=NLTK_DIR)


model_path = "sentiment_model.pkl"
vectorizer_path = "vectorizer.pkl"

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__, static_folder="static", template_folder="templates")


def preprocess_text(text):
    return text.lower()


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    sentiment_class = ""  

    if request.method == "POST":
        text = request.form.get("tweet")
        if text:
            text_vectorized = vectorizer.transform([preprocess_text(text)])
            sentiment = model.predict(text_vectorized)[0]

            
            sentiment=sentiment.lower()
            if sentiment == "positive":
                sentiment_class = "positive"
            elif sentiment == "negative":
                sentiment_class = "negative"
            elif sentiment == "neutral":
                sentiment_class = "neutral"
            elif sentiment == "irrelevant":
                sentiment_class = "irrelevant"
            print("DEBUG: Sentiment =", sentiment)
            print("DEBUG: Sentiment Class =", sentiment_class)
    
    return render_template("index.html", sentiment=sentiment, sentiment_class=sentiment_class)

import os
port = int(os.environ.get("PORT", 5000)) 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)

