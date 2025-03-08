from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained model and vectorizer
model_path = "sentiment_model.pkl"
vectorizer_path = "vectorizer.pkl"

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__, static_folder="static", template_folder="templates")

# Preprocess text function
def preprocess_text(text):
    return text.lower()

# Route to serve the frontend
@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    sentiment_class = ""  # CSS class for styling

    if request.method == "POST":
        text = request.form.get("tweet")
        if text:
            text_vectorized = vectorizer.transform([preprocess_text(text)])
            sentiment = model.predict(text_vectorized)[0]

            # Assign CSS classes for different sentiments, including "irrelevant"
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

if __name__ == "__main__":
    app.run(debug=True)
