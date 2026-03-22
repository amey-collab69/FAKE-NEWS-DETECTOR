import os
import re
import pickle
import logging
import nltk
from flask import Flask, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data on startup
for pkg in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
loaded_model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), 'rb'))
vector = pickle.load(open(os.path.join(BASE_DIR, "vector.pkl"), 'rb'))

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))


def preprocess(text):
    """Clean and preprocess input news text."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stpwrds]
    return ' '.join(tokens)


def predict_news(news):
    """Return prediction label and result string for given news text."""
    processed = preprocess(news)
    vectorized = vector.transform([processed])
    prediction = loaded_model.predict(vectorized)
    is_fake = int(prediction[0]) == 1
    result = "⚠️ This looks like Fake News 📰" if is_fake else "✅ This looks like Real News 📰"
    return result, is_fake


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = None
    is_fake = None
    error = None

    if request.method == 'POST':
        news = request.form.get('news', '').strip()
        if not news:
            error = "Please enter a news headline or article text."
        elif len(news) < 5:
            error = "Input is too short. Please enter a meaningful news text."
        else:
            try:
                prediction_text, is_fake = predict_news(news)
                logger.info("Prediction made: %s", prediction_text)
            except Exception as e:
                logger.error("Prediction error: %s", e)
                error = "Something went wrong during prediction. Please try again."

    return render_template('prediction.html',
                           prediction_text=prediction_text,
                           is_fake=is_fake,
                           error=error)


if __name__ == '__main__':
    app.run(debug=False)
