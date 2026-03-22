"""
Retrains the Passive Aggressive Classifier on dataset/train.csv
and saves fresh model.pkl and vector.pkl compatible with the
currently installed scikit-learn version.
"""

import os
import re
import pickle
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data
for pkg in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET  = os.path.join(BASE_DIR, 'dataset', 'train.csv')


def preprocess(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    # Use split() instead of nltk.word_tokenize for speed during training
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stpwrds]
    return ' '.join(tokens)


print("Loading dataset...")
df = pd.read_csv(DATASET)

# Drop rows with missing text or label
df = df.dropna(subset=['text', 'label'])

print(f"Dataset shape: {df.shape}")
print("Preprocessing text (this may take a minute)...")

df['clean'] = df['text'].apply(preprocess)

X = df['clean']
y = df['label'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print("Training Passive Aggressive Classifier...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# Save
model_path  = os.path.join(BASE_DIR, 'model.pkl')
vector_path = os.path.join(BASE_DIR, 'vector.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(vector_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"\nSaved model  → {model_path}")
print(f"Saved vector → {vector_path}")
print("Done. Restart app.py to use the new model.")
