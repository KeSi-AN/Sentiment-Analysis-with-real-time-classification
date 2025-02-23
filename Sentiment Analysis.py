import nltk
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import opinion_lexicon, stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Download necessary resources
nltk.download('vader_lexicon', force=True)
nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)
nltk.download('wordnet', force=True)
nltk.download('opinion_lexicon', force=True)
nltk.download('averaged_perceptron_tagger', force=True)

# Load dataset
columns = ['label', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv(r"C:\Users\addya\Documents\training.1600000.processed.noemoticon.csv", 
                 encoding='latin-1', names=columns, header=None, nrows=1000)

# Remove old labels and keep only text
df = df[['text']]

# Initialize Sentiment Analyzers
sia = SentimentIntensityAnalyzer()
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Function to label sentiment using VADER, TextBlob & POS-based rules
def assign_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    blob_score = TextBlob(text).sentiment.polarity
    avg_score = (score + blob_score) / 2  # Combine both for better accuracy

    # POS tagging for deeper sentiment analysis
    words = word_tokenize(text.lower())
    pos_tags = pos_tag(words)
    sentiment_score = 0
    
    for word, tag in pos_tags:
        if word in positive_words:
            sentiment_score += 1
        elif word in negative_words:
            sentiment_score -= 1

    # Final sentiment decision
    final_score = avg_score + (sentiment_score / len(words) if words else 0)
    return 1 if final_score > 0 else 0

# Apply auto-labeling
df['label'] = df['text'].apply(assign_sentiment)

# Check class distribution
print("New class distribution:\n", df['label'].value_counts())

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# XGBoost Model
xgb = XGBClassifier(eval_metric='logloss')

# Hyperparameter tuning
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1]
}
grid = GridSearchCV(xgb, param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# Best model evaluation
best_xgb = grid.best_estimator_
y_pred = best_xgb.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Function for real-time tweet classification
def classify_tweet(tweet):
    processed_tweet = preprocess_text(tweet)
    vectorized_tweet = vectorizer.transform([processed_tweet]).toarray()
    prediction = best_xgb.predict(vectorized_tweet)[0]
    return "Positive" if prediction == 1 else "Negative"

# Real-time user input classification
while True:
    user_input = input("Enter a tweet (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("Sentiment:", classify_tweet(user_input))
