import nltk
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Download necessary NLP resources
nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)
nltk.download('wordnet', force=True)

# Load dataset
columns = ['label', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv("/content/training.1600000.processed.noemoticon.csv", 
                 encoding='latin-1', names=columns, header=None)

# Convert labels (0 = Negative, 4 = Positive) to binary (0, 1)
df['label'] = df['label'].replace(4, 1)

# Ensure balanced dataset (500 positive + 500 negative samples)
negative_samples = df[df['label'] == 0].sample(n=500, random_state=42)
positive_samples = df[df['label'] == 1].sample(n=500, random_state=42)
df = pd.concat([negative_samples, positive_samples]).reset_index(drop=True)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['label']

# Train-test split (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# XGBoost Classifier
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.2,
    reg_lambda=1.0,
    reg_alpha=0.5,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

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

# Function to classify new input
def classify_tweet(tweet):
    processed_tweet = preprocess_text(tweet)
    vectorized_tweet = vectorizer.transform([processed_tweet]).toarray()
    prediction = best_xgb.predict(vectorized_tweet)[0]
    return "Positive Tweet" if prediction == 1 else "Negative Tweet"

# Take user input
while True:
    user_input = input("Enter a tweet (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("Sentiment:", classify_tweet(user_input))



