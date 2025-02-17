import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load the trained model
model = tf.keras.models.load_model("your_model.h5")

# Load the tokenizer
with open("your_tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

max_len = 100  # Adjust to match training max length

def preprocess_text(text):
    """Tokenizes and pads the input text."""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    return padded_sequence

def predict_sentiment(text):
    """Predicts sentiment from input text."""
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)[0][0]  # Get the float value

    # Convert probability to sentiment label
    if prediction > 0.6:
        sentiment_label = "Positive"
    elif prediction < 0.4:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    return sentiment_label, prediction

# Take dynamic input from the user
user_input = input("Enter a review: ")  # Allows real-time user input
sentiment, score = predict_sentiment(user_input)

# Display the result
print(f"Sentiment: {sentiment} (Score: {score:.4f})")

# Optionally, display model accuracy if available (uncomment during training phase)
# print(f"Model Accuracy: {history.history['accuracy'][-1]:.2f}")
