Sentiment-Analysis-with-real-time-classification

This project performs sentiment analysis on tweets using Logistic Regression and TF-IDF Vectorization. The model classifies tweets as Positive or Negative based on text input.
***
📌 Features
Preprocessing: Tokenization, stopword removal, and lemmatization using NLTK
Feature Extraction: TF-IDF Vectorization
Classification: Logistic Regression with hyperparameter tuning using GridSearchCV
User Input: Classify any tweet via command-line input
***
📂 Dataset
The dataset used for training is Sentiment140, which contains 1.6 million tweets labeled as positive (4) or negative (0).

🔗 [Download the dataset from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

After downloading, place the CSV file in the project directory and name it:

Copy
Edit
training.1600000.processed.noemoticon.csv
For this project, a small balanced subset of 1,000 tweets was used for training.
***
📊 Visualizations
The project includes several visualizations to analyze results:

1️⃣ Confusion Matrix – Displays model performance.
2️⃣ Class Distribution After Auto-Labeling – Shows dataset balance.
3️⃣ Top 20 Important Words for Sentiment Prediction – Highlights key words contributing to classification.

You can generate these plots using the script visualizations.py:

python visualizations.py
***
🛠 Installation & Setup

Clone the repository:
git clone https://github.com/yourusername/sentiment-analysis.git

Navigate to the project directory:
cd sentiment-analysis

Install dependencies:
pip install -r requirements.txt

Run the script:
python sentiment_analysis.py

Classify a tweet:
After running the script, enter any tweet, and it will predict the sentiment! 
Classify a tweet:
After running the script, enter any tweet, and it will predict the sentiment!
***
⚙️ Model Performance
Accuracy: 70.50% (on a small balanced subset of 1,000 tweets).
Hyperparameter tuning: Used GridSearchCV, not random search.
🔹 To improve accuracy:
✅ Train on a larger subset of the dataset.
✅ Try deep learning models like BERT or LSTMs.
✅ Improve preprocessing & feature engineering.
***
📜 License
This project is open-source under the MIT License.
