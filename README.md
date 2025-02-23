# Sentiment-Analysis

This project performs sentiment analysis on tweets using XGBoost and TF-IDF Vectorization. The model classifies tweets as Positive or Negative based on text input.
___

📌 Features

Preprocessing: Tokenization, stopword removal, and lemmatization using NLTK

Feature Extraction: TF-IDF Vectorization

Classification: XGBoost with hyperparameter tuning

User Input: Classify any tweet via command-line input
___

📂 Dataset

The dataset used for training is Sentiment140, which contains 1.6 million tweets labeled as positive (4) or negative (0).

🔗 [Download the dataset from Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

After downloading, place the CSV file in the project directory and name it:

training.1600000.processed.noemoticon.csv
___

🛠 Installation & Setup

Clone the repository

git clone https://github.com/yourusername/sentiment-analysis.git
cd sentiment-analysis

Install dependencies

pip install -r requirements.txt

Run the script

python sentiment_analysis.py

Classify a tweet
After running the script, enter any tweet, and it will predict the sentiment!
___

⚙️ Model Performance

Accuracy: ~62% (on a small balanced subset of 1000 tweets)

To improve accuracy:

Train on a larger subset

Use deep learning models like BERT

Improve preprocessing & feature engineering
___

📜 License

This project is open-source under the MIT License.

