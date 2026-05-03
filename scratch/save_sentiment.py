import os
import kagglehub
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("Quick saving sentiment model...")
dataset_path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
csv_path = os.path.join(dataset_path, "train.csv")
df = pd.read_csv(csv_path, encoding='latin1').dropna().sample(2000) # Use subset for speed

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
model = LogisticRegression(max_iter=1000)
model.fit(X, df['sentiment'])

pickle.dump(model, open('sentiment_model.pkl', 'wb'))
pickle.dump(vectorizer, open('sentiment_vectorizer.pkl', 'wb'))
print("✅ Sentiment model saved successfully.")
