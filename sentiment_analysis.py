import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Download and Load Dataset
print("Downloading dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("abhi8923shriv/sentiment-analysis-dataset")
print("Path to dataset files:", dataset_path)

# Load the CSV file (using latin1 encoding as discovered earlier)
csv_path = os.path.join(dataset_path, "train.csv")
df = pd.read_csv(csv_path, encoding='latin1')

# Keep only the necessary columns and drop rows with missing values
df = df[['text', 'sentiment']].dropna()

print("Dataset Loaded. Total rows:", len(df))
print("Class distribution:\n", df['sentiment'].value_counts())

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

# 3. Feature Extraction (TF-IDF)
print("Converting text to features...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Logistic Regression Model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 5. Evaluation
print("Evaluating model...")
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Sentiment Analysis Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
