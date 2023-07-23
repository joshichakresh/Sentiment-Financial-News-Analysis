import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Load the preprocessed dataset
data = pd.read_csv('preprocessed_data.csv')

# Split the data into features (X) and target variable (y)
X = data['News Headline']
y = data['Sentiment']

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_features = vectorizer.fit_transform(X)

# Load the trained model
loaded_model = joblib.load('saved_model.pkl')

# Predict sentiment for a new news headline
new_headline = input("Enter a new news headline: ")
new_feature = vectorizer.transform([new_headline])
prediction = loaded_model.predict(new_feature)
print("Predicted sentiment:", prediction)
