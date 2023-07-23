import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Loading the preprocessed dataset
data = pd.read_csv('preprocessed_data.csv')

# Splitting the data into features (X) and target variable (y)
X = data['News Headline']
y = data['Sentiment']

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_features = vectorizer.fit_transform(X)

# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Model training using Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Saving the trained model
joblib.dump(svm_model, 'saved_model.pkl')

# Model evaluation
y_pred = svm_model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)


