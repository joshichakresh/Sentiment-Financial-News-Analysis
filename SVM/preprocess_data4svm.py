import pandas as pd
import re
import spacy

# Loading the English tokenizer, tagger, parser, and NER
nlp = spacy.load("en_core_web_sm")

# Reading the data file without column headings
data = pd.read_csv("all-data.csv", header=None, encoding='latin-1')

# Assigning column names
data.columns = ["Sentiment", "News Headline"]

# Preprocessing the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)
    # Lemmatize the text
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc])
    return text

# Applying preprocessing to the "News Headline" column
data["News Headline"] = data["News Headline"].apply(preprocess_text)

# Print the preprocessed data
print(data)

data.to_csv('preprocessed_data.csv', index=False)

