import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Loading the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

# Loading the trained model weights on the CPU
model.load_state_dict(torch.load('saved_RoBERTa_model_CUDA.pt', map_location=torch.device('cpu')), strict=False)
model.eval()

# Function to predict sentiment given an input headline
def predict_sentiment(headline):
    # Tokenize the input headline
    encoded_input = tokenizer.encode_plus(
        headline,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Perform inference
    with torch.no_grad():
        logits = model(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])[0]
        predicted_class = torch.argmax(logits).item()

    # Map predicted class index to sentiment label
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_sentiment = sentiment_labels[predicted_class]

    return predicted_sentiment

# User input loop
while True:
    headline = input("Enter a news headline (or 'q' to quit): ")
    if headline.lower() == 'q':
        break
    sentiment = predict_sentiment(headline)
    print(f"Predicted sentiment: {sentiment}")
