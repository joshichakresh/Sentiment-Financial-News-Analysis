import torch
import pandas as pd
from transformers import BertTokenizer

# Loading the dataset without column headers
data = pd.read_csv("all-data.csv", header=None, encoding='latin-1', names=['Sentiment', 'News Headline'])

# Initializing the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenizing and preprocessing the text
input_ids = []
attention_masks = []

for text in data['News Headline']:
    # Tokenizing the text and adding special tokens
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Extracting input IDs and attention mask
    input_id = encoded_text['input_ids'].flatten()
    attn_mask = encoded_text['attention_mask'].flatten()
    
    input_ids.append(input_id)
    attention_masks.append(attn_mask)

# Converting lists to tensors
input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)

# Storing the preprocessed dataset
preprocessed_data = pd.DataFrame({
    'Input IDs': input_ids.tolist(),
    'Attention Masks': attention_masks.tolist(),
    'Sentiment': data['Sentiment']
})

preprocessed_data.to_csv('preprocessed_data4BERT.csv', index=False)
