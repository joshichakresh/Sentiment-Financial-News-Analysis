import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaForSequenceClassification, AdamW, RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

# Loading the dataset without column headers
data = pd.read_csv("/kaggle/input/sentiment-analysis-for-financial-news/all-data.csv", header=None, encoding='latin-1', names=['Sentiment', 'News Headline'])

# Initializing the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenizing and preprocessing the text
input_ids = []
attention_masks = []
max_length = 128  # Specifying the maximum length for truncation

for text in data['News Headline']:
    # Tokenizing the text and add special tokens
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # Extracting input IDs and attention mask
    input_id = encoded_text['input_ids'].flatten()
    attn_mask = encoded_text['attention_mask'].flatten()

    input_ids.append(input_id)
    attention_masks.append(attn_mask)

# Padding the input sequences to the same length
input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

# Converting the preprocessed data to tensors
input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)

# Encoding the sentiments into numeric labels
label_encoder = LabelEncoder()
data['Sentiment'] = label_encoder.fit_transform(data['Sentiment'])
sentiments = torch.tensor(data['Sentiment'].values, dtype=torch.long)

# Creating a data loader
batch_size = 32
dataset = TensorDataset(input_ids, attention_masks, sentiments)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Loading pre-trained RoBERTa model for sequence classification
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

# Setting up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(data_loader) * 5  # 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Fine-tune RoBERTa model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(5):
    model.train()
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()

# Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        _, predicted_labels = torch.max(logits, dim=1)
        predictions.extend(predicted_labels.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

# Calculating evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='macro')
recall = recall_score(true_labels, predictions, average='macro')
f1 = f1_score(true_labels, predictions, average='macro')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# Saving the trained model 
torch.save(model.state_dict(), 'saved_RoBERTa_model_CUDA.pt')
