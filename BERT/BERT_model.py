import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AdamW, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Loading the preprocessed dataset from CSV 
dataset = pd.read_csv('preprocessed_data4BERT.csv')

# Splitting the data into training and testing sets
train_df, test_df = train_test_split(dataset, test_size=0.2, random_state=42)

# Reset the indices of the DataFrames
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Encoding the sentiments into numeric labels
label_encoder = LabelEncoder()
train_df['Sentiment'] = label_encoder.fit_transform(train_df['Sentiment'])
test_df['Sentiment'] = label_encoder.transform(test_df['Sentiment'])

# Get Input IDs, Attention Masks, and sentiment labels from the preprocessed data
train_input_ids = train_df['Input IDs'].apply(lambda x: [int(num) for num in x.strip('[]').split(', ')])
train_attention_masks = train_df['Attention Masks'].apply(lambda x: [int(num) for num in x.strip('[]').split(', ')])
train_sentiments = train_df['Sentiment'].values.astype(int)

test_input_ids = test_df['Input IDs'].apply(lambda x: [int(num) for num in x.strip('[]').split(', ')])
test_attention_masks = test_df['Attention Masks'].apply(lambda x: [int(num) for num in x.strip('[]').split(', ')])
test_sentiments = test_df['Sentiment'].values.astype(int)

# Convert the data to tensors
train_input_ids = torch.tensor(train_input_ids)
train_attention_masks = torch.tensor(train_attention_masks)
train_sentiments = torch.tensor(train_sentiments, dtype=torch.long)  # Change data type to Long

test_input_ids = torch.tensor(test_input_ids)
test_attention_masks = torch.tensor(test_attention_masks)
test_sentiments = torch.tensor(test_sentiments, dtype=torch.long)  # Change data type to Long

# Create data loaders
batch_size = 32

train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_sentiments)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_sentiments)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Loading pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Setting up optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 5  # 5 epochs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Fine-tune BERT model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

for epoch in range(5):
    model.train()
    for batch in train_loader:
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
    for batch in test_loader:
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
torch.save(model.state_dict(), 'saved_BERT_model_CUDA.pt')
