import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Your labeled data: contract descriptions and their corresponding values
contract_descriptions = ["Sample contract description 1", "Sample contract description 2", ...]
values = [10.5, 15.2, ...]  # Replace with your actual values

# Tokenize the descriptions and convert them to input features
input_ids = []
attention_masks = []
for desc in contract_descriptions:
    encoded_dict = tokenizer(desc, add_special_tokens=True, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Convert lists to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
values = torch.tensor(values, dtype=torch.float32)

# Split the data into training and validation sets
train_inputs, val_inputs, train_masks, val_masks, train_values, val_values = train_test_split(
    input_ids, attention_masks, values, test_size=0.2, random_state=42)

# Create DataLoader objects for training and validation
batch_size = 16
train_data = TensorDataset(train_inputs, train_masks, train_values)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
val_data = TensorDataset(val_inputs, val_masks, val_values)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.MSELoss()

# Training loop
num_epochs = 5
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Avg. Loss: {avg_loss}")

# Evaluation on the validation set
model.eval()
val_loss = 0
predictions = []
with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        val_loss += loss_fn(outputs.logits.view(-1), labels)
        predictions.extend(outputs.logits.view(-1).tolist())

avg_val_loss = val_loss / len(val_dataloader)
print(f"Avg. Validation Loss: {avg_val_loss}")

# Now you can use the model to make predictions on new data
