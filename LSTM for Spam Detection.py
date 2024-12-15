import os
import glob
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 1: Read the email files
def read_emails(paths):
    emails = []
    for path in paths:
        for file_path in glob.glob(path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                emails.append(file.read())
    return emails

train_paths = ["enron1/spam/*.txt", "enron2/spam/*.txt"]
test_paths = ["enron5/spam/*.txt", "enron6/spam/*.txt"]

train_emails = read_emails(train_paths)
test_emails = read_emails(test_paths)

# Step 2: Preprocess the text data
def preprocess_text(text):
    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

train_emails = [preprocess_text(email) for email in train_emails]
test_emails = [preprocess_text(email) for email in test_emails]

# Step 3: Create a vocabulary and convert words to indices
word_counts = Counter(word for email in train_emails + test_emails for word in email.split())
vocab = {word: idx + 1 for idx, (word, count) in enumerate(word_counts.items())}

def email_to_indices(email, vocab):
    return [vocab[word] for word in email.split() if word in vocab]

train_emails_indices = [email_to_indices(email, vocab) for email in train_emails]
test_emails_indices = [email_to_indices(email, vocab) for email in test_emails]

# Pad sequences to ensure uniform input size
max_length = max(len(email) for email in train_emails_indices + test_emails_indices)
padded_train_emails = [email + [0] * (max_length - len(email)) for email in train_emails_indices]
padded_test_emails = [email + [0] * (max_length - len(email)) for email in test_emails_indices]

# Convert to tensor
train_emails_tensor = torch.tensor(padded_train_emails)
test_emails_tensor = torch.tensor(padded_test_emails)

# Labels (assuming all provided emails are spam)
train_labels = torch.ones(train_emails_tensor.size(0), dtype=torch.long)
test_labels = torch.ones(test_emails_tensor.size(0), dtype=torch.long)

# Step 4: Define the LSTM model
class SpamDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SpamDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out

vocab_size = len(vocab) + 1  # +1 for padding token
embedding_dim = 100
hidden_dim = 128
output_dim = 1

model = SpamDetector(vocab_size, embedding_dim, hidden_dim, output_dim)

# Step 5: Train the model
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

num_epochs = 3
batch_size = 2

train_dataset = torch.utils.data.TensorDataset(train_emails_tensor, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        predictions = model(batch[0]).squeeze(1)
        loss = criterion(predictions, batch[1].float())
        acc = binary_accuracy(predictions, batch[1])
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}, Acc: {epoch_acc/len(train_loader)}')

print("Training complete.")

# Step 6: Evaluate the model on the test set
test_dataset = torch.utils.data.TensorDataset(test_emails_tensor, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()

test_epoch_loss = 0
test_epoch_acc = 0

with torch.no_grad():
    for batch in test_loader:
        predictions = model(batch[0]).squeeze(1)
        loss = criterion(predictions, batch[1].float())
        acc = binary_accuracy(predictions, batch[1])

        test_epoch_loss += loss.item()
        test_epoch_acc += acc.item()

print(f'Test Loss: {test_epoch_loss/len(test_loader)}, Test Acc: {test_epoch_acc/len(test_loader)}')



