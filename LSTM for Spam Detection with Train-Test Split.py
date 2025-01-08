import glob
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 1: Read the email files
def read_emails(paths, label):
    emails = []
    labels = []
    for path in paths:
        for file_path in glob.glob(path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                emails.append(file.read())
                labels.append(label)
    return emails, labels

train_spam_paths = ["enron6/spam/*.txt", ]
train_ham_paths = ["enron6/ham/*.txt", ]

train_spam_emails, train_spam_labels = read_emails(train_spam_paths, 1)  # Label 1 for spam
train_ham_emails, train_ham_labels = read_emails(train_ham_paths, 0)     # Label 0 for ham

train_emails = train_spam_emails + train_ham_emails
train_labels = train_spam_labels + train_ham_labels

test_spam_paths = ["enron5/spam/*.txt"]
test_ham_paths = ["enron5/ham/*.txt"]

test_spam_emails, test_spam_labels = read_emails(test_spam_paths, 1)  # Label 1 for spam
test_ham_emails, test_ham_labels = read_emails(test_ham_paths, 0)     # Label 0 for ham

test_emails = test_spam_emails + test_ham_emails
test_labels = test_spam_labels + test_ham_labels

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
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# # Step 4: Define the LSTM model
# class SpamDetector(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
#         super(SpamDetector, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x):
#         embedded = self.embedding(x)
#         lstm_out, _ = self.lstm(embedded)
#         out = self.fc(lstm_out[:, -1, :])
#         return out

# vocab_size = len(vocab) + 1  # +1 for padding token
# embedding_dim = 100
# hidden_dim = 128
# output_dim = 1

# model = SpamDetector(vocab_size, embedding_dim, hidden_dim, output_dim)

# Step 4: Define the optimized LSTM model
class SpamDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers):
        super(SpamDetector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out

vocab_size = len(vocab) + 1  # +1 for padding token
embedding_dim = 50  # Reduced
hidden_dim = 64    # Reduced
output_dim = 1
num_layers = 1     # Reduced to 1 layer

model = SpamDetector(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)

# Step 5: Train the model
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

num_epochs = 3
batch_size = 2

train_dataset = torch.utils.data.TensorDataset(train_emails_tensor, train_labels_tensor)
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
test_dataset = torch.utils.data.TensorDataset(test_emails_tensor, test_labels_tensor)
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