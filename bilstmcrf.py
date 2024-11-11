import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load and Preprocess Dataset
data = pd.read_csv("./Twitterdata/annotatedData.csv", encoding="latin1")
data = data.ffill()  # Forward fill missing values

# Encode tags and words
word_encoder = LabelEncoder()
tag_encoder = LabelEncoder()

# Preparing sentences and tags grouped by sentence
class SentenceGetter:
    def __init__(self, data):
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
        self.grouped = data.groupby("Sent").apply(agg_func)
        self.sentences = [s for s in self.grouped]

getter = SentenceGetter(data)
sentences = getter.sentences

# Extract words and tags to fit label encoders
words = list(set(data["Word"].values))
tags = list(set(data["Tag"].values))

# Fit encoders
word_encoder.fit(words)
tag_encoder.fit(tags)

# Special tokens for padding and unknown words
PAD_WORD = "<PAD>"
UNK_WORD = "<UNK>"
PAD_TAG = "O"

word_encoder.classes_ = np.append(word_encoder.classes_, [PAD_WORD, UNK_WORD])
tag_encoder.classes_ = np.append(tag_encoder.classes_, PAD_TAG)

PAD_WORD_IDX = word_encoder.transform([PAD_WORD])[0]
PAD_TAG_IDX = tag_encoder.transform([PAD_TAG])[0]

# Convert sentences to indices
def encode_sentence(sentence):
    word_indices = [word_encoder.transform([word])[0] if word in word_encoder.classes_ else word_encoder.transform([UNK_WORD])[0] for word, _ in sentence]
    tag_indices = [tag_encoder.transform([tag])[0] for _, tag in sentence]
    return word_indices, tag_indices

# Prepare data in terms of indices
data_encoded = [encode_sentence(s) for s in sentences]
X = [s[0] for s in data_encoded]
y = [s[1] for s in data_encoded]

# Padding sequences for DataLoader
def pad_sequences(sequences, pad_value):
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

X_padded = pad_sequences(X, PAD_WORD_IDX)
y_padded = pad_sequences(y, PAD_TAG_IDX)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2, random_state=42)

# Custom Dataset
class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NERDataset(X_train, y_train)
test_dataset = NERDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# BiLSTM-CRF Model
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=128):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_WORD_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, sentences, tags=None):
        embeddings = self.embedding(sentences)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.hidden2tag(lstm_out)

        if tags is not None:
            # Calculate negative log-likelihood for training
            mask = sentences != PAD_WORD_IDX  # Ignore padding in loss calculation
            loss = -self.crf(emissions, tags, mask=mask)
            return loss
        else:
            # Inference: decode the best sequence of tags
            return self.crf.decode(emissions)

# Initialize model
vocab_size = len(word_encoder.classes_)
tagset_size = len(tag_encoder.classes_)
model = BiLSTM_CRF(vocab_size, tagset_size)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training Loop
n_epochs = 5
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for sentences, tags in train_loader:
        optimizer.zero_grad()
        loss = model(sentences, tags)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss}")

# Evaluation
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for sentences, tags in test_loader:
        predictions = model(sentences)
        for i in range(len(predictions)):
            mask = sentences[i] != PAD_WORD_IDX
            y_true.append(tags[i][mask].cpu().numpy())
            # y_pred.append(predictions[i])
            y_pred.append([predictions[i][j] for j in range(len(predictions[i])) if mask[j]])

# Flatten the lists for classification report
y_true_flat = [tag for sentence in y_true for tag in sentence]
y_pred_flat = [tag for sentence in y_pred for tag in sentence]

# Generate Classification Report
# print(classification_report(y_true_flat, y_pred_flat, target_names=tag_encoder.classes_))
unique_classes = np.unique(y_true_flat + y_pred_flat)  # Get the unique classes in y_true and y_pred
print(classification_report(y_true_flat, y_pred_flat, labels=unique_classes, target_names=tag_encoder.inverse_transform(unique_classes),digits=6))


