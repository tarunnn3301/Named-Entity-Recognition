import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from sklearn.model_selection import train_test_split
from evaluate import load
import pandas as pd

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

# Function to load and prepare data
def load_data(filename):
    df = pd.read_csv(filename)
    df['Word'] = df['Word'].astype(str)
    df['Tag'] = df['Tag'].astype(str)

    sentences = []
    for _, sentence in df.groupby("Sent"):
        words = sentence["Word"].tolist()
        labels = sentence["Tag"].tolist()
        sentences.append({"Word": words, "Tag": labels})

    output_data = [(" ".join(item['Word']), item['Tag']) for item in sentences]
    return output_data

data = load_data("Twitterdata/annotatedData.csv")

# Tokenization function with label alignment
def tokenize_and_align_labels(sentences, labels):
    tokenized_inputs = tokenizer(sentences, padding='max_length', truncation=True, return_tensors="pt", max_length=128, is_split_into_words=True)
    word_ids = tokenized_inputs.word_ids(batch_index=0)
    aligned_labels = []

    for i, word_id in enumerate(word_ids):
        if word_id is None:
            aligned_labels.append(-100)  # Ignore padding tokens
        elif i == 0 or word_id != word_ids[i - 1]:  # Assign labels only to the first token of each word
            aligned_labels.append(labels[word_id])
        else:
            aligned_labels.append(-100)  # Ignore subword tokens

    tokenized_inputs["labels"] = torch.tensor(aligned_labels)
    return {k: v.squeeze() for k, v in tokenized_inputs.items()}

# Prepare data for model
tag_map = {
    "B-Loc": 0,
    "B-Org": 1,
    "B-Per": 2,
    "I-Loc": 3,
    "I-Org": 4,
    "I-Per": 5,
    "Other": 6  # 'O' stands for 'Other'
}
id_to_label = {v: k for k, v in tag_map.items()}

# Tokenizing sentences and mapping tags to IDs
sentences = [sentence.split() for sentence, _ in data]
tags = [[tag_map[tag] for tag in label] for _, label in data]

# Filter out mismatched sentence-tag pairs
valid_sentences = []
valid_tags = []
for i in range(len(sentences)):
    if len(sentences[i]) == len(tags[i]):
        valid_sentences.append(sentences[i])
        valid_tags.append(tags[i])

train_sentences, val_sentences, train_tags, val_tags = train_test_split(valid_sentences, valid_tags, test_size=0.2)

# Tokenize and align labels
train_encodings = [tokenize_and_align_labels(sent, label) for sent, label in zip(train_sentences, train_tags)]
val_encodings = [tokenize_and_align_labels(sent, label) for sent, label in zip(val_sentences, val_tags)]

# Consolidate encodings into a single dictionary
def consolidate_encodings(encodings):
    consolidated = {key: [] for key in encodings[0].keys()}
    for enc in encodings:
        for key, value in enc.items():
            consolidated[key].append(value)
    for key, value in consolidated.items():
        consolidated[key] = torch.stack(value)
    return consolidated

train_encodings = consolidate_encodings(train_encodings)
val_encodings = consolidate_encodings(val_encodings)

# Define a PyTorch Dataset
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])

# Create datasets
train_dataset = NERDataset(train_encodings)
val_dataset = NERDataset(val_encodings)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Data collator for token classification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Load the metric
metric = load("seqeval")

# Compute metrics function
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Define the Trainer
trainer = Trainer(
    model=BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(tag_map)),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
results = trainer.evaluate(val_dataset)
print("Test set evaluation results:", results)