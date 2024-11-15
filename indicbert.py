# -*- coding: utf-8 -*-
"""IndicBERT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VDxmKVPx-NDi7SlP-lvDRqJnx-56sZfH
"""

# !pip install transformers torch
# !pip install datasets --upgrade
# !pip install evaluate
# !pip install seqeval



# DatasetPath = "/content/drive/MyDrive/Dataset"

from pickle import TRUE
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import DatasetDict, Dataset
import pandas as pd
from evaluate import load


# Load your data from CSV into a Hugging Face dataset format
def load_data(filename):
    df = pd.read_csv(filename)
    df['Word'] = df['Word'].astype(str)
    df['Tag'] = df['Tag'].astype(str)

    # Create a list of dictionaries with the sentence_id, words, and labels
    sentences = []
    for sentence_id, sentence in df.groupby("Sent"):
        words = sentence["Word"].tolist()
        labels = sentence["Tag"].tolist()
        sentences.append({"Word": words, "Tag": labels})

    # Convert to Hugging Face DatasetDict format and split into train and test sets
    dataset = Dataset.from_pandas(pd.DataFrame(sentences))
    train_test_split = dataset.train_test_split(test_size=0.3, seed=42)  # 80% for training, 20% for testing
    return DatasetDict({
        'train': train_test_split['train'],
        'test': train_test_split['test']
    })

# Define tags and labels
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Load dataset and split into train/test sets
dataset = load_data('Twitterdata/annotatedData.csv')

print(dataset['train'].shape)
print(dataset['test'].shape)

# Load tokenizer and model
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

# Ensure all parameters are contiguous
for param in model.parameters():
    param.data = param.data.contiguous()

# Tokenize dataset
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["Word"], truncation=True, padding="max_length", max_length=128, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["Tag"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                current_label = label[word_idx].upper()
                label_ids.append(label_to_id.get(current_label, label_to_id['O']))
            else:
                current_label = label[word_idx].upper()
                label_ids.append(label_to_id.get(current_label, label_to_id['O']) if current_label.startswith("I-") else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Define metrics
metric = load("seqeval")

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

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer with separate train and test datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # Use the test set for evaluation
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test set
results = trainer.evaluate(tokenized_datasets["test"])
print("Test set evaluation results:", results)

# # Make predictions on the test set
# predictions, labels, _ = trainer.predict(tokenized_datasets["test"])

# # Convert predictions to tag IDs
# predicted_tag_ids = predictions.argmax(axis=2)

# # Decode predictions to tag labels
# predicted_tags = [
#     [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
#     for prediction, label in zip(predicted_tag_ids, labels)
# ]

# # Decode true labels for comparison (optional)
# true_tags = [
#     [id_to_label[l] for l in label if l != -100]
#     for label in labels
# ]

# # Print out the predictions and the true labels
# for i in range(len(predicted_tags)):
#     print(f"Sentence {i+1}:")
#     print("Predicted Tags:", predicted_tags[i])
#     print("True Tags:", true_tags[i])
#     print()

