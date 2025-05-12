


import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

"""# **Data Loading and Preprocessing**"""



import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"D:/COLLEGE/reseach_project/HATE_PROJECT_MAIN_D1/hate_text.csv")

# Create 'labels' column based on 'hate_text', 'offensive_language', and 'neither'
def get_label(row):
  """
  This function takes a row from the DataFrame and assigns the corresponding label.
  """
  if row['hate_speech'] == 1:
    return 0  # Hate text
  elif row['offensive_language'] == 1:
    return 1  # Offensive Language
  else:
    return 2  # Neither

data['labels'] = data.apply(get_label, axis=1)

# Select the desired columns
data = data[['comments', 'labels']]

# Split into train and test sets (70/30 split)
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

"""#**Model Selection, Fine Tuning and Training**"""

import os
os.environ["HUGGING_FACE_HUB_TOKEN"] = "your token" # Replace with your actual API key

# 2. Model Selection and Fine-tuning
from datasets import Dataset, DatasetDict # Import the Dataset class here

model_name = "bert-base-uncased"  # Choose BERT (you can explore others)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


# 3. Prepare Data for Hugging Face Trainer
train_dataset = DatasetDict({"train": Dataset.from_pandas(train_data)})
test_dataset = DatasetDict({"test": Dataset.from_pandas(test_data)})

def preprocess_function(examples):
    return tokenizer(examples["comments"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 4. Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,  # Adjust as needed
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # Adjust as needed
    evaluation_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,  # Adjust as needed
)

#Hugging Face Hub token (Important!)
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"  # Replace with your actual API key


#Create Trainer and Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset["train"],
    eval_dataset=test_dataset["test"],
)

trainer.train()


# 6. Evaluate the Model
eval_results = trainer.evaluate()
print(eval_results)

"""# **Model Evaluation and Analysis:**"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Assuming you have predictions and true labels:
predictions = trainer.predict(test_dataset["test"]).predictions.argmax(-1)
true_labels = test_dataset["test"]["labels"]

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted')  # For multi-class
recall = recall_score(true_labels, predictions, average='weighted')
f1 = f1_score(true_labels, predictions, average='weighted')
# AUC (requires probabilities instead of class predictions):
# probabilities = trainer.predict(test_dataset["test"]).predictions[:, 1] # Assuming binary classification
# auc = roc_auc_score(true_labels, probabilities)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
# print(f"AUC: {auc}")

"""# Error Analysis"""

import pandas as pd

# Create a DataFrame to compare predictions with true labels
results_df = pd.DataFrame({'comments': test_dataset["test"]["tweet"], 'true_label': true_labels, 'predicted_label': predictions})

# Filter for misclassified examples
misclassified_df = results_df[results_df['true_label'] != results_df['predicted_label']]

# Print or analyze the misclassified examples
print(misclassified_df)
# Further analysis (e.g., count errors by category, look for patterns, etc.)
# Group by true label and count misclassifications
error_counts = misclassified_df.groupby('true_label')['comments'].count()

# Print the error counts
print("Misclassifications by True Label:")
print(error_counts)

# Print some misclassified tweets for each category (e.g., first 5)
for label in error_counts.index:
  print(f"\nMisclassified Tweets for True Label {label}:")
  print(misclassified_df[misclassified_df['true_label'] == label]['comments'].head(5))

from sklearn.metrics import confusion_matrix

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)

# Print the matrix
print("Confusion Matrix:")
print(conf_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predictions)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Hate Speech', 'Offensive', 'Neither'],
            yticklabels=['Hate Speech', 'Offensive', 'Neither'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

