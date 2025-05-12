import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.optim as optim
import pandas as pd

# Load the dataset (adjust the file path and column names based on your dataset)
# Assuming a CSV file with 7 columns, and the relevant columns are 'text' and 'label'
dataset = pd.read_csv(r"C:\Users\ps844\Downloads\hate_speech\hate_text.csv")

# Inspect the dataset to identify column names
print(dataset.head())

# Extract the 'text' and 'label' columns (assuming 'label' is the last column, adjust if needed)
texts = dataset['text'].tolist()
labels = dataset['label'].tolist()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Preprocess text into tokens
def tokenize_data(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return torch.tensor(encodings['input_ids'])

train_inputs = tokenize_data(X_train)
test_inputs = tokenize_data(X_test)
train_labels = torch.tensor(y_train)
test_labels = torch.tensor(y_test)

# Define GRU-based model for hate speech recognition (3 classes)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRUModel, self).__init__()
        self.bert = bert_model
        self.gru = nn.GRU(input_size=768, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids):
        # Get BERT embeddings
        outputs = self.bert(input_ids)
        hidden_states = outputs.last_hidden_state
        gru_out, _ = self.gru(hidden_states)
        # Get the final hidden state for classification
        final_hidden_state = gru_out[:, -1, :]
        logits = self.fc(final_hidden_state)
        return self.softmax(logits)

# Initialize model
input_size = 768  # BERT output size
hidden_size = 128
num_classes = 3  # Hate, Offensive, Neutral
model = GRUModel(input_size, hidden_size, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
def train_model(model, train_inputs, train_labels, epochs=5):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_inputs)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

train_model(model, train_inputs, train_labels)

# Evaluate the model on the test set
def evaluate_model(model, test_inputs, test_labels):
    model.eval()
    with torch.no_grad():
        outputs = model(test_inputs)
        _, predicted = torch.max(outputs, 1)
        print(classification_report(test_labels, predicted, target_names=['Neutral', 'Offensive', 'Hate']))

evaluate_model(model, test_inputs, test_labels)
