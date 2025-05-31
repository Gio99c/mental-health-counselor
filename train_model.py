import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SuicideClassificationDataset:
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_and_preprocess_data():
    """Load and preprocess the suicide severity dataset."""
    print("Loading dataset...")
    df = pd.read_csv('500_Reddit_users_posts_labels.csv')
    
    # Create severity score mapping
    label_to_score = {
        'Supportive': 0,     # No risk
        'Indicator': 2,      # Low risk 
        'Ideation': 4,       # Moderate risk
        'Behavior': 7,       # High risk
        'Attempt': 9         # Severe risk
    }
    
    # Map labels to scores (0-10 scale)
    df['severity_score'] = df['Label'].map(label_to_score)
    
    # Clean text data
    df['Post'] = df['Post'].astype(str)
    df['Post'] = df['Post'].str.replace(r'\[|\]|\'', '', regex=True)  # Remove brackets and quotes
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['Label'].value_counts()}")
    print(f"Severity score distribution:\n{df['severity_score'].value_counts().sort_index()}")
    
    return df

def train_bert_model():
    """Train BERT model for suicide severity classification."""
    
    # Load data
    df = load_and_preprocess_data()
    
    # Prepare data
    texts = df['Post'].tolist()
    labels = df['severity_score'].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"  # Lighter than full BERT
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': X_train,
        'labels': y_train
    })
    
    test_dataset = Dataset.from_dict({
        'text': X_test,
        'labels': y_test
    })
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=512
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Initialize model for regression (predicting scores 0-10)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Regression task
        problem_type="regression"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./suicide_severity_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained('./suicide_severity_model')
    tokenizer.save_pretrained('./suicide_severity_model')
    
    # Evaluate model
    print("Evaluating model...")
    predictions = trainer.predict(test_dataset)
    predicted_scores = predictions.predictions.flatten()
    
    # Round to nearest integer and clip to 0-10 range
    predicted_scores = np.clip(np.round(predicted_scores), 0, 10).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predicted_scores)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test MAE: {np.mean(np.abs(np.array(y_test) - predicted_scores)):.4f}")
    
    # Print some example predictions
    print("\nExample predictions:")
    for i in range(min(5, len(X_test))):
        print(f"Text: {X_test[i][:100]}...")
        print(f"True score: {y_test[i]}, Predicted score: {predicted_scores[i]}")
        print("-" * 50)
    
    return model, tokenizer

if __name__ == "__main__":
    # Create output directory
    os.makedirs('./suicide_severity_model', exist_ok=True)
    
    # Train model
    model, tokenizer = train_bert_model()
    print("Training completed! Model saved to './suicide_severity_model'") 