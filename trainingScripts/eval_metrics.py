import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import ElectraForPreTraining, ElectraTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Configuration ---
MODEL_DIR = "ilocano_electra"
TOKENIZER_DIR = "ilocano_tokenizer"
CORPUS_FILE = "iloELECTRA_pretrain.txt"
EVAL_BATCH_SIZE = 32
MAX_LENGTH = 128

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Model and Tokenizer ---
print("Loading model and tokenizer...")

# Check if model directory exists and has required files
if not os.path.exists(MODEL_DIR):
    print(f"❌ Model directory '{MODEL_DIR}' not found!")
    print("Please run the training script first to create the model.")
    exit(1)

# List files in model directory
model_files = os.listdir(MODEL_DIR)
print(f"Files in {MODEL_DIR}: {model_files}")

# Check for checkpoint directories (Trainer saves to checkpoint-* folders)
checkpoint_dirs = [d for d in model_files if d.startswith('checkpoint-')]
if checkpoint_dirs:
    # Use the latest checkpoint
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))
    MODEL_DIR = os.path.join(MODEL_DIR, latest_checkpoint)
    print(f"Using checkpoint: {MODEL_DIR}")

try:
    # Try loading tokenizer first
    if os.path.exists(TOKENIZER_DIR):
        tokenizer = ElectraTokenizerFast.from_pretrained(TOKENIZER_DIR)
        print("✓ Tokenizer loaded successfully")
    else:
        print(f"❌ Tokenizer directory '{TOKENIZER_DIR}' not found!")
        exit(1)
    
    # Try loading model
    model = ElectraForPreTraining.from_pretrained(MODEL_DIR).to(device)
    print("✓ Model loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure the training script completed successfully")
    print("2. Check if model files exist in the output directory")
    print("3. Look for checkpoint-* directories in the model folder")
    exit(1)

# --- Load Test Dataset ---
print("Loading test dataset...")
raw_dataset = load_dataset('text', data_files={'data': CORPUS_FILE})['data']
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
test_dataset = split_dataset['test']

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding='max_length', 
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

# Tokenize test dataset
print("Tokenizing test dataset...")
tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# --- Create Data Collator for Evaluation ---
from transformers import DataCollatorForLanguageModeling

class ElectraEvalCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm_probability=mlm_probability
        )
    
    def __call__(self, examples):
        # Create masked inputs for evaluation
        batch = self.mlm_collator(examples)
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        # Create discriminator labels (1 for replaced tokens, 0 for original)
        discriminator_labels = torch.zeros_like(input_ids)
        
        # Simulate token replacement for evaluation
        # In real RTD, this would use generator predictions
        mask = (labels != -100)
        discriminator_labels[mask] = 1
        
        return {
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"],
            "labels": discriminator_labels,
        }

eval_collator = ElectraEvalCollator(tokenizer)

# Create DataLoader
dataloader = DataLoader(
    tokenized_test, 
    batch_size=EVAL_BATCH_SIZE, 
    collate_fn=eval_collator,
    shuffle=False
)

# --- Evaluation Function ---
def evaluate_model():
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Get loss and predictions
            loss = outputs.loss
            logits = outputs.logits
            
            # Convert logits to predictions (0 or 1)
            predictions = torch.sigmoid(logits) > 0.5
            
            # Collect predictions and labels
            predictions_flat = predictions.cpu().numpy().flatten()
            labels_flat = labels.cpu().numpy().flatten()
            
            # Only consider non-padded tokens
            mask = attention_mask.cpu().numpy().flatten().astype(bool)
            
            all_predictions.extend(predictions_flat[mask])
            all_labels.extend(labels_flat[mask])
            
            total_loss += loss.item()
            num_batches += 1
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    avg_loss = total_loss / num_batches
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': avg_loss
    }

# --- Run Evaluation ---
print("\n" + "="*50)
print("STARTING EVALUATION")
print("="*50)

metrics = evaluate_model()

# --- Display Results ---
print("\n" + "="*50)
print("EVALUATION METRICS:")
print("="*50)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1_score']:.3f}")
print(f"Final Training Loss: {metrics['loss']:.3f}")
print("="*50)

# --- Save Metrics to File ---
metrics_file = os.path.join(MODEL_DIR, "evaluation_metrics.txt")
with open(metrics_file, 'w') as f:
    f.write("Evaluation Metrics:\n")
    f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
    f.write(f"Precision: {metrics['precision']:.3f}\n")
    f.write(f"Recall: {metrics['recall']:.3f}\n")
    f.write(f"F1-Score: {metrics['f1_score']:.3f}\n")
    f.write(f"Final Training Loss: {metrics['loss']:.3f}\n")

print(f"✓ Metrics saved to: {metrics_file}")

# --- Create Metrics Visualization ---
def plot_metrics():
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metrics_values = [
        metrics['accuracy'], 
        metrics['precision'], 
        metrics['recall'], 
        metrics['f1_score']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylim(0, 1)
    plt.title('iloELECTRA Model Evaluation Metrics')
    plt.ylabel('Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = os.path.join("training_plots", "evaluation_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Metrics plot saved to: {plot_path}")

# Create visualization
plot_metrics()

# --- Model Information ---
print("\n" + "="*50)
print("MODEL INFORMATION:")
print("="*50)
print(f"Model Directory: {MODEL_DIR}")
print(f"Tokenizer Directory: {TOKENIZER_DIR}")
print(f"Vocabulary Size: {tokenizer.vocab_size}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Test Dataset Size: {len(tokenized_test)}")
print("="*50)

print("\niloELECTRA evaluation complete!")