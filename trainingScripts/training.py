import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import (
    ElectraConfig, ElectraForPreTraining, ElectraForMaskedLM, ElectraTokenizerFast,
    Trainer, TrainingArguments
)
from tokenizers import BertWordPieceTokenizer
from datasets import load_dataset
import numpy as np

# --- CUDA Info ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)

# --- Paths and Hyperparameters ---
CORPUS_FILE = "iloELECTRA_pretrain.txt"
TOKENIZER_DIR = "ilocano_tokenizer"
VOCAB_SIZE = 30000
MODEL_DIR = "ilocano_electra"
PLOT_DIR = "training_plots"
os.makedirs(TOKENIZER_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Check if corpus file exists ---
if not os.path.exists(CORPUS_FILE):
    print(f"❌ Corpus file '{CORPUS_FILE}' not found!")
    print("Please make sure the training data file exists.")
    exit(1)

print(f"✓ Found corpus file: {CORPUS_FILE}")

# --- Tokenizer ---
print("Training tokenizer...")
tokenizer = BertWordPieceTokenizer(lowercase=False)
tokenizer.train(files=[CORPUS_FILE], vocab_size=VOCAB_SIZE)
tokenizer.save_model(TOKENIZER_DIR)
print(f"✓ Tokenizer saved to: {TOKENIZER_DIR}")

hf_tokenizer = ElectraTokenizerFast.from_pretrained(TOKENIZER_DIR)

# --- Load and Split Dataset ---
print("Loading dataset...")
raw_dataset = load_dataset('text', data_files={'data': CORPUS_FILE})['data']
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
print(f"✓ Dataset loaded - Train: {len(split_dataset['train'])}, Test: {len(split_dataset['test'])}")

# --- Tokenization ---
def tokenize_function(example):
    return hf_tokenizer(example["text"], truncation=True, max_length=128)

print("Tokenizing dataset...")
tokenized_dataset = split_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
print("✓ Dataset tokenized")

# --- RTD-Compatible Collator (Fixed) ---
from transformers import DataCollatorForLanguageModeling

class ElectraDataCollatorRTD:
    def __init__(self, tokenizer, generator, mlm_probability=0.15, device='cpu'):
        self.tokenizer = tokenizer
        self.generator = generator
        self.device = device
        self.mlm_probability = mlm_probability
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=mlm_probability)

    def __call__(self, examples):
        batch = self.mlm_collator(examples)
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Move to device for generator inference
        input_ids_gpu = input_ids.to(self.device)
        
        with torch.no_grad():
            gen_outputs = self.generator(input_ids=input_ids_gpu)
            logits = gen_outputs.logits

        probs = torch.softmax(logits, dim=-1)
        sampled_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(input_ids.size())
        
        # Move back to CPU to avoid pin memory error
        sampled_ids = sampled_ids.cpu()
        
        corrupted_input_ids = input_ids.clone()
        discriminator_labels = torch.zeros_like(input_ids)

        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)):
                if labels[i, j] != -100:
                    corrupted_input_ids[i, j] = sampled_ids[i, j]
                    discriminator_labels[i, j] = 1

        return {
            "input_ids": corrupted_input_ids,
            "attention_mask": batch["attention_mask"],
            "labels": discriminator_labels,
        }

# --- Generator (for replacing tokens) ---
print("Creating generator model...")
gen_config = ElectraConfig(
    vocab_size=VOCAB_SIZE,
    embedding_size=64,
    hidden_size=128,
    num_attention_heads=2,
    num_hidden_layers=6,
    intermediate_size=512,
)
generator = ElectraForMaskedLM(config=gen_config).to(device)
print("✓ Generator model created")

rtd_collator = ElectraDataCollatorRTD(
    tokenizer=hf_tokenizer,
    generator=generator,
    mlm_probability=0.15,
    device=device
)

# --- ELECTRA Small Discriminator ---
print("Creating discriminator model...")
disc_config = ElectraConfig(
    vocab_size=VOCAB_SIZE,
    embedding_size=128,
    hidden_size=256,
    num_attention_heads=4,
    num_hidden_layers=12,
    intermediate_size=1024,
)
model = ElectraForPreTraining(config=disc_config).to(device)
print("✓ Discriminator model created")

# --- Training Setup ---
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    save_steps=1000,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    dataloader_pin_memory=False,
    # Add these for better model saving
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=hf_tokenizer,
    data_collator=rtd_collator,
)

# --- Train ---
print("Starting training...")
print("="*50)
train_output = trainer.train()
print("="*50)
print("✓ Training completed!")

# --- Save Model and Tokenizer Explicitly ---
print("Saving final model and tokenizer...")
final_model_dir = os.path.join(MODEL_DIR, "final_model")
os.makedirs(final_model_dir, exist_ok=True)

# Save model
model.save_pretrained(final_model_dir)
hf_tokenizer.save_pretrained(final_model_dir)

# Also save to main directory
model.save_pretrained(MODEL_DIR)
hf_tokenizer.save_pretrained(MODEL_DIR)

print(f"✓ Model saved to: {MODEL_DIR}")
print(f"✓ Final model saved to: {final_model_dir}")

# --- Plot Training + Validation Loss ---
print("Creating training plots...")
logs = trainer.state.log_history
train_losses = [log["loss"] for log in logs if "loss" in log]
eval_losses = [log["eval_loss"] for log in logs if "eval_loss" in log]

if train_losses:
    steps = list(range(0, len(train_losses)*training_args.logging_steps, training_args.logging_steps))
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss", linewidth=2)
    if eval_losses:
        eval_steps = list(range(0, len(eval_losses)*training_args.eval_steps, training_args.eval_steps))
        plt.plot(eval_steps, eval_losses, label="Validation Loss", linewidth=2)
    
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("iloELECTRA Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = f"{PLOT_DIR}/loss_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training plot saved to: {plot_path}")

# --- Training Summary ---
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Model Directory: {MODEL_DIR}")
print(f"Tokenizer Directory: {TOKENIZER_DIR}")
print(f"Vocabulary Size: {VOCAB_SIZE}")
print(f"Training Steps: {train_output.global_step}")
print(f"Final Training Loss: {train_output.training_loss:.3f}")
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Check saved files
model_files = os.listdir(MODEL_DIR)
print(f"Saved Files: {model_files}")
print("="*50)

print("\niloELECTRA pretraining with RTD complete!")
print("✓ Tokenizer saved")
print("✓ Model saved") 
print("✓ Training plots saved")
print("\nYou can now run the evaluation script!")