import json
import numpy as np
import torch
from datasets import Dataset, Features, Value, Sequence
from transformers import (
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report, multilabel_confusion_matrix, hamming_loss
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load Data ===
with open(r"C:\Users\Rosh\Documents\GitHub\iloELECTRA\finetuning\Multilabel_Dataset_Final_cleaned.json", encoding="utf-8") as f:
    raw_data = json.load(f)

# === Define Dataset Features Explicitly ===
features = Features({
    "text": Value("string"),
    "labels": Sequence(Value("string"))
})
dataset = Dataset.from_list(raw_data)

# === Label setup ===
LABELS = ["DRUG", "SYMPTOM"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)

# === Encode labels to multi-hot vector ===
def encode_labels(example):
    vec = [0] * NUM_LABELS
    if "NONE" not in example["labels"]:
        for label in example["labels"]:
            if label in LABEL2ID:
                vec[LABEL2ID[label]] = 1
    example["label_vector"] = vec
    return example

dataset = dataset.map(encode_labels)

# === Split Dataset ===
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# === Load Tokenizer ===
tokenizer = ElectraTokenizerFast.from_pretrained("ilocano_electra/final_model")

# === Tokenize ===
def tokenize(example):
    encoding = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    encoding["labels"] = [float(x) for x in example["label_vector"]]
    return encoding

train_dataset = train_dataset.map(tokenize, remove_columns=[col for col in train_dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]])
val_dataset = val_dataset.map(tokenize, remove_columns=[col for col in val_dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]])

# === Load Model ===
model = ElectraForSequenceClassification.from_pretrained(
    "ilocano_electra/final_model",
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

# === Metrics ===
def compute_metrics(p):
    preds = torch.sigmoid(torch.tensor(p.predictions)) > 0.5
    labels = torch.tensor(p.label_ids)

    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    avg = report["weighted avg"]
    h_loss = hamming_loss(labels, preds)

    # Confusion Matrices
    cm_dict = multilabel_confusion_matrix(labels, preds)
    os.makedirs("conf_matrices_multilabel", exist_ok=True)
    for i, label in enumerate(LABELS):
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm_dict[i], annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
        plt.title(f"Confusion Matrix: {label}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"conf_matrices_multilabel/{label}_conf_matrix.png")
        plt.close()

    return {
        "accuracy": avg["precision"],  # Optional: you can use a better multi-label accuracy if needed
        "precision": avg["precision"],
        "recall": avg["recall"],
        "f1": avg["f1-score"],
        "hamming_loss": h_loss
    }

# === Training Arguments ===
args = TrainingArguments(
    output_dir="./mClassification_results_iloELECTRA",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"
)

# === Trainer ===
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# === Train ===
trainer.train()

# === Save Final Model ===
trainer.save_model("ilocano_electra/multilabel_finetuned")
tokenizer.save_pretrained("ilocano_electra/multilabel_finetuned")

print("✅ Training complete. Confusion matrices saved in 'conf_matrices_multilabel'.")

# === Final Evaluation ===
print("\n=== 🧪 Final Evaluation on Validation Set ===")
eval_metrics = trainer.evaluate()
for k, v in eval_metrics.items():
    if isinstance(v, float):
        print(f"{k:<20}: {v:.4f}")

# Optional: Print classification report per label
print("\n=== 🧾 Detailed Per-Label Report ===")
outputs = trainer.predict(val_dataset)
preds = torch.sigmoid(torch.tensor(outputs.predictions)) > 0.5
labels = torch.tensor(outputs.label_ids)
print(classification_report(labels, preds, target_names=LABELS, zero_division=0))

print("✅ Training complete. Confusion matrices saved in 'conf_matrices_multilabel'.")