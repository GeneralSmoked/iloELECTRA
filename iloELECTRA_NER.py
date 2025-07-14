import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import Dataset
from transformers import (
    ElectraTokenizerFast,
    ElectraForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

# === CONFIG ===
MODEL_PATH = "ilocano_electra/final_model"
DATA_PATH = r"C:\Users\Rosh\Documents\GitHub\iloELECTRA\finetuning\GAMOTPH_NER_DATASET_ILOCANO.json"
LABELS = ['O', 'B-DRUG', 'I-DRUG', 'B-SYMPTOM', 'I-SYMPTOM']
LABELS_EXCL_O = LABELS[1:]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
NUM_LABELS = len(LABELS)

# === Load Tokenizer & Model ===
tokenizer = ElectraTokenizerFast.from_pretrained(MODEL_PATH)
model = ElectraForTokenClassification.from_pretrained(
    MODEL_PATH,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

# === Load & Prepare Dataset ===
with open(DATA_PATH, encoding='utf-8') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)
    word_ids = tokenized_inputs.word_ids()
    labels = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label2id[example["ner_tags"][word_idx]])
        else:
            tag = example["ner_tags"][word_idx]
            if tag.startswith("B-"):
                tag = tag.replace("B-", "I-")
            labels.append(label2id[tag])
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply preprocessing
dataset = dataset.map(tokenize_and_align_labels)

# Split into train/val
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

# === Define Metrics ===
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred, label in zip(predictions, labels):
        temp_labels = []
        temp_preds = []
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                temp_labels.append(id2label[l_i])
                temp_preds.append(id2label[p_i])
        true_labels.append(temp_labels)
        true_predictions.append(temp_preds)

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions)
    }

# === Training Setup ===
args = TrainingArguments(
    output_dir="./ner_results_iloELECTRA",
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

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# === Train ===
trainer.train()

# === Save final model ===
trainer.save_model("ilocano_electra/ner_finetuned")
tokenizer.save_pretrained("ilocano_electra/ner_finetuned")

# === Evaluate ===
print("\n=== Evaluation on Validation Set ===")
predictions_output = trainer.predict(val_dataset)
metrics = predictions_output.metrics
print(f"Accuracy:  {metrics['test_accuracy']:.4f}")
print(f"Precision: {metrics['test_precision']:.4f}")
print(f"Recall:    {metrics['test_recall']:.4f}")
print(f"F1 Score:  {metrics['test_f1']:.4f}")

# === Confusion Matrix (Excluding 'O') ===
true_tags = []
pred_tags = []

predictions = np.argmax(predictions_output.predictions, axis=2)
for pred, label in zip(predictions, predictions_output.label_ids):
    for p_i, l_i in zip(pred, label):
        if l_i != -100:
            true_tag = id2label[l_i]
            pred_tag = id2label[p_i]
            if true_tag != "O":
                true_tags.append(true_tag)
                pred_tags.append(pred_tag)

# Confusion matrix only for B-/I- classes
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_tags, pred_tags, labels=LABELS_EXCL_O)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=LABELS_EXCL_O, yticklabels=LABELS_EXCL_O)
plt.title("Confusion Matrix (Excluding 'O')")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("ner_results_iloELECTRA/confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved to ner_results_iloELECTRA/confusion_matrix.png")
