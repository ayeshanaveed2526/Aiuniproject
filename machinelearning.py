# ============================================================
#  ML/NLP Project — BART Text Classifier (3 Classes)
#  Dataset: AG News (World / Sports / Technology)
#  Run this in Google Colab (GPU recommended)
# ============================================================

# ── STEP 0: Install dependencies ────────────────────────────
# !pip install datasets transformers torch scikit-learn imbalanced-learn matplotlib seaborn torchviz

# ── STEP 1: Imports ─────────────────────────────────────────
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import RandomOverSampler

# ── STEP 2: Load AG News and keep only 3 classes ────────────
# AG News label map: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
# We keep: 0 (World), 1 (Sports), 3 (Sci/Tech) → rename to 0/1/2

CLASS_MAP = {0: "world", 1: "sports", 3: "technology"}
CLASS_NAMES = ["world", "sports", "technology"]

print("Loading AG News dataset...")
raw = load_dataset("ag_news")

def filter_and_map(split):
    texts, labels = [], []
    for item in raw[split]:
        if item["label"] in CLASS_MAP:
            texts.append(item["text"])
            labels.append(CLASS_MAP[item["label"]])
    return texts, labels

all_texts, all_labels = filter_and_map("train")

# Subsample to 3000 per class for faster training (optional — remove cap for full run)
MAX_PER_CLASS = 1000
texts, labels = [], []
counter = Counter()
for t, l in zip(all_texts, all_labels):
    if counter[l] < MAX_PER_CLASS:
        texts.append(t)
        labels.append(l)
        counter[l] += 1

print("Class distribution:", Counter(labels))

# ── STEP 3: Train / test split ───────────────────────────────
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print("Train distribution:", Counter(train_labels))

# ── STEP 4: Oversample minority class ───────────────────────
oversampler = RandomOverSampler(random_state=42)
train_texts_res, train_labels_res = oversampler.fit_resample(
    np.array(train_texts).reshape(-1, 1), train_labels
)
train_texts_res = train_texts_res.flatten().tolist()
print("After oversampling:", Counter(train_labels_res))

# ── STEP 5: Tokenize with BART ───────────────────────────────
MODEL_NAME = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(texts, labels):
    inputs = tokenizer(
        [f"classify: {t}" for t in texts],
        truncation=True, padding=True, max_length=256, return_tensors="pt"
    )
    targets = tokenizer(
        labels, truncation=True, padding=True, max_length=10, return_tensors="pt"
    )
    return inputs, targets

print("Tokenizing...")
train_enc, train_tgt = preprocess(train_texts_res, train_labels_res)
test_enc,  test_tgt  = preprocess(test_texts,      test_labels)

# ── STEP 6: PyTorch Dataset ──────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, encodings, targets):
        self.enc = encodings
        self.tgt = targets
    def __len__(self):
        return self.tgt["input_ids"].shape[0]
    def __getitem__(self, idx):
        item = {k: self.enc[k][idx] for k in self.enc}
        item["labels"] = self.tgt["input_ids"][idx]
        return item

train_dataset = TextDataset(train_enc, train_tgt)
test_dataset  = TextDataset(test_enc,  test_tgt)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=16)

# ── STEP 7: Load BART model ──────────────────────────────────
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Training on: {device}")

# ── STEP 8: Optimizer & Scheduler ───────────────────────────
NUM_EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# ── STEP 9: Helper functions ─────────────────────────────────
def normalize_prediction(text):
    """Map raw model output to one of the 3 class names."""
    text = text.strip().lower()
    for cls in CLASS_NAMES:
        if cls in text:
            return cls
    return CLASS_NAMES[0]  # default fallback

def evaluate_loader(loader):
    preds_raw, labels_decoded = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out  = model.generate(ids, attention_mask=mask, max_length=10)
            preds_raw.extend([tokenizer.decode(g, skip_special_tokens=True) for g in out])
            labels_decoded.extend([tokenizer.decode(l, skip_special_tokens=True) for l in batch["labels"]])
    preds  = [normalize_prediction(p) for p in preds_raw]
    labels = [normalize_prediction(l) for l in labels_decoded]
    return preds, labels

def plot_confusion_matrix(true, pred, title, filename):
    cm = confusion_matrix(true, pred, labels=CLASS_NAMES)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")

def plot_roc_curve_multiclass(true, pred, title, filename):
    """One-vs-Rest ROC for each class."""
    true_bin = label_binarize(true, classes=CLASS_NAMES)
    pred_bin = label_binarize(pred, classes=CLASS_NAMES)

    plt.figure(figsize=(8, 6))
    colors = ["darkorange", "steelblue", "green"]
    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(true_bin[:, i], pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{cls} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Saved: {filename}")

# ── STEP 10: Training Loop ───────────────────────────────────
best_val_acc = 0.0
train_acc_history, val_acc_history, loss_history = [], [], []

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch + 1} / {NUM_EPOCHS}")
    print(f"{'='*50}")

    # --- Training ---
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        lbls   = batch["labels"].to(device)
        out    = model(input_ids=ids, attention_mask=mask, labels=lbls)
        loss   = out.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Train Loss: {avg_loss:.4f}")

    # --- Evaluation ---
    model.eval()
    train_preds, train_true = evaluate_loader(train_loader)
    val_preds,   val_true   = evaluate_loader(test_loader)

    train_acc = accuracy_score(train_true, train_preds)
    val_acc   = accuracy_score(val_true,   val_preds)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val   Accuracy: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_bart_agnews.pt")
        print("✅ Best model saved!")

# ── STEP 11: Final Evaluation & Plots ───────────────────────
print("\n\n===== FINAL EVALUATION =====")
model.load_state_dict(torch.load("best_bart_agnews.pt"))
model.eval()

val_preds, val_true = evaluate_loader(test_loader)

# Classification Report
print("\nClassification Report:")
print(classification_report(val_true, val_preds, labels=CLASS_NAMES))

# Save report to file
with open("classification_report.txt", "w") as f:
    f.write(classification_report(val_true, val_preds, labels=CLASS_NAMES))
print("Saved: classification_report.txt")

# Confusion Matrix
plot_confusion_matrix(val_true, val_preds,
                      title="Validation Confusion Matrix",
                      filename="confusion_matrix.png")

# ROC Curve (multiclass)
plot_roc_curve_multiclass(val_true, val_preds,
                          title="Validation ROC Curve (One-vs-Rest)",
                          filename="roc_curve.png")

# Loss & Accuracy Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, NUM_EPOCHS+1), loss_history, marker="o", color="crimson")
axes[0].set_title("Training Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

axes[1].plot(range(1, NUM_EPOCHS+1), train_acc_history, marker="o", label="Train", color="steelblue")
axes[1].plot(range(1, NUM_EPOCHS+1), val_acc_history,   marker="s", label="Validation", color="orange")
axes[1].set_title("Accuracy per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
print("Saved: training_curves.png")

print("\n✅ All done! Files saved:")
print("  - best_bart_agnews.pt       (trained model weights)")
print("  - classification_report.txt (precision / recall / F1)")
print("  - confusion_matrix.png")
print("  - roc_curve.png")
print("  - training_curves.png")