import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from torchviz import make_dot
from collections import Counter

# 1. Load the dataset from folders
# NOTE: Update these paths for your local Windows environment if necessary
recommended_dir = '/content/drive/MyDrive/music final round/dataset/recommended'
not_recommended_dir = '/content/drive/MyDrive/music final round/dataset/not recommended'

def load_data_from_directory(directory, label):
    texts = []
    labels = []
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} not found.")
        return texts, labels
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                texts.append(file.read())
            labels.append(label)
    return texts, labels

recommended_texts, recommended_labels = load_data_from_directory(recommended_dir, "recommended")
not_recommended_texts, not_recommended_labels = load_data_from_directory(not_recommended_dir, "not recommended")

texts = recommended_texts + not_recommended_texts
labels = recommended_labels + not_recommended_labels

if not texts:
    print("Error: No text files found. Please check your dataset paths.")
else:
    # Print overall class distribution
    print("Overall class distribution:", Counter(labels))

    # Split the data into training and testing sets (stratify to maintain class balance)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print("Training class distribution before oversampling:", Counter(train_labels))

    # 2. Oversample the minority class using RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    train_texts_resampled, train_labels_resampled = oversampler.fit_resample(
        np.array(train_texts).reshape(-1, 1), train_labels
    )
    train_texts_resampled = train_texts_resampled.flatten()
    print("Training class distribution after oversampling:", Counter(train_labels_resampled))

    # 3. Tokenize the data using a pre-trained BART tokenizer
    model_name = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_data(texts, labels, tokenizer):
        inputs = tokenizer([f"classify: {text}" for text in texts],
                           truncation=True, padding=True, max_length=512, return_tensors="pt")
        targets = tokenizer(labels, truncation=True, padding=True, max_length=10, return_tensors="pt")
        return inputs, targets

    train_encodings, train_targets = preprocess_data(train_texts_resampled, train_labels_resampled, tokenizer)
    test_encodings, test_targets = preprocess_data(test_texts, test_labels, tokenizer)

    # 4. Prepare a custom PyTorch Dataset
    class SongLyricsDataset(Dataset):
        def __init__(self, encodings, targets):
            self.encodings = encodings
            self.targets = targets
        def __len__(self):
            return self.targets['input_ids'].shape[0]
        def __getitem__(self, idx):
            item = {key: self.encodings[key][idx] for key in self.encodings.keys()}
            item['labels'] = self.targets['input_ids'][idx]
            return item

    train_dataset = SongLyricsDataset(train_encodings, train_targets)
    test_dataset = SongLyricsDataset(test_encodings, test_targets)

    # 5. Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 6. Load the pre-trained BART model for sequence-to-sequence tasks
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # 7. Define Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 10
    total_steps = len(train_loader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    # Helper functions for evaluation metrics
    def calculate_accuracy(labels, preds):
        return accuracy_score(labels, preds)

    def plot_confusion_matrix(labels, preds, title="Confusion Matrix"):
        cm = confusion_matrix(labels, preds, labels=["not recommended", "recommended"])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Not Recommended", "Recommended"],
                    yticklabels=["Not Recommended", "Recommended"])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        plt.show()

    def plot_roc_curve(labels, preds, title="ROC Curve"):
        labels_binary = [1 if label == "recommended" else 0 for label in labels]
        preds_binary = [1 if pred == "recommended" else 0 for pred in preds]
        fpr, tpr, _ = roc_curve(labels_binary, preds_binary)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:0.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    def per_class_accuracy(true_labels, pred_labels):
        classes = list(set(true_labels))
        accuracies = {}
        for cls in classes:
            indices = [i for i, label in enumerate(true_labels) if label == cls]
            cls_true = [true_labels[i] for i in indices]
            cls_pred = [pred_labels[i] for i in indices]
            accuracies[cls] = accuracy_score(cls_true, cls_pred)
        return accuracies

    def normalize_prediction(text):
        text = text.strip().lower()
        if text.startswith("not"):
            return "not recommended"
        else:
            return "recommended"

    # 8. Training and evaluation loop
    best_model_saved = False
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        print(f"Train loss: {train_loss:.4f}")

        model.eval()
        train_preds_raw, train_labels_decoded = [], []
        val_preds_raw, val_labels_decoded = [], []
        with torch.no_grad():
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
                train_preds_raw.extend([tokenizer.decode(g, skip_special_tokens=True) for g in outputs])
                train_labels_decoded.extend([tokenizer.decode(l, skip_special_tokens=True) for l in batch['labels']])
            
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=10)
                val_preds_raw.extend([tokenizer.decode(g, skip_special_tokens=True) for g in outputs])
                val_labels_decoded.extend([tokenizer.decode(l, skip_special_tokens=True) for l in batch['labels']])

        train_preds = [normalize_prediction(pred) for pred in train_preds_raw]
        val_preds = [normalize_prediction(pred) for pred in val_preds_raw]
        train_labels_norm = [normalize_prediction(label) for label in train_labels_decoded]
        val_labels_norm = [normalize_prediction(label) for label in val_labels_decoded]

        train_accuracy = calculate_accuracy(train_labels_norm, train_preds)
        val_accuracy = calculate_accuracy(val_labels_norm, val_preds)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        train_class_acc = per_class_accuracy(train_labels_norm, train_preds)
        val_class_acc = per_class_accuracy(val_labels_norm, val_preds)
        print("Training per-class accuracy:", train_class_acc)
        print("Validation per-class accuracy:", val_class_acc)

        print("Training Classification Report:")
        print(classification_report(train_labels_norm, train_preds, labels=["not recommended", "recommended"]))
        print("Validation Classification Report:")
        print(classification_report(val_labels_norm, val_preds, labels=["not recommended", "recommended"]))

        plot_confusion_matrix(train_labels_norm, train_preds, title="Training Confusion Matrix")
        plot_confusion_matrix(val_labels_norm, val_preds, title="Validation Confusion Matrix")
        plot_roc_curve(val_labels_norm, val_preds, title="Validation ROC Curve")

        if train_accuracy >= 0.99 and not best_model_saved:
            # Update path for local Windows environment
            model_save_path = "bart_song_lyrics_model_99.pt"
            torch.save(model.state_dict(), model_save_path)
            best_model_saved = True
            print(f"Model with 99% training accuracy saved to {model_save_path}")

    # 9. Plot the model architecture using torchviz
    def plot_model_architecture(model, tokenizer, device):
        dummy_texts = ["classify: This is a dummy text.", "classify: Another example."]
        dummy_inputs = tokenizer(dummy_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        input_ids = dummy_inputs['input_ids'].to(device)
        attention_mask = dummy_inputs['attention_mask'].to(device)
        outputs = model.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        dot = make_dot(outputs.last_hidden_state, params=dict(model.named_parameters()))
        dot.format = "png"
        dot.render("bart_encoder_architecture", view=True)
        print("Model architecture diagram saved as 'bart_encoder_architecture.png'.")

    plot_model_architecture(model, tokenizer, device)
