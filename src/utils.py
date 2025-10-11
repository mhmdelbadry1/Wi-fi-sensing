# src/utils.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

def get_split_ids(task_path, split_name):
    split_file = os.path.join(task_path, "splits", f"{split_name}.json")
    with open(split_file, 'r') as f:
        split_ids = set(json.load(f))
    return split_ids

def mixup_data(x, y, alpha=0.3):
    """Apply mixup augmentation for better generalization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = __import__('torch').randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def evaluate_predictions(y_true, y_pred, out_prefix="outputs/eval"):
    """
    Compute accuracy, classification report, confusion matrix and save results.
    Returns accuracy, report_str, conf_matrix.
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Non-Fall','Fall'], zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save classification report
    with open(f"{out_prefix}_report.txt", "w") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)

    # Plot and save confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fall','Fall'], yticklabels=['Non-Fall','Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confusion.png", dpi=150)
    plt.close()

    return acc, report, cm
