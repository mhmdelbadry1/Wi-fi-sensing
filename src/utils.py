# ============================================================================
# src/utils.py - Utility functions
# ============================================================================
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import torch

def get_split_ids(task_path, split_name):
    """Load split IDs from JSON file"""
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
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def evaluate_predictions(y_true, y_pred, out_prefix="outputs/eval", plot_dir="outputs/plots"):
    """
    Compute accuracy, classification report, confusion matrix and save results.
    Returns accuracy, report_str, conf_matrix.
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Non-Fall', 'Fall'], zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Save classification report
    with open(f"{out_prefix}_report.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fall', 'Fall'], yticklabels=['Non-Fall', 'Fall'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    # Save to plot directory
    plot_filename = os.path.basename(out_prefix) + "_confusion.png"
    plt.savefig(os.path.join(plot_dir, plot_filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    return acc, report, cm
