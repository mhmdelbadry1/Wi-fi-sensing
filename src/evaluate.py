# src/evaluate.py
import torch
import numpy as np
from .utils import evaluate_predictions
from config import CONFIG
from tqdm.auto import tqdm

def evaluate_model(model, test_loader, config, device=None):
    device = device or config['DEVICE']
    if torch.cuda.is_available() and device.startswith('cuda'):
        model.to(device)
    model.eval()
    all_preds=[]; all_labels=[]
    with torch.no_grad():
        for inputs, labels, *_ in tqdm(test_loader, desc="Eval"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            task_out, *_ = model(inputs)
            preds = torch.argmax(task_out, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc, report, cm = evaluate_predictions(all_labels, all_preds, out_prefix="outputs/eval")
    print("Accuracy:", acc); print(report)
    return acc, report, cm
