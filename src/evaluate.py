# ============================================================================
# src/evaluate.py - Evaluation Functions
# ============================================================================
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from src.utils import evaluate_predictions

def evaluate_model(model, test_loader, config, test_name="Test", device=None):
    """Evaluate model on test set"""
    device = device or config['DEVICE']
    
    # ✅ CRITICAL: Load the best saved model
    if os.path.exists(config["MODEL_SAVE_PATH"]):
        print(f"Loading best model from {config['MODEL_SAVE_PATH']}...")
        checkpoint = torch.load(config["MODEL_SAVE_PATH"], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model (Val Acc: {checkpoint['val_acc']:.2f}%)")
        print(f"  Stage: {checkpoint['stage']}, Epoch: {checkpoint['epoch']}")
    else:
        print(f"⚠ WARNING: No saved model found at {config['MODEL_SAVE_PATH']}")
        print("   Evaluating with current model state (may not be optimal)")
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_uncertainties = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {test_name}"):
            # Handle variable batch outputs
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            
            # Get model predictions
            task_out, _, _, _, uncertainty = model(inputs)
            preds = torch.argmax(task_out, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Fix: Handle both single values and batches
            unc_values = uncertainty.cpu().squeeze().tolist()
            if isinstance(unc_values, float):
                all_uncertainties.append(unc_values)
            else:
                all_uncertainties.extend(unc_values)
    
    # Compute metrics
    out_prefix = os.path.join(config["OUTPUT_DIR"], test_name.lower().replace(' ', '_'))
    acc, report, cm = evaluate_predictions(
        all_labels, all_preds, 
        out_prefix=out_prefix,
        plot_dir=config["PLOT_DIR"]
    )
    
    avg_uncertainty = np.mean(all_uncertainties)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"{test_name.upper()} SET EVALUATION")
    print(f"{'='*80}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Average Uncertainty: {avg_uncertainty:.4f}")
    print("\nClassification Report:")
    print(report)
    
    return acc, report, cm, avg_uncertainty

