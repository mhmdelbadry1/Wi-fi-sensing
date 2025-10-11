# ============================================================================
# main.py - Main execution script
# ============================================================================
import os
import sys
import time
from datetime import datetime, timedelta
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG
from src.data import prepare_data_loaders
from src.model import GeneralizableFallDetector
from src.train import train_with_curriculum, plot_training_history
from src.evaluate import evaluate_model

def format_time(seconds):
    """Format seconds into human readable time"""
    return str(timedelta(seconds=int(seconds)))

def main():
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(CONFIG["LOG_DIR"], f"training_log_{timestamp}.txt")
    
    def log_print(msg):
        """Print and log message"""
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
    
    log_print("\n" + "="*80)
    log_print("GENERALIZABLE FALL DETECTION MODEL")
    log_print("Production-Ready for ESP32-S3 Deployment")
    log_print("="*80)
    log_print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Log file: {log_file}")
    
    # Set seeds for reproducibility
    log_print("\n[SETUP] Setting random seeds...")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Device info
    device = CONFIG["DEVICE"]
    log_print(f"[SETUP] Using device: {device}")
    if torch.cuda.is_available():
        log_print(f"[SETUP] GPU: {torch.cuda.get_device_name(0)}")
        log_print(f"[SETUP] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Prepare data
    log_print("\n[1/4] Loading and preparing data...")
    data_start = time.time()
    
    (train_loader, val_loader, test_easy_loader,
     test_medium_loader, test_hard_loader, train_dataset) = prepare_data_loaders(CONFIG)
    
    data_time = time.time() - data_start
    log_print(f"✓ Data loading completed in {format_time(data_time)}")
    
    # Get number of unique users and environments for model
    num_users = len(train_dataset.metadata['user_label'].unique())
    num_envs = len(train_dataset.metadata['env_label'].unique())
    
    # Create model
    log_print("\n[2/4] Creating model...")
    model_start = time.time()
    
    model = GeneralizableFallDetector(
        pretrained_model=CONFIG["PRETRAINED_MODEL"],
        num_classes=2,
        num_devices=2,
        num_users=num_users,
        num_envs=num_envs,
        dropout=CONFIG["DROPOUT_RATE"],
        freeze_backbone=False
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_time = time.time() - model_start
    log_print(f"✓ Model created in {format_time(model_time)}")
    log_print(f"  Total parameters: {total_params:,}")
    log_print(f"  Trainable parameters: {trainable_params:,}")
    log_print(f"  Model size (approx): {total_params * 4 / 1e6:.2f} MB (FP32)")
    
    # Train
    log_print("\n[3/4] Training with curriculum learning...")
    train_start = time.time()
    
    history, best_val_acc = train_with_curriculum(
        model, train_dataset, val_loader, CONFIG, log_file=log_file
    )
    
    train_time = time.time() - train_start
    log_print(f"\n✓ Training completed in {format_time(train_time)}")
    log_print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    plot_path = os.path.join(CONFIG["PLOT_DIR"], f"training_history_{timestamp}.png")
    plot_training_history(history, plot_path)
    
    # Comprehensive evaluation
    log_print("\n[4/4] Evaluating on all test sets...")
    eval_start = time.time()
    
    log_print("\n" + "="*80)
    log_print("COMPREHENSIVE EVALUATION")
    log_print("="*80)
    
    test_results = {}
    
    log_print("\n--- Easy Test Set ---")
    easy_acc, _, _, easy_unc = evaluate_model(model, test_easy_loader, CONFIG, "Test Easy")
    test_results['easy'] = (easy_acc, easy_unc)
    
    log_print("\n--- Medium Test Set ---")
    med_acc, _, _, med_unc = evaluate_model(model, test_medium_loader, CONFIG, "Test Medium")
    test_results['medium'] = (med_acc, med_unc)
    
    log_print("\n--- Hard Test Set (Critical for Generalization) ---")
    hard_acc, _, _, hard_unc = evaluate_model(model, test_hard_loader, CONFIG, "Test Hard")
    test_results['hard'] = (hard_acc, hard_unc)
    
    eval_time = time.time() - eval_start
    log_print(f"\n✓ Evaluation completed in {format_time(eval_time)}")
    
    # Final Summary
    total_time = time.time() - data_start
    
    log_print("\n" + "="*80)
    log_print("FINAL SUMMARY")
    log_print("="*80)
    log_print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    log_print(f"\nTest Set Performance:")
    log_print(f"  Easy:   {test_results['easy'][0]*100:.2f}% (Uncertainty: {test_results['easy'][1]:.4f})")
    log_print(f"  Medium: {test_results['medium'][0]*100:.2f}% (Uncertainty: {test_results['medium'][1]:.4f})")
    log_print(f"  Hard:   {test_results['hard'][0]*100:.2f}% (Uncertainty: {test_results['hard'][1]:.4f})")
    
    generalization_score = (test_results['easy'][0] + test_results['medium'][0] +
                            test_results['hard'][0] * 2) / 4
    log_print(f"\nGeneralization Score: {generalization_score*100:.2f}%")
    
    if test_results['hard'][0] > 0.70:
        log_print("\n✓ EXCELLENT: Model shows strong generalization capability!")
        log_print("  Ready for diverse deployment scenarios including ESP32-S3")
    elif test_results['hard'][0] > 0.60:
        log_print("\n✓ GOOD: Model has decent generalization.")
        log_print("  Consider additional training on edge-case data for ESP32-S3")
    else:
        log_print("\n⚠ WARNING: Model may struggle with out-of-distribution data.")
        log_print("  Recommend more diverse training data before ESP32-S3 deployment")
    
    # Time breakdown
    log_print("\n" + "="*80)
    log_print("TIME BREAKDOWN")
    log_print("="*80)
    log_print(f"Data Loading:  {format_time(data_time)}")
    log_print(f"Model Creation: {format_time(model_time)}")
    log_print(f"Training:       {format_time(train_time)}")
    log_print(f"Evaluation:     {format_time(eval_time)}")
    log_print(f"{'='*80}")
    log_print(f"TOTAL TIME:     {format_time(total_time)}")
    log_print(f"{'='*80}")
    
    log_print("\n" + "="*80)
    log_print("MODEL READY FOR PRODUCTION")
    log_print(f"Model saved at: {CONFIG['MODEL_SAVE_PATH']}")
    log_print(f"Outputs saved in: {CONFIG['OUTPUT_DIR']}")
    log_print(f"Plots saved in: {CONFIG['PLOT_DIR']}")
    log_print(f"Log saved at: {log_file}")
    log_print("="*80)
    log_print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model, history, test_results

if __name__ == '__main__':
    try:
        model, history, results = main()
        print("\n✓ All tasks completed successfully!")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)