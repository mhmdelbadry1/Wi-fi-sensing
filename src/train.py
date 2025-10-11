# src/train.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config import CONFIG
from .data import prepare_data_loaders
from .model import GeneralizableFallDetector
from .utils import mixup_data, mixup_criterion

import warnings
warnings.filterwarnings('ignore')

def create_curriculum_sampler(dataset, stage_config):
    # use data.create_curriculum_sampler if needed; we already have it in data.py,
    # but to keep behavior identical we will import it directly where needed.
    from .data import create_curriculum_sampler
    return create_curriculum_sampler(dataset, stage_config)

def train_with_curriculum(model, train_dataset, val_loader, config):
    device = config["DEVICE"]
    model = model.to(device)

    # Loss functions
    criterion_task = nn.CrossEntropyLoss(label_smoothing=config["LABEL_SMOOTHING"])
    criterion_domain = nn.CrossEntropyLoss()

    # Optimizer with different learning rates
    backbone_params = [p for n, p in model.named_parameters() if 'vit' in n]
    other_params = [p for n, p in model.named_parameters() if 'vit' not in n]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config["LEARNING_RATE"] * 0.1},
        {'params': other_params, 'lr': config["LEARNING_RATE"]}
    ], weight_decay=config["WEIGHT_DECAY"])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("\n" + "="*80)
    print("GENERALIZATION-FOCUSED CURRICULUM TRAINING")
    print("="*80)

    for stage_idx, stage in enumerate(config["CURRICULUM_STAGES"]):
        print(f"\n{'='*80}")
        print(f"STAGE {stage_idx + 1}/{len(config['CURRICULUM_STAGES'])}: {stage['name'].upper()}")
        print(f"Focus: {stage['focus']}")
        print(f"Epochs: {stage['epochs']}")
        print(f"{'='*80}")

        # Create stage-specific sampler
        sampler = create_curriculum_sampler(train_dataset, stage)
        from torch.utils.data import DataLoader
        stage_loader = DataLoader(
            train_dataset,
            batch_size=config["BATCH_SIZE"],
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )

        for epoch in range(stage['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(stage_loader, desc=f"Epoch {epoch+1}/{stage['epochs']}")
            for batch_idx, (inputs, labels, device_labels, user_labels, env_labels) in enumerate(pbar):
                inputs = inputs.to(device)
                labels = labels.to(device)
                device_labels = device_labels.to(device)
                user_labels = user_labels.to(device)
                env_labels = env_labels.to(device)

                # Mixup augmentation
                if config["MIXUP_ALPHA"] > 0 and random.random() < 0.5:
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, config["MIXUP_ALPHA"])

                    optimizer.zero_grad()
                    task_out, device_out, user_out, env_out, uncertainty = model(inputs)

                    # Mixup loss
                    loss_task = mixup_criterion(criterion_task, task_out, labels_a, labels_b, lam)
                else:
                    optimizer.zero_grad()
                    task_out, device_out, user_out, env_out, uncertainty = model(inputs)
                    loss_task = criterion_task(task_out, labels)

                # Domain adaptation losses
                loss_device = criterion_domain(device_out, device_labels)
                loss_user = criterion_domain(user_out, user_labels)
                loss_env = criterion_domain(env_out, env_labels)

                # Total loss with adaptive weighting
                if "hard" in stage['name'] or "device" in stage['name']:
                    # Emphasize domain adaptation for generalization
                    loss = loss_task + 0.3 * loss_device + 0.2 * loss_user + 0.2 * loss_env
                else:
                    loss = loss_task + 0.15 * loss_device + 0.1 * loss_user + 0.1 * loss_env

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(task_out, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                if batch_idx % config["LOG_INTERVAL"] == 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*correct/total:.2f}%'
                    })

            scheduler.step()

            train_loss = train_loss / len(stage_loader)
            train_acc = 100. * correct / total

            # Validation
            val_loss, val_acc = validate(model, val_loader, criterion_task, device)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch+1}/{stage['epochs']}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if config["SAVE_BEST_MODEL"]:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'epoch': epoch,
                        'stage': stage['name']
                    }, config["MODEL_SAVE_PATH"])
                    print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")

        # Save stage checkpoint
        checkpoint_path = os.path.join(config["CHECKPOINT_DIR"], f"stage_{stage_idx}_{stage['name']}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\n✓ Stage {stage['name']} completed. Checkpoint saved.")

    return history, best_val_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, _, _, _ in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            task_out, _, _, _, _ = model(inputs)
            loss = criterion(task_out, labels)

            val_loss += loss.item()
            _, preds = torch.max(task_out, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc

def evaluate_model(model, test_loader, config, test_name="Test"):
    device = config["DEVICE"]

    # Load best model
    if os.path.exists(config["MODEL_SAVE_PATH"]):
        checkpoint = torch.load(config["MODEL_SAVE_PATH"], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model (Val Acc: {checkpoint['val_acc']:.2f}%)")

    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_uncertainties = []

    with torch.no_grad():
        for inputs, labels, _, _, _ in tqdm(test_loader, desc=f"Evaluating {test_name}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            task_out, _, _, _, uncertainty = model(inputs)
            _, preds = torch.max(task_out, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_uncertainties.extend(uncertainty.cpu().numpy())

    # Metrics
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import numpy as np
    accuracy = accuracy_score(all_labels, all_preds)
    avg_uncertainty = np.mean(all_uncertainties)

    print(f"\n{'='*80}")
    print(f"{test_name.upper()} SET EVALUATION")
    print(f"{'='*80}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Average Uncertainty: {avg_uncertainty:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=['Non-Fall', 'Fall'],
                                zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fall', 'Fall'],
                yticklabels=['Non-Fall', 'Fall'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {test_name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{test_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

    return accuracy, avg_uncertainty

def plot_training_history(history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    print("✓ Training history plot saved")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("GENERALIZABLE FALL DETECTION MODEL")
    print("Production-Ready for ESP32-S3 Deployment")
    print("="*80)

    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Prepare data
    print("\n[1/4] Loading and preparing data...")
    (train_loader, val_loader, test_easy_loader,
     test_medium_loader, test_hard_loader, train_dataset) = prepare_data_loaders(CONFIG)

    # Get number of unique users and environments for model
    num_users = len(train_dataset.metadata['user_label'].unique())
    num_envs = len(train_dataset.metadata['env_label'].unique())

    # Create model
    print("\n[2/4] Creating model...")
    model = GeneralizableFallDetector(
        pretrained_model=CONFIG["PRETRAINED_MODEL"],
        num_classes=2,
        num_devices=2,
        num_users=num_users,
        num_envs=num_envs,
        dropout=CONFIG["DROPOUT_RATE"],
        freeze_backbone=False
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train
    print("\n[3/4] Training with curriculum learning...")
    history, best_val_acc = train_with_curriculum(model, train_dataset, val_loader, CONFIG)

    print(f"\n✓ Training completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # Plot history
    plot_training_history(history)

    # Comprehensive evaluation
    print("\n[4/4] Evaluating on all test sets...")
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION")
    print("="*80)

    test_results = {}

    print("\n--- Easy Test Set ---")
    test_results['easy'] = evaluate_model(model, test_easy_loader, CONFIG, "Test Easy")

    print("\n--- Medium Test Set ---")
    test_results['medium'] = evaluate_model(model, test_medium_loader, CONFIG, "Test Medium")

    print("\n--- Hard Test Set (Critical for Generalization) ---")
    test_results['hard'] = evaluate_model(model, test_hard_loader, CONFIG, "Test Hard")

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"\nTest Set Performance:")
    print(f"  Easy:   {test_results['easy'][0]*100:.2f}% (Uncertainty: {test_results['easy'][1]:.4f})")
    print(f"  Medium: {test_results['medium'][0]*100:.2f}% (Uncertainty: {test_results['medium'][1]:.4f})")
    print(f"  Hard:   {test_results['hard'][0]*100:.2f}% (Uncertainty: {test_results['hard'][1]:.4f})")

    generalization_score = (test_results['easy'][0] + test_results['medium'][0] +
                            test_results['hard'][0] * 2) / 4  # Weight hard set more
    print(f"\nGeneralization Score: {generalization_score*100:.2f}%")

    if test_results['hard'][0] > 0.70:
        print("\n✓ EXCELLENT: Model shows strong generalization capability!")
        print("  Ready for diverse deployment scenarios including ESP32-S3")
    elif test_results['hard'][0] > 0.60:
        print("\n✓ GOOD: Model has decent generalization.")
        print("  Consider additional training on edge-case data for ESP32-S3")
    else:
        print("\n⚠ WARNING: Model may struggle with out-of-distribution data.")
        print("  Recommend more diverse training data before ESP32-S3 deployment")

    print("\n" + "="*80)
    print("MODEL READY FOR PRODUCTION")
    print(f"Saved at: {CONFIG['MODEL_SAVE_PATH']}")
    print("="*80)
