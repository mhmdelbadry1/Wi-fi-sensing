# ============================================================================
# src/train.py - Training Functions
# ============================================================================
import os
import random
import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from config import CONFIG
from src.utils import mixup_data, mixup_criterion

def train_with_curriculum(model, train_dataset, val_loader, config, log_file=None):
    """Train model with curriculum learning"""
    device = config["DEVICE"]
    model = model.to(device)
    
    # Loss functions
    criterion_task = nn.CrossEntropyLoss(label_smoothing=config["LABEL_SMOOTHING"])
    criterion_domain = nn.CrossEntropyLoss()
    
    # Optimizer
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
    
    def log_print(msg):
        """Print and log message"""
        print(msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
    
    log_print("\n" + "="*80)
    log_print("GENERALIZATION-FOCUSED CURRICULUM TRAINING")
    log_print("="*80)
    
    from src.data import create_curriculum_sampler
    from torch.utils.data import DataLoader
    
    for stage_idx, stage in enumerate(config["CURRICULUM_STAGES"]):
        log_print(f"\n{'='*80}")
        log_print(f"STAGE {stage_idx + 1}/{len(config['CURRICULUM_STAGES'])}: {stage['name'].upper()}")
        log_print(f"Focus: {stage['focus']}")
        log_print(f"Epochs: {stage['epochs']}")
        log_print(f"{'='*80}")
        
        # Create stage-specific sampler
        sampler = create_curriculum_sampler(train_dataset, stage)
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
            
            epoch_msg = (f"Epoch {epoch+1}/{stage['epochs']}: "
                        f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
            log_print(epoch_msg)
            
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
                    log_print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        # Save stage checkpoint
        checkpoint_path = os.path.join(config["CHECKPOINT_DIR"], f"stage_{stage_idx}_{stage['name']}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        log_print(f"\n✓ Stage {stage['name']} completed. Checkpoint saved.")
    
    return history, best_val_acc

def validate(model, val_loader, criterion, device):
    """Validate model"""
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

def plot_training_history(history, save_path):
    """Plot and save training curves"""
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training history plot saved to {save_path}")


