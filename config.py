# ============================================================================
# config.py - Configuration file
# ============================================================================
import os
import torch

CONFIG = {
    "TASK_PATH": "csi-bench/csi-bench-dataset/FallDetection",
    "TARGET_SUBCARRIERS": 232,
    "TARGET_TIME_SAMPLES": 500,
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 1e-4,
    "NUM_EPOCHS": 40,
    "WEIGHT_DECAY": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PRETRAINED_MODEL": "google/vit-base-patch16-224-in21k",
    
    # Generalization-focused settings
    "MIXUP_ALPHA": 0.3,
    "LABEL_SMOOTHING": 0.1,
    "DROPOUT_RATE": 0.4,
    "AUGMENTATION_STRENGTH": 0.7,
    
    # Domain adaptation
    "DOMAIN_ADAPTATION_WEIGHT": 0.15,
    "ADVERSARIAL_TRAINING": True,
    "DEVICE_AWARE_NORMALIZATION": True,
    
    # Advanced curriculum learning
    "CURRICULUM_STAGES": [
        {"name": "easy_patterns", "epochs": 6, "sampler": "balanced", "focus": "high_confidence"},
        {"name": "user_diversity", "epochs": 6, "sampler": "user_balanced", "focus": "cross_user"},
        {"name": "device_adaptation", "epochs": 8, "sampler": "device_balanced", "focus": "cross_device"},
        {"name": "environment_robust", "epochs": 8, "sampler": "env_balanced", "focus": "cross_env"},
        {"name": "hard_mining", "epochs": 8, "sampler": "hard_mining", "focus": "difficult_cases"},
        {"name": "full_integration", "epochs": 4, "sampler": "balanced", "focus": "final_polish"}
    ],
    
    # Paths
    "MODEL_SAVE_PATH": "checkpoints/fall_detection_best.pth",
    "CHECKPOINT_DIR": "checkpoints",
    "OUTPUT_DIR": "outputs",
    "PLOT_DIR": "outputs/plots",
    "LOG_DIR": "outputs/logs",
    
    # Logging
    "SAVE_BEST_MODEL": True,
    "LOG_INTERVAL": 20,
}

# Create directories
os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
os.makedirs(CONFIG["PLOT_DIR"], exist_ok=True)
os.makedirs(CONFIG["LOG_DIR"], exist_ok=True)

