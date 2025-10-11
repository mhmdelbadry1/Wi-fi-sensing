# config.py
import os
import torch

CONFIG = {
    "TASK_PATH": "csi-bench/csi-bench-dataset/FallDetection",
    "TARGET_SUBCARRIERS": 232,
    "TARGET_TIME_SAMPLES": 500,
    "BATCH_SIZE": 16,  # Increased for better generalization
    "LEARNING_RATE": 1e-4,  # Conservative learning rate
    "NUM_EPOCHS": 40,
    "WEIGHT_DECAY": 1e-4,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PRETRAINED_MODEL": "google/vit-base-patch16-224-in21k",

    # Generalization-focused settings
    "MIXUP_ALPHA": 0.3,  # Data mixing for robustness
    "LABEL_SMOOTHING": 0.1,  # Prevent overconfidence
    "DROPOUT_RATE": 0.4,  # Strong regularization
    "AUGMENTATION_STRENGTH": 0.7,  # Aggressive augmentation

    # Domain adaptation
    "DOMAIN_ADAPTATION_WEIGHT": 0.15,  # Balance task and domain
    "ADVERSARIAL_TRAINING": True,
    "DEVICE_AWARE_NORMALIZATION": True,

    # Cross-validation and ensembling
    "USE_CROSS_VALIDATION": True,
    "NUM_FOLDS": 5,
    "ENSEMBLE_MODELS": 3,

    # Advanced curriculum learning
    "CURRICULUM_STAGES": [
        {"name": "easy_patterns", "epochs": 6, "sampler": "balanced", "focus": "high_confidence"},
        {"name": "user_diversity", "epochs": 6, "sampler": "user_balanced", "focus": "cross_user"},
        {"name": "device_adaptation", "epochs": 8, "sampler": "device_balanced", "focus": "cross_device"},
        {"name": "environment_robust", "epochs": 8, "sampler": "env_balanced", "focus": "cross_env"},
        {"name": "hard_mining", "epochs": 8, "sampler": "hard_mining", "focus": "difficult_cases"},
        {"name": "full_integration", "epochs": 4, "sampler": "balanced", "focus": "final_polish"}
    ],

    # Save and logging
    "SAVE_BEST_MODEL": True,
    "LOG_INTERVAL": 20,
    "MODEL_SAVE_PATH": "fall_detection_generalizable.pth",
    "CHECKPOINT_DIR": "checkpoints"
}

os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)
os.makedirs("outputs", exist_ok=True)
