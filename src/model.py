# src/model.py
import torch
import torch.nn as nn
from transformers import ViTModel
from config import CONFIG

class CSIToRGBConverter(nn.Module):
    def __init__(self, target_size=(224, 224), dropout=0.3):
        super().__init__()
        self.target_size = target_size

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(3)

        self.dropout = nn.Dropout2d(dropout)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(target_size)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return x

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val=1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)

    def update_lambda(self, lambda_val):
        self.lambda_val = lambda_val

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None

class GeneralizableFallDetector(nn.Module):
    """
    Production-ready fall detection model with strong generalization
    """
    def __init__(self, pretrained_model="google/vit-base-patch16-224-in21k",
                 num_classes=2, num_devices=2, num_users=20, num_envs=10,
                 dropout=0.4, freeze_backbone=False):
        super().__init__()

        # ViT backbone
        self.vit = ViTModel.from_pretrained(pretrained_model)

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.hidden_size = self.vit.config.hidden_size

        # CSI to RGB converter
        self.csi_to_rgb = CSIToRGBConverter(target_size=(224, 224), dropout=dropout)

        # Multi-head attention for feature refinement
        self.feature_attention = nn.MultiheadAttention(
            self.hidden_size, num_heads=8, dropout=dropout)
        self.attention_norm = nn.LayerNorm(self.hidden_size)

        # Task classifier (main fall detection)
        self.task_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # Domain adversarial classifiers
        self.device_discriminator = nn.Sequential(
            GradientReversalLayer(CONFIG["DOMAIN_ADAPTATION_WEIGHT"]),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_devices)
        )

        self.user_discriminator = nn.Sequential(
            GradientReversalLayer(CONFIG["DOMAIN_ADAPTATION_WEIGHT"]),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_users)
        )

        self.env_discriminator = nn.Sequential(
            GradientReversalLayer(CONFIG["DOMAIN_ADAPTATION_WEIGHT"]),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_envs)
        )

        # Uncertainty estimation for out-of-distribution detection
        self.uncertainty = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, return_features=False):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # Convert to RGB
        x_rgb = self.csi_to_rgb(x)

        # ViT features
        outputs = self.vit(pixel_values=x_rgb)
        cls_features = outputs.last_hidden_state[:, 0]

        # Attention refinement
        cls_reshaped = cls_features.unsqueeze(0)
        attn_out, _ = self.feature_attention(cls_reshaped, cls_reshaped, cls_reshaped)
        refined_features = self.attention_norm(attn_out.squeeze(0) + cls_features)

        # Predictions
        task_out = self.task_classifier(refined_features)
        device_out = self.device_discriminator(refined_features)
        user_out = self.user_discriminator(refined_features)
        env_out = self.env_discriminator(refined_features)
        uncertainty_out = self.uncertainty(refined_features)

        if return_features:
            return task_out, device_out, user_out, env_out, uncertainty_out, refined_features
        return task_out, device_out, user_out, env_out, uncertainty_out

    def update_domain_weight(self, weight):
        for module in self.modules():
            if isinstance(module, GradientReversalLayer):
                module.update_lambda(weight)
