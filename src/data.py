# src/data.py
import os
import glob
import random
import numpy as np
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from config import CONFIG
from .utils import get_split_ids

class GeneralizableCSIDataset(Dataset):
    def __init__(self, metadata_df, augment=False, target_shape=(232, 500),
                 augmentation_strength=0.5, device_norm=False):
        self.metadata = metadata_df.reset_index(drop=True).copy()
        self.label_map = {'Nonfall': 0, 'Fall': 1}
        self.device_map = {'HP': 0, 'ESP32': 1}
        self.metadata['label'] = self.metadata['label'].map(self.label_map)
        self.metadata['device_label'] = self.metadata['device'].apply(
            lambda x: self.device_map.get(x, 0))

        # Environment mapping
        unique_envs = self.metadata['environment'].unique()
        self.env_map = {env: i for i, env in enumerate(unique_envs)}
        self.metadata['env_label'] = self.metadata['environment'].map(self.env_map)

        # User mapping
        unique_users = self.metadata['user'].unique()
        self.user_map = {user: i for i, user in enumerate(unique_users)}
        self.metadata['user_label'] = self.metadata['user'].map(self.user_map)

        self.augment = augment
        self.target_shape = target_shape
        self.augmentation_strength = augmentation_strength
        self.device_norm = device_norm

        # Compute device-specific normalization statistics
        if device_norm:
            self._compute_device_stats()

        self._filter_existing_files()

    def _compute_device_stats(self):
        """Compute mean/std for each device type"""
        self.device_stats = {}
        for device in ['HP', 'ESP32']:
            device_data = []
            device_samples = self.metadata[self.metadata['device'] == device].head(50)

            for _, row in device_samples.iterrows():
                try:
                    relative_path = row['file_path'].lstrip('./')
                    full_path = os.path.join(CONFIG["TASK_PATH"], relative_path)
                    if os.path.exists(full_path):
                        with h5py.File(full_path, 'r') as f:
                            arr = np.array(f['CSI_amps'])
                            device_data.append(arr.flatten())
                except:
                    continue

            if device_data:
                all_data = np.concatenate(device_data)
                self.device_stats[device] = {
                    'mean': np.mean(all_data),
                    'std': np.std(all_data) + 1e-6
                }
            else:
                self.device_stats[device] = {'mean': 0.0, 'std': 1.0}

    def _filter_existing_files(self):
        valid_indices = []
        for idx, row in self.metadata.iterrows():
            relative_path = row['file_path'].lstrip('./')
            full_file_path = os.path.join(CONFIG["TASK_PATH"], relative_path)

            if not os.path.exists(full_file_path):
                basename = os.path.basename(full_file_path)
                candidates = glob.glob(os.path.join(CONFIG["TASK_PATH"], "**", basename),
                                     recursive=True)
                if candidates:
                    self.metadata.at[idx, 'file_path'] = os.path.relpath(
                        candidates[0], CONFIG["TASK_PATH"])
                    valid_indices.append(idx)
            else:
                valid_indices.append(idx)

        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)

    def _load_and_standardize_csi(self, file_path, device):
        with h5py.File(file_path, 'r') as f:
            arr = np.array(f['CSI_amps'])
            arr = np.squeeze(arr)

        if arr.ndim == 1:
            arr = arr[:, None]

        target_c, target_t = self.target_shape

        # Standardize dimensions
        if arr.shape[0] > target_c:
            arr = arr[:target_c, :]
        elif arr.shape[0] < target_c:
            pad_c = target_c - arr.shape[0]
            arr = np.pad(arr, ((0, pad_c), (0, 0)), mode='edge')

        if arr.shape[1] > target_t:
            arr = arr[:, :target_t]
        elif arr.shape[1] < target_t:
            pad_t = target_t - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, pad_t)), mode='edge')

        # Device-aware normalization
        if self.device_norm and device in self.device_stats:
            stats = self.device_stats[device]
            arr = (arr - stats['mean']) / stats['std']
        else:
            # Standard normalization
            arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-6)

        return arr.astype(np.float32)

    def _apply_strong_augmentation(self, csi_data):
        """Aggressive augmentation for generalization"""
        strength = self.augmentation_strength

        # Time domain augmentations
        if random.random() < 0.5 * strength:
            # Time shift
            shift = random.randint(-50, 50)
            csi_data = np.roll(csi_data, shift, axis=1)

        if random.random() < 0.4 * strength:
            # Time stretching
            scale = random.uniform(0.7, 1.3)
            new_len = int(csi_data.shape[1] * scale)
            if new_len > 0:
                indices = np.linspace(0, csi_data.shape[1] - 1, new_len).astype(int)
                csi_data = csi_data[:, indices]
                # Restore to target size
                if csi_data.shape[1] < self.target_shape[1]:
                    pad = self.target_shape[1] - csi_data.shape[1]
                    csi_data = np.pad(csi_data, ((0, 0), (0, pad)), mode='edge')
                else:
                    csi_data = csi_data[:, :self.target_shape[1]]

        # Frequency domain augmentations
        if random.random() < 0.4 * strength:
            # Frequency masking
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                mask_size = random.randint(10, 30)
                f0 = random.randint(0, max(0, csi_data.shape[0] - mask_size))
                csi_data[f0:f0 + mask_size, :] *= random.uniform(0, 0.3)

        if random.random() < 0.3 * strength:
            # Frequency scaling
            scale = random.uniform(0.8, 1.2)
            new_len = int(csi_data.shape[0] * scale)
            if new_len > 0:
                indices = np.linspace(0, csi_data.shape[0] - 1, new_len).astype(int)
                csi_data = csi_data[indices, :]
                if csi_data.shape[0] < self.target_shape[0]:
                    pad = self.target_shape[0] - csi_data.shape[0]
                    csi_data = np.pad(csi_data, ((0, pad), (0, 0)), mode='edge')
                else:
                    csi_data = csi_data[:self.target_shape[0], :]

        # Noise augmentations
        if random.random() < 0.5 * strength:
            # Gaussian noise
            noise_level = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_level, csi_data.shape).astype(np.float32)
            csi_data = csi_data + noise

        if random.random() < 0.3 * strength:
            # Amplitude scaling (simulates different device sensitivities)
            scale = random.uniform(0.7, 1.3)
            csi_data = csi_data * scale

        if random.random() < 0.2 * strength:
            # Random dropout
            dropout_mask = np.random.binomial(1, 0.9, csi_data.shape)
            csi_data = csi_data * dropout_mask

        return csi_data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        relative_path = row['file_path'].lstrip('./')
        full_file_path = os.path.join(CONFIG["TASK_PATH"], relative_path)

        if not os.path.exists(full_file_path):
            basename = os.path.basename(full_file_path)
            candidates = glob.glob(os.path.join(CONFIG["TASK_PATH"], "**", basename),
                                 recursive=True)
            if candidates:
                full_file_path = candidates[0]
            else:
                return self.__getitem__(0)

        device = row['device']
        csi = self._load_and_standardize_csi(full_file_path, device)

        label = int(row['label'])
        device_label = int(row['device_label'])
        user_label = int(row['user_label'])
        env_label = int(row['env_label'])

        if self.augment:
            csi = self._apply_strong_augmentation(csi)

        tensor = __import__('torch').from_numpy(csi).unsqueeze(0).float()
        return tensor, label, device_label, user_label, env_label

def create_curriculum_sampler(dataset, stage_config):
    """Create sampler based on curriculum stage"""
    sampler_type = stage_config["sampler"]
    metadata = dataset.metadata

    if sampler_type == "balanced":
        # Standard balanced sampling
        label_counts = metadata['label'].value_counts().to_dict()
        weights = [1.0 / label_counts.get(label, 1) for label in metadata['label']]
        return WeightedRandomSampler(weights, len(weights))

    elif sampler_type == "user_balanced":
        # Balance across users
        user_counts = metadata['user_label'].value_counts().to_dict()
        weights = [1.0 / user_counts.get(user, 1) for user in metadata['user_label']]
        return WeightedRandomSampler(weights, len(weights))

    elif sampler_type == "device_balanced":
        # Balance across devices (critical for ESP32)
        device_counts = metadata['device_label'].value_counts().to_dict()
        weights = [1.0 / device_counts.get(dev, 1) for dev in metadata['device_label']]
        return WeightedRandomSampler(weights, len(weights))

    elif sampler_type == "env_balanced":
        # Balance across environments
        env_counts = metadata['env_label'].value_counts().to_dict()
        weights = [1.0 / env_counts.get(env, 1) for env in metadata['env_label']]
        return WeightedRandomSampler(weights, len(weights))

    elif sampler_type == "hard_mining":
        # Focus on difficult samples (implement simple heuristic)
        # Prefer ESP32 samples and minority class
        weights = []
        for _, row in metadata.iterrows():
            weight = 1.0
            if row['device_label'] == 1:  # ESP32
                weight *= 3.0
            if row['label'] == 1:  # Fall (usually minority)
                weight *= 2.0
            weights.append(weight)
        return WeightedRandomSampler(weights, len(weights))

    else:
        return None

def prepare_data_loaders(config):
    meta_csv = os.path.join(config["TASK_PATH"], "metadata", "sample_metadata.csv")
    full_df = pd.read_csv(meta_csv)

    print("Dataset Statistics:")
    print(f"Total samples: {len(full_df)}")
    print(f"Labels: {full_df['label'].value_counts().to_dict()}")
    print(f"Devices: {full_df['device'].value_counts().to_dict()}")
    print(f"Users: {len(full_df['user'].unique())} unique users")
    print(f"Environments: {len(full_df['environment'].unique())} unique environments")

    # Get splits
    train_ids = get_split_ids(config["TASK_PATH"], "train_id")
    val_ids = get_split_ids(config["TASK_PATH"], "val_id")
    test_easy_ids = get_split_ids(config["TASK_PATH"], "test_easy")
    test_medium_ids = get_split_ids(config["TASK_PATH"], "test_medium")
    test_hard_ids = get_split_ids(config["TASK_PATH"], "test_hard")

    train_df = full_df[full_df['id'].isin(train_ids)].copy()
    val_df = full_df[full_df['id'].isin(val_ids)].copy()
    test_easy_df = full_df[full_df['id'].isin(test_easy_ids)].copy()
    test_medium_df = full_df[full_df['id'].isin(test_medium_ids)].copy()
    test_hard_df = full_df[full_df['id'].isin(test_hard_ids)].copy()

    print(f"\nSplit sizes:")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Test Easy: {len(test_easy_df)}, Medium: {len(test_medium_df)}, Hard: {len(test_hard_df)}")

    # Create datasets
    target_shape = (config["TARGET_SUBCARRIERS"], config["TARGET_TIME_SAMPLES"])
    train_dataset = GeneralizableCSIDataset(
        train_df, augment=True, target_shape=target_shape,
        augmentation_strength=config["AUGMENTATION_STRENGTH"],
        device_norm=config["DEVICE_AWARE_NORMALIZATION"]
    )
    val_dataset = GeneralizableCSIDataset(
        val_df, augment=False, target_shape=target_shape,
        device_norm=config["DEVICE_AWARE_NORMALIZATION"]
    )
    test_easy_dataset = GeneralizableCSIDataset(
        test_easy_df, augment=False, target_shape=target_shape,
        device_norm=config["DEVICE_AWARE_NORMALIZATION"]
    )
    test_medium_dataset = GeneralizableCSIDataset(
        test_medium_df, augment=False, target_shape=target_shape,
        device_norm=config["DEVICE_AWARE_NORMALIZATION"]
    )
    test_hard_dataset = GeneralizableCSIDataset(
        test_hard_df, augment=False, target_shape=target_shape,
        device_norm=config["DEVICE_AWARE_NORMALIZATION"]
    )

    # Balanced sampler
    label_counts = train_df['label'].value_counts().to_dict()
    class_weights = [1.0 / label_counts.get(label, 1) for label in train_dataset.metadata['label']]
    sampler = WeightedRandomSampler(class_weights, len(class_weights))

    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"],
                             sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"],
                           shuffle=False, num_workers=4, pin_memory=True)
    test_easy_loader = DataLoader(test_easy_dataset, batch_size=config["BATCH_SIZE"],
                                 shuffle=False, num_workers=4)
    test_medium_loader = DataLoader(test_medium_dataset, batch_size=config["BATCH_SIZE"],
                                   shuffle=False, num_workers=4)
    test_hard_loader = DataLoader(test_hard_dataset, batch_size=config["BATCH_SIZE"],
                                 shuffle=False, num_workers=4)

    return (train_loader, val_loader, test_easy_loader,
            test_medium_loader, test_hard_loader, train_dataset)
