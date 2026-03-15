import os
import random
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit


def _dataset_name(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path if save_path.endswith('/') else save_path + '/'


def normalize(data, min_val=None, max_val=None):
    # If min and max values are not provided, calculate them from the data
    if min_val is None or max_val is None:
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)

    # Normalize
    normalized_data = (data - min_val) / (max_val - min_val)

    # Replace NaN values in normalized_data with 1.0
    normalized_data = np.nan_to_num(normalized_data, nan=1.0)

    return normalized_data, min_val, max_val


def mix_out(x, y, groups, n_splits, test_size, seed=42):
    mix_out_list = []
    kfold = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    for train_idx, test_idx in kfold.split(x, y, groups):
        mix_out_list.append((train_idx, test_idx))
    return mix_out_list


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

