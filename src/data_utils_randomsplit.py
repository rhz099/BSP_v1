# src/data_utils_randomsplit.py

import torch
import numpy as np
from sklearn.model_selection import train_test_split

def random_split(data, test_size=0.3, random_state=None):
    labeled_mask = data.y != -1
    labeled_indices = torch.where(labeled_mask)[0].cpu().numpy()
    labels = data.y[labeled_indices].cpu().numpy()

    train_idx_np, val_idx_np = train_test_split(
        labeled_indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state  # <-- pass the outer seed here
    )

    train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=data.x.device)
    val_idx = torch.tensor(val_idx_np, dtype=torch.long, device=data.x.device)

    return train_idx, val_idx

