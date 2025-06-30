# src/features/base_features.py

import torch
import numpy as np

def compute_base_features(data):
    """
    Extracts base node features from the PyG data object.

    Args:
        data (torch_geometric.data.Data): The PyG data object

    Returns:
        np.ndarray: [num_nodes, num_features] feature matrix
    """
    return data.x.cpu().numpy()

def base_feature_names(num_features=166):
    """
    Generates standardized feature names for base features.

    Args:
        num_features (int): Number of base features. Default = 166.

    Returns:
        List[str]: Feature names ["base_feat_0", ..., "base_feat_{num_features-1}"]
    """
    return [f"base_feat_{i}" for i in range(num_features)]
