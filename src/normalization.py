import torch

def normalize_base_features_only(x, feature_names):
    """
    Normalize only base features identified by name prefix.
    Args:
        x (torch.Tensor): Feature matrix [N, D]
        feature_names (List[str]): Corresponding column names
    Returns:
        torch.Tensor: Normalized feature matrix
    """
    x_norm = x.clone()
    base_indices = [i for i, name in enumerate(feature_names) if name.startswith("base_feat_")]

    if not base_indices:
        print("No base features found to normalize.")
        return x_norm

    base_tensor = x[:, base_indices]
    mean = base_tensor.mean(dim=0, keepdim=True)
    std = base_tensor.std(dim=0, keepdim=True) + 1e-6

    x_norm[:, base_indices] = (base_tensor - mean) / std
    return x_norm

def normalize_engineered_features(x, feature_names, target_features):
    """
    Normalize selected engineered features by name.
    """
    x_norm = x.clone()
    for fname in target_features:
        if fname in feature_names:
            idx = feature_names.index(fname)
            feat = x[:, idx]
            mean = feat.mean()
            std = feat.std() + 1e-6
            x_norm[:, idx] = (feat - mean) / std
    return x_norm