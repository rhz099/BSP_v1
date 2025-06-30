# src/features/feature_runner.py

import torch
from .feature_pipeline import generate_all_features

def apply_engineered_features(data, G_nx, node_timestamps, config):
    """
    Applies feature pipeline based on configuration flags,
    and replaces data.x with the resulting engineered feature matrix.

    Args:
        data (torch_geometric.data.Data): PyG graph object
        G_nx (networkx.Graph): Full directed transaction graph (undirected does not work)
        node_timestamps (List[List[int]]): per-node timestamp history
        config (dict): feature config dictionary (from feature_config.py)

    Returns:
        torch_geometric.data.Data: updated with data.x as new features
    """
    X_feats, feat_names = generate_all_features(
        data=data,
        nx_graph=G_nx,
        node_timestamps=node_timestamps,
        **config
    )

    data.x = torch.tensor(X_feats, dtype=torch.float)
    print(f"✓ Engineered feature matrix loaded: shape = {data.x.shape}")
    print(f"✓ Features used: {feat_names}")
    return data
