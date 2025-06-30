# src/features/feature_pipeline.py

import numpy as np

from .base_features import compute_base_features, base_feature_names
from .structural_features import compute_structural_features, structural_feature_names
from .typology_features import compute_typology_features, typology_feature_names
from .temporal_features import compute_temporal_features

def generate_all_features(
    data,
    nx_graph=None,
    node_timestamps=None,
    include_base=True,
    include_structural=False,
    include_typology=False,
    include_temporal=False,
    structural_graph=None,
    typology_graph=None,
    temporal_graph=None
):
    """
    Generate feature matrix with selected ablation configurations.
    Supports overriding graph input per feature type.

    Args:
        data (torch_geometric.data.Data)
        nx_graph (networkx.Graph) – used if individual graphs not provided
        node_timestamps (List[List[int]])
        include_* (bool) – ablation flags
        *_graph (networkx.Graph) – optional per-feature-type graph override

    Returns:
        X (np.ndarray): [num_nodes, num_features]
        feature_names (List[str])
    """
    X_list = []
    names = []
    num_nodes = data.num_nodes
    valid_node_ids = list(range(num_nodes))

    # Base features (from raw matrix)
    if include_base:
        base = compute_base_features(data)
        X_list.append(base)
        names += base_feature_names(num_features=base.shape[1])

    # Structural
    if include_structural:
        graph = structural_graph or nx_graph
        structural, struct_names = compute_structural_features(graph, valid_node_ids)
        X_list.append(structural)
        names += struct_names

    # Typology
    if include_typology:
        graph = typology_graph or nx_graph
        typology_feats, typology_names_ = compute_typology_features(graph, valid_node_ids)
        X_list.append(typology_feats)
        names += typology_names_

    # Temporal
    if include_temporal:
        graph = temporal_graph or nx_graph
        temporal_feats, temporal_names_ = compute_temporal_features(data, graph, node_timestamps)
        X_list.append(temporal_feats)
        names += temporal_names_

    X = np.hstack(X_list)
    return X, names
