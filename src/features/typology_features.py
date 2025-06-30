# src/features/typology_features.py

import numpy as np
import networkx as nx

def compute_typology_features(graph, valid_node_ids):
    """
    Compute typology features:
    - log-transformed fan_in_out_ratio
    - input diversity (in_deg / total_deg)

    Returns:
        np.ndarray: shape [num_nodes, 2]
        List[str]: feature names
    """
    fan_in_out_ratios = []
    input_diversities = []

    for node in valid_node_ids:
        in_deg = graph.in_degree(node) if graph.is_directed() else graph.degree(node)
        out_deg = graph.out_degree(node) if graph.is_directed() else graph.degree(node)

        # Feature 1: log-transformed fan_in_out_ratio
        raw_ratio = out_deg / (in_deg + 1e-6)
        fan_in_out_ratios.append(np.log1p(raw_ratio))

        # Feature 2: input diversity
        diversity = in_deg / (in_deg + out_deg + 1e-6)
        input_diversities.append(diversity)

    X_typ = np.vstack([fan_in_out_ratios, input_diversities]).T
    feature_names = [
        "log_fan_in_out_ratio",
        "input_diversity"
    ]
    return X_typ, feature_names

def typology_feature_names():
    return ["log_fan_in_out_ratio", "input_diversity"]