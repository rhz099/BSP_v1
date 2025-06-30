# src/features/structural_features.py

import numpy as np
import networkx as nx

def compute_structural_features(graph, valid_node_ids):
    """
    Computes structural features:
    - in_degree
    - out_degree
    - balance (in - out)
    - clustering coefficient (undirected)

    Args:
        graph
        valid_node_ids (List[int])

    Returns:
        np.ndarray: [num_nodes, 4]
        List[str]: feature names
    """
    in_deg_dict = dict(graph.in_degree()) if graph.is_directed() else dict(graph.degree())
    out_deg_dict = dict(graph.out_degree()) if graph.is_directed() else dict(graph.degree())
    clustering_dict = nx.clustering(graph.to_undirected(), nodes=valid_node_ids)

    features = []
    for nid in valid_node_ids:
        in_d = in_deg_dict.get(nid, 0)
        out_d = out_deg_dict.get(nid, 0)
        balance = in_d - out_d
        clustering = clustering_dict.get(nid, 0.0)
        features.append((in_d, out_d, balance, clustering))

    X_struct = np.array(features, dtype=np.float32)
    feature_names = ["in_degree", "out_degree", "balance", "clustering_coefficient"]
    return X_struct, feature_names

def structural_feature_names():
    return ["in_degree", "out_degree", "balance", "clustering_coefficient"]
