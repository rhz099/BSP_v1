# src/features/feature_utils.py

import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import json

from model_gat import GATNet
from model_gcn import GCNNet
from model_sage import GraphSAGENet

def prepare_graph_and_timestamps(
    edgelist_path: str,
    features_path: str,
    num_nodes: int
):
    """
    Loads directed NetworkX graph and builds timestamp history for each node aligned with PyG indexing?

    Args:
        edgelist_path (str): Path to elliptic_txs_edgelist.csv (with header!!!)
        features_path (str): Path to elliptic_txs_features.csv (no header)
        num_nodes (int): Number of nodes in PyG data (for alignment)

    Returns:
        G_nx (networkx.DiGraph): directed transaction graph
        node_timestamps (List[List[int]]): list of time steps per node aligned to PyG node indices
    """

    # --- Load edgelist with header, build directed graph ---
    edgelist_df = pd.read_csv(edgelist_path)
    edgelist_df.columns = ["txId1", "txId2"]  # ensure consistent naming

    # Create sorted list of unique txIds (all nodes)
    tx_ids = sorted(set(edgelist_df["txId1"]).union(set(edgelist_df["txId2"])))
    node_mapping = {tx: idx for idx, tx in enumerate(tx_ids)}

    # Build directed graph with relabeled nodes
    G_nx = nx.from_pandas_edgelist(
        edgelist_df,
        source="txId1",
        target="txId2",
        create_using=nx.DiGraph()
    )
    
    G_nx = nx.relabel_nodes(G_nx, node_mapping)

    print(f"Loaded directed graph with {G_nx.number_of_nodes()} nodes and {G_nx.number_of_edges()} edges.")
    print(f"Graph type: {type(G_nx)}")

    # --- Load features (no header) ---
    features_df = pd.read_csv(features_path, header=None)

    num_cols = features_df.shape[1]
    num_feats = num_cols - 2  # exclude txId and time_step

    # Corrected ordering: txId, time_step, f0, ..., f164
    features_df.columns = ["txId", "time_step"] + [f"f{i}" for i in range(num_feats)]
    features_df["time_step"] = features_df["time_step"].astype(int)

    print(f"Features DataFrame shape: {features_df.shape}")
    print(f"Number of features (excluding txId and time_step): {num_feats}")

    # --- Build timestamp mapping ---
    node_ts_map = defaultdict(list)
    for _, row in features_df.iterrows():
        node_id = node_mapping.get(row["txId"])
        if node_id is not None:
            node_ts_map[node_id].append(row["time_step"])

    node_timestamps = [node_ts_map.get(n, []) for n in range(num_nodes)]
    ts_counts = sum(len(ts) > 0 for ts in node_timestamps)
    print(f"Timestamps found for {ts_counts} out of {num_nodes} nodes.")

    all_ts = [ts for ts_list in node_timestamps for ts in ts_list]
    if all_ts:
        print(f"✓ Time step range: {min(all_ts)} to {max(all_ts)}")
    else:
        print("Warning: no timestamps found in node_timestamps!")

    return G_nx, node_timestamps

import torch
import torch.nn.functional as F

def make_model_class_from_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config["model"]
    feature_dim = config["feature_dim"]
    hidden_channels = config["hidden_channels"]
    heads = config.get("heads", 8)
    out_channels = config.get("out_channels", 2)
    dropout = config.get("dropout", 0.1)

    if model_name == "GAT":
        return lambda: GATNet(
            in_channels=feature_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout
        )
    elif model_name == "GCN":
        return lambda: GCNNet(
            in_channels=feature_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout
        )
    elif model_name == "SAGE":
        return lambda: GraphSAGENet(
            in_channels=feature_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def evaluate_model_from_checkpoint(model_class, model_path, data_object, val_idx, config):
    """
    Instantiates model using config.json and runs prediction on val_idx.
    """
    model = model_class()  # ← this assumes make_model_class_from_config returns a no-arg lambda
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    x, edge_index = data_object.x, data_object.edge_index
    out = model(x, edge_index)

    proba = F.softmax(out[val_idx], dim=1).detach().cpu().numpy()
    pred = proba.argmax(axis=1)
    true = data_object.y[val_idx].cpu().numpy()

    return true, pred, proba
