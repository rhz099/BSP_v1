# src/load_elliptic_data.py

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, degree
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_elliptic_data(data_dir: str = "../elliptic_bitcoin_dataset") -> Data:
    """
    Loads and processes the Elliptic dataset into a PyTorch Geometric Data object.

    Args:
        data_dir (str): Path to the directory containing the CSV files.

    Returns:
        Data: A PyG Data object with normalized features and no isolated nodes.
    """
    # Load raw data
    classes_df = pd.read_csv(f"{data_dir}/elliptic_txs_classes.csv")
    edges_df = pd.read_csv(f"{data_dir}/elliptic_txs_edgelist.csv")
    features_df = pd.read_csv(f"{data_dir}/elliptic_txs_features.csv", header=None)

    # Map all transaction IDs to continuous indices
    all_ids = pd.concat([
        classes_df.iloc[:, 0].astype(str),
        edges_df.iloc[:, 0].astype(str),
        edges_df.iloc[:, 1].astype(str),
        features_df.iloc[:, 0].astype(str)
    ])
    unique_tx_ids = np.sort(all_ids.unique())
    txid2idx = {tx_id: idx for idx, tx_id in enumerate(unique_tx_ids)}

    classes_df['node_idx'] = classes_df.iloc[:, 0].astype(str).map(txid2idx)
    edges_df['src_idx'] = edges_df.iloc[:, 0].astype(str).map(txid2idx)
    edges_df['dst_idx'] = edges_df.iloc[:, 1].astype(str).map(txid2idx)
    features_df['node_idx'] = features_df.iloc[:, 0].astype(str).map(txid2idx)

    # Build node feature tensor
    num_nodes = len(unique_tx_ids)
    num_features = features_df.shape[1] - 2  # exclude tx_id and node_idx
    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    for row in features_df.itertuples(index=False):
        node_idx = getattr(row, 'node_idx')
        feats = list(row)[1:-1]
        x[node_idx] = torch.tensor(feats, dtype=torch.float)

    # Build edge index tensor
    edge_index = torch.stack([
        torch.tensor(edges_df['src_idx'].values, dtype=torch.long),
        torch.tensor(edges_df['dst_idx'].values, dtype=torch.long)
    ], dim=0)

    # Build label tensor
    y = -1 * torch.ones(num_nodes, dtype=torch.long)
    label_map = classes_df.set_index('node_idx')['class'].to_dict()
    for node_idx, label in label_map.items():
        if str(label).strip() == '1':
            y[node_idx] = 0
        elif str(label).strip() == '2':
            y[node_idx] = 1

    data = Data(x=x, edge_index=edge_index, y=y)

    # Normalize features
    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x), dtype=torch.float)

    # Remove isolated nodes
    return filter_isolated_nodes(data)



def filter_isolated_nodes(data: Data) -> Data:
    """
    Filters out isolated nodes (degree 0) and remaps edge indices.

    Args:
        data (Data): PyG Data object possibly containing isolated nodes.

    Returns:
        Data: New Data object with isolated nodes removed and edge_index remapped.
    """
    edge_index = to_undirected(data.edge_index)
    deg = degree(edge_index[0], num_nodes=data.num_nodes)
    mask = deg != 0
    non_iso_idx = mask.nonzero(as_tuple=True)[0]
    remap = {int(old): i for i, old in enumerate(non_iso_idx)}

    # Filter features and labels
    x = data.x[non_iso_idx]
    y = data.y[non_iso_idx]

    # Filter and remap edges
    src_all, dst_all = data.edge_index
    edge_mask = mask[src_all] & mask[dst_all]
    src_f = src_all[edge_mask]
    dst_f = dst_all[edge_mask]

    src_remap = [remap[int(s)] for s in src_f]
    dst_remap = [remap[int(d)] for d in dst_f]
    edge_index = torch.tensor([src_remap, dst_remap], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

