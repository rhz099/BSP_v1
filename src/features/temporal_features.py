import numpy as np
from tqdm import tqdm
import networkx as nx

def compute_temporal_features(data, nx_graph, node_timestamps, max_time_step=49):
    """
    Computes selected temporal features per node:
    - temporal_lag: Time since node's last appearance.
    - log_component_size: log(1 + size of weakly connected component (WCC) containing the node)

    Returns:
        np.ndarray: shape [num_nodes, 2]
        List[str]: feature names
    """
    num_nodes = data.num_nodes
    temporal_lag = np.zeros(num_nodes)
    component_size = np.zeros(num_nodes)

    # Precompute component sizes (based on weakly connected components)
    comp_map = {}
    for comp in nx.weakly_connected_components(nx_graph):
        size = len(comp)
        for n in comp:
            comp_map[n] = size

    for node in tqdm(range(num_nodes), desc="Computing temporal features"):
        ts_list = node_timestamps[node]

        if ts_list:
            last_time = max(ts_list)
            temporal_lag[node] = max_time_step - last_time
        else:
            temporal_lag[node] = max_time_step

        raw_size = comp_map.get(node, 1)
        component_size[node] = np.log1p(raw_size)  # log-scaling here

    X_temp = np.vstack([temporal_lag, component_size]).T
    return X_temp, ["temporal_lag", "log_component_size"]
