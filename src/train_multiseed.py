# src/train_multiseed.py

import torch
import numpy as np
import os
import json
from data_utils_randomsplit import random_split
from train_utils import set_seed, train_full, save_config
from model_gat import GATNet
from model_gcn import GCNNet
from model_sage import GraphSAGENet
from config import MODELS_DIR
import pandas as pd

def safe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_numpy_safe(array, path):
    np.save(path, array)

def save_json_safe(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

def save_full_experiment(model, logs_dict, val_idx, config_dict, model_type, seed):
    base_path = os.path.join(MODELS_DIR, model_type, f"seed_{seed}")
    safe_create_dir(base_path)

    torch.save(model.state_dict(), os.path.join(base_path, "model.pth"))

    save_numpy_safe(logs_dict['loss'], os.path.join(base_path, "loss.npy"))
    save_numpy_safe(logs_dict['val_acc'], os.path.join(base_path, "val_acc.npy"))
    save_numpy_safe(logs_dict['val_f1_macro'], os.path.join(base_path, "val_f1_macro.npy"))
    save_numpy_safe(logs_dict['val_f1_illicit'], os.path.join(base_path, "val_f1_illicit.npy"))
    save_numpy_safe(val_idx.cpu().numpy(), os.path.join(base_path, "val_idx.npy"))

    save_json_safe(config_dict, os.path.join(base_path, "config.json"))

def train_and_evaluate_multiseed(seed, data, model_type="GAT", test_size=0.3):
    set_seed(seed)

    train_idx, val_idx = random_split(data, test_size=test_size, random_state=seed)

    in_channels = data.x.shape[1]
    out_channels = 2
    dropout = 0.3
    hidden_channels = 8 if model_type == "GAT" else 64
    heads = 8 if model_type == "GAT" else None

    if model_type == "GAT":
        model = GATNet(in_channels, hidden_channels, out_channels, heads, dropout)
    elif model_type == "GCN":
        model = GCNNet(in_channels, hidden_channels, out_channels, dropout)
    elif model_type == "SAGE":
        model = GraphSAGENet(in_channels, hidden_channels, out_channels, dropout)
    else:
        raise ValueError("Invalid model_type. Choose from GAT, GCN, SAGE.")
    
    model = model.to(data.x.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    model, loss_hist, val_acc_hist, val_f1_macro_hist, val_f1_illicit_hist = train_full(
        model, data, train_idx, val_idx, optimizer,
        num_epochs=300, patience=30
    )

    logs_dict = {
        'loss': np.array(loss_hist),
        'val_acc': np.array(val_acc_hist),
        'val_f1_macro': np.array(val_f1_macro_hist),
        'val_f1_illicit': np.array(val_f1_illicit_hist)
    }

    config_dict = {
        "model": model_type,
        "seed": seed,
        "hidden_channels": hidden_channels,
        "dropout": dropout,
        "lr": 0.001,
        "weight_decay": 5e-4
    }

    save_full_experiment(model, logs_dict, val_idx, config_dict, model_type.upper(), seed)


