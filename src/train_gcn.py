# src/train_gcn.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, val_idx):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[val_idx].argmax(dim=1)
        acc = (pred == data.y[val_idx]).sum().item() / val_idx.size(0)
    return acc