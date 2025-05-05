# src/train_gat.py
import torch
import torch.nn.functional as F

def train(model, data, train_idx, optimizer, clip_grad=1.0):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[train_idx], data.y[train_idx])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    return loss.item()

def evaluate(model, data, val_idx):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out[val_idx].max(dim=1)[1]
        acc = (pred == data.y[val_idx]).sum().item() / val_idx.numel()
    return acc
