# src/train_utils.py

import os
import random
import numpy as np
import torch
import json
from sklearn.metrics import f1_score

# ======================================================
# 1. Reproducibility — full seed control
# ======================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======================================================
# 2. Train loop — unified across GAT, GCN, GraphSAGE
# ======================================================

def train(model, data, train_idx, optimizer, clip_grad=1.0, loss_fn=None):
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    criterion = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()
    loss = criterion(out[train_idx], data.y[train_idx])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()

    return loss.item()


# ======================================================
# 3. Validation loop (accuracy & F1 per epoch)
# ======================================================

@torch.no_grad()
def evaluate(model, data, val_idx):
    model.eval()
    out = model(data.x, data.edge_index)
    preds = out[val_idx].argmax(dim=1).cpu()
    y_true = data.y[val_idx].cpu()

    acc = (preds == y_true).sum().item() / val_idx.size(0)
    f1_macro = f1_score(y_true, preds, average='macro')
    f1_illicit = f1_score(y_true, preds, pos_label=1)

    return acc, f1_macro, f1_illicit

# ======================================================
# 4. Train full run with early stopping & logging
# ======================================================

def train_full(model, data, train_idx, val_idx, optimizer, 
               num_epochs=200, patience=20, clip_grad=1.0, verbose=True, loss_fn=None):
    
    loss_history = []
    val_acc_history = []
    val_f1_macro_history = []
    val_f1_illicit_history = []

    best_f1 = 0
    best_acc = 0
    best_epoch = 0
    counter = 0
    best_weights = None

    # Default to standard cross-entropy if none provided
    criterion = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        loss = train(model, data, train_idx, optimizer, clip_grad, loss_fn=criterion)
        val_acc, f1_macro, f1_illicit = evaluate(model, data, val_idx)

        loss_history.append(loss)
        val_acc_history.append(val_acc)
        val_f1_macro_history.append(f1_macro)
        val_f1_illicit_history.append(f1_illicit)

        if verbose:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Val F1: {f1_macro:.4f} | F1 Illicit: {f1_illicit:.4f}")

        # Multi-metric early stopping
        if f1_macro > best_f1 or val_acc > best_acc:
            if f1_macro > best_f1:
                best_f1 = f1_macro
            if val_acc > best_acc:
                best_acc = val_acc
            best_epoch = epoch
            best_weights = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, loss_history, val_acc_history, val_f1_macro_history, val_f1_illicit_history



# ======================================================
# 5. Save hyperparams to disk for reproducibility
# ======================================================

def save_config(path, config_dict):
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)
        


def save_feature_experiment(output_dir, model, results, seed, config, val_idx):
    """
    Saves model weights, metric histories, validation indices, and config.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

    # Unpack metrics
    loss_history, val_acc_history, val_f1_macro_history, val_f1_illicit_history = results
    np.save(os.path.join(output_dir, "loss.npy"), np.array(loss_history))
    np.save(os.path.join(output_dir, "val_acc.npy"), np.array(val_acc_history))
    np.save(os.path.join(output_dir, "val_f1_macro.npy"), np.array(val_f1_macro_history))
    np.save(os.path.join(output_dir, "val_f1_illicit.npy"), np.array(val_f1_illicit_history))
    np.save(os.path.join(output_dir, "val_idx.npy"), val_idx.cpu().numpy())

    # Save experiment config
    config_with_seed = {**config, "seed": seed}
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_with_seed, f, indent=4)
