"""
analysis_utils.py

Reusable utility functions for analyzing GNN model performance

Sections:
1. Metric Loading
2. Metric Aggregation and Tabulation
3. Loss Curve Visualization
4. Inference and Prediction Recovery
5. Metric Computation from Predictions
6. Statistical Testing
7. Error Analysis

"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef
)


from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt


# ===============================
# 1. Metric Loading
# ===============================

def load_metrics_across_seeds(model_dir, metric_names):
    all_metrics = {name: [] for name in metric_names}
    for seed_folder in sorted(os.listdir(model_dir)):
        seed_path = os.path.join(model_dir, seed_folder)
        for metric in metric_names:
            file_path = os.path.join(seed_path, f"{metric}.npy")
            if os.path.exists(file_path):
                data = np.load(file_path)
                if metric == 'loss':
                    all_metrics[metric].append(data)
                else:
                    # Append last epoch value (or best if needed)
                    all_metrics[metric].append(data[-1])
    return all_metrics



# ===============================
# 2. Metric Aggregation and Tabulation
# ===============================

def aggregate_metrics_table(metrics_dict):
    records = []
    for metric, values in metrics_dict.items():
        if metric == "loss":
            continue
        mean = np.mean(values)
        std = np.std(values)
        records.append({
            "Metric": metric,
            "Mean": round(mean, 4),
            "Std": round(std, 4),
            "Formatted": f"{mean:.4f} ± {std:.4f}"
        })
    return pd.DataFrame(records)


# ===============================
# 3. Loss Curve Visualization (Refactored as some models converged early, not all 300 epochs)
# ===============================
def plot_loss_curve(loss_arrays, label="Loss"):
    import numpy as np
    import matplotlib.pyplot as plt

    # Handle unequal lengths by padding with NaN
    max_len = max(len(arr) for arr in loss_arrays)
    padded = [np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) for arr in loss_arrays]
    all_losses = np.vstack(padded)

    mean_loss = np.nanmean(all_losses, axis=0)
    std_loss = np.nanstd(all_losses, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(mean_loss, label=f"{label} (mean)", color="blue")
    plt.fill_between(np.arange(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, alpha=0.3)
    plt.title("Loss Curve Across Seeds")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.legend()
    plt.show()



# ===============================
# 4. Inference and Prediction Recovery
# ===============================

def evaluate_model_from_checkpoint(model_class, model_path, data_object, val_idx, config):
    in_channels = data_object.x.shape[1]
    out_channels = 2
    dropout = config.get("dropout", 0.3)
    hidden_channels = config.get("hidden_channels", 64)
    heads = config.get("heads", 8)  # Only used for GAT

    if "GAT" in config.get("model", ""):
        model = model_class(in_channels, hidden_channels, out_channels, heads, dropout)
    else:
        model = model_class(in_channels, hidden_channels, out_channels, dropout=dropout)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        out = model(data_object.x, data_object.edge_index)
        logits = out[val_idx]
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_proba = F.softmax(logits, dim=1).cpu().numpy()
        y_true = data_object.y[val_idx].cpu().numpy()

    return y_true, y_pred, y_proba



# ===============================
# 5. Metric Computation from Predictions
# ===============================


def compute_prediction_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_illicit': f1_score(y_true, y_pred, pos_label=1),
        'precision_illicit': precision_score(y_true, y_pred, pos_label=1),
        'recall_illicit': recall_score(y_true, y_pred, pos_label=1),
        'bal_acc': balanced_accuracy_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            metrics['roc_auc_macro'] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            metrics['pr_auc_illicit'] = average_precision_score((np.array(y_true) == 1).astype(int), y_proba[:, 1])
        except:
            metrics['roc_auc_macro'] = None
            metrics['pr_auc_illicit'] = None

    # Confusion matrix derived metrics
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    return metrics



# ===============================
# 6. Statistical Testing
# ===============================

def perform_statistical_tests(metric_list_a, metric_list_b, metric_name="F1"):
    t_stat, t_pval = ttest_rel(metric_list_a, metric_list_b)
    w_stat, w_pval = wilcoxon(metric_list_a, metric_list_b)
    return {
        "metric": metric_name,
        "t_test_pval": t_pval,
        "wilcoxon_pval": w_pval,
        "mean_a": np.mean(metric_list_a),
        "mean_b": np.mean(metric_list_b)
    }


# ===============================
# 7. Error Analysis
# ===============================

def analyze_fp_fn_errors(y_true, y_pred, node_metadata=None):
    fp_idx = np.where((y_pred == 1) & (y_true != 1))[0]
    fn_idx = np.where((y_pred != 1) & (y_true == 1))[0]
    results = {
        "false_positives": fp_idx.tolist(),
        "false_negatives": fn_idx.tolist()
    }
    if node_metadata is not None:
        results["fp_details"] = node_metadata.iloc[fp_idx]
        results["fn_details"] = node_metadata.iloc[fn_idx]
    return results

# ===============================
# 8. turn results into csv
# ===============================
import csv

def log_metrics_to_csv(
    model_name: str,
    split_name: str,
    ablation: str,
    seeds: list,
    val_acc_list: list,
    seed_metrics: list,
    is_feature: bool = False,
    results_dir: str = "../results"
):
    """
    Appends evaluation metrics to a CSV under results/.

    Args:
        model_name (str): e.g., "GAT-Random"
        split_name (str): e.g., "randomsplit"
        ablation (str): e.g., "base+temporal"
        seeds (List[int])
        val_acc_list (List[float])
        seed_metrics (List[Dict])
        is_feature (bool): whether to append '_feature' to the filename
        results_dir (str): base directory for saving results
    """
    os.makedirs(results_dir, exist_ok=True)
    suffix = "_metrics_feature.csv" if is_feature else "_metrics.csv"
    csv_path = os.path.join(results_dir, f"{model_name}{suffix}")

    header = [
        "model", "split", "ablation", "seed",
        "val_acc", "bal_acc", "f1_macro", "f1_illicit",
        "precision_illicit", "recall_illicit", "mcc"
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()

        for i, seed in enumerate(seeds):
            m = seed_metrics[i]
            writer.writerow({
                "model": model_name,
                "split": split_name,
                "ablation": ablation,
                "seed": seed,
                "val_acc": round(val_acc_list[i], 4),
                "bal_acc": round(m["bal_acc"], 4),
                "f1_macro": round(m["f1_macro"], 4),
                "f1_illicit": round(m["f1_illicit"], 4),
                "precision_illicit": round(m["precision_illicit"], 4),
                "recall_illicit": round(m["recall_illicit"], 4),
                "mcc": round(m["mcc"], 4)
            })
