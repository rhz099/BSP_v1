import os
import json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from analysis_utils import (
    evaluate_model_from_checkpoint,
    compute_prediction_metrics,
    load_metrics_across_seeds,
    aggregate_metrics_table,
    plot_loss_curve
)


def run_inference_all_seeds(model_dir, model_class, data, seeds):
    y_true_all, y_pred_all, y_proba_all, seed_metrics = [], [], [], []

    for seed in seeds:
        base = os.path.join(model_dir, f"seed_{seed}")
        val_idx = np.load(os.path.join(base, "val_idx.npy"))
        config = json.load(open(os.path.join(base, "config.json")))
        model_path = os.path.join(base, "model.pth")

        y_true, y_pred, y_proba = evaluate_model_from_checkpoint(
            model_class=model_class,
            model_path=model_path,
            data_object=data,
            val_idx=val_idx,
            config=config
        )

        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        y_proba_all.append(y_proba)
        seed_metrics.append(compute_prediction_metrics(y_true, y_pred, y_proba))

    return y_true_all, y_pred_all, y_proba_all, seed_metrics


def plot_conf_matrices(y_true_all, y_pred_all, seeds, model_name):
    conf_matrices = [
        confusion_matrix(y_true_all[i], y_pred_all[i])
        for i in range(len(seeds))
    ]

    fig, axes = plt.subplots(1, len(seeds), figsize=(20, 4))
    for ax, cm, seed in zip(axes, conf_matrices, seeds):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred 0", "Pred 1"],
                    yticklabels=["True 0", "True 1"])
        ax.set_title(f"Seed {seed}")
    plt.suptitle(f"Raw Confusion Matrices per Seed ({model_name})", y=1.05)
    plt.tight_layout()
    plt.show()

    mean_cm = np.mean(conf_matrices, axis=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(mean_cm, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title(f"Mean Confusion Matrix (Unnormalized, {model_name})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    mean_cm_norm = mean_cm / mean_cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(mean_cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title(f"Normalized Mean Confusion Matrix (% per row, {model_name})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def summarize_classification_report(y_true_all, y_pred_all):
    report_all = [
        classification_report(y_true_all[i], y_pred_all[i], output_dict=True, zero_division=0)
        for i in range(len(y_true_all))
    ]
    classes = ['0', '1']
    metrics = ['precision', 'recall', 'f1-score']

    summary = {}
    for cls in classes:
        summary[cls] = {}
        for metric in metrics:
            vals = [rep[cls][metric] for rep in report_all]
            summary[cls][metric] = (np.mean(vals), np.std(vals))

    for cls in classes:
        label = "Legit (0)" if cls == "0" else "Illicit (1)"
        print(f"\n{label} Metrics:")
        for metric in metrics:
            mean, std = summary[cls][metric]
            print(f"{metric}: {mean:.4f} ± {std:.4f}")


def save_fp_fn_indices(y_true_all, y_pred_all, seeds, model_dir):
    for i, seed in enumerate(seeds):
        y_true = np.array(y_true_all[i])
        y_pred = np.array(y_pred_all[i])

        fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
        fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]

        np.save(os.path.join(model_dir, f"seed_{seed}/fp_indices.npy"), fp_idx)
        np.save(os.path.join(model_dir, f"seed_{seed}/fn_indices.npy"), fn_idx)


def attach_node_times(data, node_id_csv_path):
    df_feat = pd.read_csv(node_id_csv_path, header=None)
    node_ids = df_feat.iloc[:, 0].values
    timestamps = df_feat.iloc[:, 1].values
    node_id_to_time = dict(zip(node_ids, timestamps))
    sorted_node_ids = np.sort(node_ids)
    timestamps_ordered = np.array([node_id_to_time[n] for n in sorted_node_ids])
    data.node_times = torch.tensor(timestamps_ordered)
    return data


def plot_fp_fn_time_distributions(data, seeds, model_dir, model_name):
    timestamps = data.node_times.cpu().numpy()
    fp_time_all, fn_time_all = [], []

    for seed in seeds:
        seed_dir = os.path.join(model_dir, f"seed_{seed}")
        fp_indices = np.load(os.path.join(seed_dir, "fp_indices.npy"))
        fn_indices = np.load(os.path.join(seed_dir, "fn_indices.npy"))
        fp_time_all.extend(timestamps[fp_indices])
        fn_time_all.extend(timestamps[fn_indices])

    # --- Raw count histogram ---
    fp_df = pd.DataFrame({"time": fp_time_all, "error_type": "False Positive"})
    fn_df = pd.DataFrame({"time": fn_time_all, "error_type": "False Negative"})
    all_errors = pd.concat([fp_df, fn_df])

    plt.figure(figsize=(10, 5))
    sns.histplot(data=all_errors, x="time", hue="error_type", multiple="stack", bins=49)
    plt.title(f"Raw Time Distribution of FP/FN Errors ({model_name})")
    plt.xlabel("Time Step")
    plt.ylabel("Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # --- Normalized error rate per time step ---
    time_steps = np.arange(1, 50)
    fp_times = np.array(fp_time_all)
    fn_times = np.array(fn_time_all)
    fp_count = np.array([np.sum(fp_times == t) for t in time_steps])
    fn_count = np.array([np.sum(fn_times == t) for t in time_steps])
    node_count = np.array([np.sum(timestamps == t) for t in time_steps])

    fp_rate = fp_count / node_count
    fn_rate = fn_count / node_count

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, fp_rate, label="False Positive Rate", marker="o", color="blue")
    plt.plot(time_steps, fn_rate, label="False Negative Rate", marker="o", color="orange")
    plt.title(f"Normalized FP/FN Error Rates by Time Step ({model_name})")
    plt.xlabel("Time Step")
    plt.ylabel("Error Rate per Node")
    plt.ylim(0, max(fp_rate.max(), fn_rate.max()) * 1.1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

import json
import os

def export_full_evaluation_to_json(
    seed_metrics,
    conf_matrices,
    fp_counts,
    fn_counts,
    fp_time_all,
    fn_time_all,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "seed_metrics.json"), "w") as f:
        json.dump(seed_metrics, f, indent=4)

    with open(os.path.join(output_dir, "conf_matrices.json"), "w") as f:
        json.dump([cm.tolist() for cm in conf_matrices], f, indent=2)

    with open(os.path.join(output_dir, "fp_counts.json"), "w") as f:
        json.dump(fp_counts, f)

    with open(os.path.join(output_dir, "fn_counts.json"), "w") as f:
        json.dump(fn_counts, f)

    with open(os.path.join(output_dir, "fp_time.json"), "w") as f:
        json.dump(fp_time_all, f)

    with open(os.path.join(output_dir, "fn_time.json"), "w") as f:
        json.dump(fn_time_all, f)
