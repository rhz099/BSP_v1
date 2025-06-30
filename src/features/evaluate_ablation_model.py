# src/features/evaluate_ablation_model.py

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import confusion_matrix

from model_gat import GATNet
from model_gcn import GCNNet
from model_sage import GraphSAGENet

import sys
sys.path.append("../src")


from .feature_utils import make_model_class_from_config
from analysis_utils import (
    load_metrics_across_seeds,
    aggregate_metrics_table,
    plot_loss_curve,
    compute_prediction_metrics
)
from evaluation_pipeline import (
    save_fp_fn_indices,
    attach_node_times,
    plot_fp_fn_time_distributions,
    export_full_evaluation_to_json,
    summarize_classification_report,
    run_inference_all_seeds,
    plot_conf_matrices
)

from load_elliptic_data import load_and_preprocess_elliptic_data


def evaluate_ablation_model(model_dir, model_class, model_name, seeds, data_dir, node_id_csv_path):
    """
    Full evaluation pipeline for a single ablation model using pre-saved data_exp.pt per seed.
    """

    # === Load sample data object from seed_42 to attach timestamps later ===
    seed_path = os.path.join(data_dir, "seed_42", "data_exp.pt")
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Missing data_exp.pt at: {seed_path}")
    data = torch.load(seed_path, map_location="cpu", weights_only=False)

    # === Load training logs ===
    metrics_logs = load_metrics_across_seeds(model_dir, ["val_acc", "val_f1_macro", "val_f1_illicit", "loss"])
    summary_df = aggregate_metrics_table(metrics_logs)
    print(f"\nSummary Metrics (from training logs):\n{summary_df}\n")
    plot_loss_curve(metrics_logs["loss"], label=f"{model_name} Loss")

    # === Inference (per seed) ===
    y_true_all, y_pred_all, y_proba_all, seed_metrics = run_inference_all_seeds(
        model_dir=model_dir,
        model_class=model_class,
        data_dir=data_dir,
        seeds=seeds
    )

    # === Confusion Matrices ===
    plot_conf_matrices(y_true_all, y_pred_all, seeds, model_name)

    # === Seed-wise Metric Summary ===
    val_acc_list = metrics_logs["val_acc"]
    for i in range(len(seed_metrics)):
        seed_metrics[i]["val_acc"] = val_acc_list[i]
    df = pd.DataFrame(seed_metrics)

    core_metrics = ['val_acc', 'bal_acc', 'f1_macro', 'f1_illicit', 'precision_illicit', 'recall_illicit', 'mcc']
    means = df[core_metrics].mean()
    stds = df[core_metrics].std()

    # Bar Plot
    plt.figure(figsize=(8, 5))
    plt.bar(core_metrics, means, yerr=stds, capsize=5)
    plt.ylim(0.65, 1.0)
    plt.title(f"{model_name} Metrics Across Seeds (Mean ± Std)")
    plt.xticks(rotation=30)
    plt.ylabel("Score")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(10, 6))
    df[core_metrics].boxplot()
    plt.ylim(0.65, 1.0)
    plt.title(f"{model_name}: Seed-wise Distribution of Metrics")
    plt.xticks(rotation=30)
    plt.ylabel("Score")
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Tabular Summary
    summary_table = pd.DataFrame({
        "Metric": core_metrics,
        "Mean": means.round(4).values,
        "Std": stds.round(4).values,
        "Formatted": [f"{m:.4f} ± {s:.4f}" for m, s in zip(means, stds)]
    })
    summary_table = summary_table.set_index("Metric").loc[core_metrics].reset_index()
    display(summary_table)

    # === Classification Report Summary ===
    summarize_classification_report(y_true_all, y_pred_all)

    # === Save FP/FN indices + time info ===
    save_fp_fn_indices(y_true_all, y_pred_all, seeds, model_dir)

    # === Add timestamps + plot distributions ===
    data = attach_node_times(data, node_id_csv_path)
    plot_fp_fn_time_distributions(data, seeds, model_dir, model_name)

    # === Export to JSON for aggregate analysis ===
    conf_matrices = [confusion_matrix(y_true_all[i], y_pred_all[i]) for i in range(len(seeds))]
    timestamps = data.node_times.cpu().numpy()
    fp_counts, fn_counts, fp_time_all, fn_time_all = [], [], [], []
    for seed in seeds:
        base = os.path.join(model_dir, f"seed_{seed}")
        fp_idx = np.load(os.path.join(base, "fp_indices.npy"))
        fn_idx = np.load(os.path.join(base, "fn_indices.npy"))
        fp_counts.append(len(fp_idx))
        fn_counts.append(len(fn_idx))
        fp_time_all.extend(timestamps[fp_idx])
        fn_time_all.extend(timestamps[fn_idx])
    fp_time_all = [int(t) for t in fp_time_all]
    fn_time_all = [int(t) for t in fn_time_all]

    export_full_evaluation_to_json(
        seed_metrics=seed_metrics,
        conf_matrices=conf_matrices,
        fp_counts=fp_counts,
        fn_counts=fn_counts,
        fp_time_all=fp_time_all,
        fn_time_all=fn_time_all,
        output_dir=model_dir
    )
    