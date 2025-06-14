# ===============================
# src/visualise.py
# ===============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# -------------------------------------------------
# Training Curve Plots
# -------------------------------------------------

def plot_training_curves(logs_dict):
    epochs = np.arange(1, len(logs_dict['loss']) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.plot(epochs, logs_dict['loss'])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 4, 2)
    plt.plot(epochs, logs_dict['val_acc'])
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 4, 3)
    plt.plot(epochs, logs_dict['val_f1_macro'])
    plt.title("F1 Macro")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Macro")

    plt.subplot(1, 4, 4)
    plt.plot(epochs, logs_dict['val_f1_illicit'])
    plt.title("F1 Illicit")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Illicit")

    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=["Legit (0)", "Illicit (1)"],
                yticklabels=["Legit (0)", "Illicit (1)"])
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

# -------------------------------------------------
# ROC and PR Curve
# -------------------------------------------------

def plot_roc_pr_curves(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()
