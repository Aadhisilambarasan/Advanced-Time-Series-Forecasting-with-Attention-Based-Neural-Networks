import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from statsmodels.graphics.tsaplots import plot_acf


# ------------------------------------------------------------------
# Ensure results folder exists
# ------------------------------------------------------------------
os.makedirs("results", exist_ok=True)


# ------------------------------------------------------------------
# Prediction vs Actual
# ------------------------------------------------------------------
def plot_prediction_vs_actual(y_true, y_pred, n_points=300):

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    plt.figure(figsize=(14, 6))
    plt.plot(y_true[:n_points], linewidth=2)
    plt.plot(y_pred[:n_points], linewidth=2)

    plt.title("Energy Forecast vs Actual (Test Set)")
    plt.xlabel("Time Step")
    plt.ylabel("Energy Consumption")
    plt.legend(["Actual", "Predicted"])
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/prediction_vs_actual.png")
    plt.close()


# ------------------------------------------------------------------
# Residual Distribution
# ------------------------------------------------------------------
def plot_error_distribution(y_true, y_pred):

    errors = (np.asarray(y_true) - np.asarray(y_pred)).flatten()

    plt.figure(figsize=(10, 5))
    sns.histplot(errors, bins=50, kde=True)

    plt.axvline(0, linestyle="--")
    plt.title("Residual Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("results/error_distribution.png")
    plt.close()


# ------------------------------------------------------------------
# Residual Autocorrelation
# ------------------------------------------------------------------
def plot_residual_acf(y_true, y_pred):

    errors = (np.asarray(y_true) - np.asarray(y_pred)).flatten()

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(errors, lags=50, ax=ax)

    plt.title("Residual Autocorrelation")
    plt.tight_layout()
    plt.savefig("results/residual_acf.png")
    plt.close()


# ------------------------------------------------------------------
# Rolling MAE
# ------------------------------------------------------------------
def plot_rolling_mae(y_true, y_pred, window=50):

    errors = np.abs((np.asarray(y_true) - np.asarray(y_pred)).flatten())

    rolling_mae = np.convolve(
        errors,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(12, 4))
    plt.plot(rolling_mae)

    plt.title(f"Rolling MAE (Window={window})")
    plt.xlabel("Time")
    plt.ylabel("MAE")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/rolling_mae.png")
    plt.close()


# ------------------------------------------------------------------
# Model Comparison (MASE)
# ------------------------------------------------------------------
def plot_model_comparison(metrics_dict):

    models = list(metrics_dict.keys())
    mase_scores = [metrics_dict[m]["MASE"] for m in models]

    plt.figure(figsize=(8, 5))
    plt.bar(models, mase_scores)

    plt.axhline(1.0, linestyle="--")
    plt.title("Model Comparison (MASE)")
    plt.ylabel("MASE Score")

    plt.tight_layout()
    plt.savefig("results/model_comparison.png")
    plt.close()


# ------------------------------------------------------------------
# Attention Importance (Robust + Shape Safe)
# ------------------------------------------------------------------
def plot_attention_importance(model, sample_input):

    model.eval()

    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    with torch.no_grad():
        _, attentions = model(
            sample_input.unsqueeze(0),
            return_attention=True
        )

    # attentions = list of layers
    # each layer shape: (batch, heads, seq_len, seq_len)

    if len(attentions) == 0:
        print("No attention weights returned.")
        return

    # Take last layer
    attn = attentions[-1]

    # Ensure correct shape
    if attn.dim() != 4:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    # Average across heads
    attn = attn.mean(dim=1)  # (batch, seq_len, seq_len)

    # Take last time step query
    attn = attn[0, -1, :]  # (seq_len,)

    attn = attn.detach().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.plot(attn)

    plt.title("Attention Importance Across Historical Window")
    plt.xlabel("Historical Time Steps")
    plt.ylabel("Average Attention Weight")

    plt.tight_layout()
    plt.savefig("results/attention_importance.png")
    plt.close()


# ------------------------------------------------------------------
# Training Curves Comparison
# ------------------------------------------------------------------
def plot_all_training_curves(results):

    plt.figure(figsize=(12, 6))

    for model_name, metrics in results.items():

        val_loss = metrics.get("val_loss")

        if val_loss is not None and len(val_loss) > 0:
            plt.plot(val_loss, label=f"{model_name} Val")

    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/model_training_comparison.png")
    plt.close()
