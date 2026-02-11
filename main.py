import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
import pandas as pd

from models.transformer import TemporalTransformer
from models.lstm import LSTMModel
from utils.data_factory import create_dataloaders
from utils.evaluator import rmse, mape, mase
from utils.visualizer import (
    plot_prediction_vs_actual,
    plot_error_distribution,
    plot_residual_acf,
    plot_model_comparison,
    plot_attention_importance,
    plot_all_training_curves
)
from optuna_tuner import run_optimization
from trainer import train_model


# ------------------------------------------------------------
# GPU SETUP (RTX OPTIMIZED)
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n===== DEVICE CONFIGURATION =====")
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
    torch.backends.cudnn.benchmark = True  # RTX performance boost
else:
    print("Running on CPU")

print("=" * 40)


# ------------------------------------------------------------
# Evaluation Function (GPU Safe)
# ------------------------------------------------------------
def evaluate(model, test_loader, scaler_y):

    model.to(device)
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            output = model(X)

            preds.append(output.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())

    preds = np.vstack(preds)
    trues = np.vstack(trues)

    preds = scaler_y.inverse_transform(preds)
    trues = scaler_y.inverse_transform(trues)

    return trues, preds


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    print("\n===== ADVANCED TIME SERIES FORECASTING FRAMEWORK =====\n")

    USE_OPTUNA = True
    results = {}

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------
    SEQ_LEN = 168   # try 24 / 96 / 168
    BATCH_SIZE = 32

    train_loader, val_loader, test_loader, scaler_y, y_train_raw = create_dataloaders(
        seq_len=SEQ_LEN,
        batch_size=BATCH_SIZE
    )

    sample_batch, _ = next(iter(train_loader))
    actual_input_dim = sample_batch.shape[2]
    print(f"Detected Input Features: {actual_input_dim}")

    # --------------------------------------------------------
    # Extract test ground truth
    # --------------------------------------------------------
    print("Extracting test ground truth...")
    y_test_true = []
    for _, y in test_loader:
        y_test_true.append(y.numpy())

    y_test_true = np.vstack(y_test_true)
    y_test_true = scaler_y.inverse_transform(y_test_true)

    # --------------------------------------------------------
    # Seasonal Naive Baseline
    # --------------------------------------------------------
    print("Calculating Seasonal Naive Baseline...")
    start_time_naive = time.time()

    y_train_unscaled = y_train_raw.flatten()
    seasonal_lag = 24

    context = y_train_unscaled[-seasonal_lag:]
    full_series = np.concatenate([context, y_test_true.flatten()], axis=0)
    naive_pred_final = full_series[:-seasonal_lag].reshape(-1, 1)

    results["Naïve (Benchmark)"] = {
        "RMSE": rmse(y_test_true, naive_pred_final),
        "MAPE": mape(y_test_true, naive_pred_final),
        "MASE": 1.0,
        "Time": time.time() - start_time_naive,
        "train_loss": None,
        "val_loss": None
    }

    # --------------------------------------------------------
    # LSTM BASELINE
    # --------------------------------------------------------
    print("\nTraining LSTM Baseline...")
    start_time_lstm = time.time()

    lstm_model = LSTMModel(
        input_dim=actual_input_dim,
        hidden_dim=64,
        num_layers=2
    ).to(device)

    lstm_train_losses, lstm_val_losses = train_model(
        lstm_model,
        train_loader,
        val_loader,
        epochs=20,
        device=device
    )

    _, lstm_pred = evaluate(lstm_model, test_loader, scaler_y)

    results["LSTM (Baseline)"] = {
        "RMSE": rmse(y_test_true, lstm_pred),
        "MAPE": mape(y_test_true, lstm_pred),
        "MASE": mase(y_test_true, lstm_pred, y_train_raw),
        "Time": time.time() - start_time_lstm,
        "train_loss": lstm_train_losses,
        "val_loss": lstm_val_losses
    }

    # --------------------------------------------------------
    # TRANSFORMER (ATTENTION)
    # --------------------------------------------------------
    if USE_OPTUNA:
        print("\nRunning Bayesian Optimization for Transformer...")
        best_params = run_optimization()
        print(f"Best Params Found: {best_params}")
    else:
        best_params = {
            "d_model": 64,
            "n_heads": 4,
            "num_layers": 2,
            "dropout": 0.2,
            "lr": 1e-4
        }

    print("\nTraining Optimized Transformer...")
    start_time_trans = time.time()

    transformer_model = TemporalTransformer(
        input_dim=actual_input_dim,
        d_model=best_params["d_model"],
        n_heads=best_params["n_heads"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"]
    ).to(device)

    trans_train_losses, trans_val_losses = train_model(
        transformer_model,
        train_loader,
        val_loader,
        epochs=30,
        lr=best_params["lr"],
        device=device
    )

    _, trans_pred = evaluate(transformer_model, test_loader, scaler_y)

    results["Transformer (Attention)"] = {
        "RMSE": rmse(y_test_true, trans_pred),
        "MAPE": mape(y_test_true, trans_pred),
        "MASE": mase(y_test_true, trans_pred, y_train_raw),
        "Time": time.time() - start_time_trans,
        "train_loss": trans_train_losses,
        "val_loss": trans_val_losses
    }

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Model':<25} {'RMSE':<12} {'MAPE(%)':<12} {'MASE':<10} {'Time(s)':<8}")
    print("-" * 70)
    for model, m in results.items():
        print(f"{model:<25} {m['RMSE']:<12.4f} {m['MAPE']:<12.2f} {m['MASE']:<10.4f} {m['Time']:<8.2f}")
    print("=" * 70 + "\n")

    # --------------------------------------------------------
    # Save & Visualize
    # --------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).T.to_csv("results/models_comparison.csv")

    plot_all_training_curves(results)
    plot_model_comparison(results)
    plot_prediction_vs_actual(y_test_true, trans_pred)
    plot_error_distribution(y_test_true, trans_pred)
    plot_residual_acf(y_test_true, trans_pred)

    # --------------------------------------------------------
    # Attention Analysis (Device Safe)
    # --------------------------------------------------------
    sample_input = sample_batch[0].to(device)
    plot_attention_importance(transformer_model, sample_input)

    print("\nGPU Memory Allocated:",
          torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else "CPU Mode",
          "MB")
