# Advanced Time Series Forecasting with Attention-Based Neural Networks 
Energy Demand Forecasting


# 🏗️ System Architecture

This project implements a production-oriented time series forecasting pipeline with explicit baseline benchmarking and attention-based modeling.

### Data Generation

* Synthetic multivariate energy consumption time series
* Trend + daily, weekly, and yearly seasonality
* Exogenous temperature variable
* Additive Gaussian noise

Designed to emulate realistic non-stationary demand dynamics.

### Models Implemented

**LSTM (Baseline)**

* Multi-layer recurrent architecture
* Fully connected regression head
* Sequential dependency modeling via hidden states

**Transformer (Attention Model)**

* Linear input projection
* Sinusoidal positional encoding
* Stacked Transformer encoder layers
* Multi-head self-attention
* Residual connections + LayerNorm
* Global average pooling
* Final regression head

Self-attention enables dynamic weighting of historical time steps, improving long-range dependency modeling.

### Optimization Strategy

* Bayesian hyperparameter tuning (Optuna)
* Adam optimizer with weight decay
* ReduceLROnPlateau scheduler
* Early stopping
* GPU acceleration with mixed precision (AMP)

---

# 📊 Performance Benchmark

Evaluation performed on a held-out test set using scale-aware forecasting metrics.

| Model                       | RMSE ↓   | MAPE ↓    | MASE ↓     | Training Time |
| --------------------------- | -------- | --------- | ---------- | ------------- |
| Naïve (Benchmark)           | 4.98     | 10.66%    | 1.0000     | 0.00s         |
| LSTM (Baseline)             | 3.82     | 8.33%     | 0.8986     | 23.48s        |
| **Transformer (Attention)** | **3.44** | **7.55%** | **0.7994** | 22.38s        |

### Key Observations

* ~20% improvement over seasonal naïve baseline (MASE)
* Lowest RMSE, reducing large-error risk
* Comparable training cost to LSTM despite higher model complexity
* Attention mechanism delivers measurable reliability gains

---

# 📂 Project Structure

```
.
├── main.py                 # End-to-end experiment orchestration
├── trainer.py              # Training loop (AMP + early stopping)
├── optuna_tuner.py         # Bayesian hyperparameter optimization
├── requirements.txt
│
├── models/
│   ├── lstm.py             # LSTM baseline implementation
│   └── transformer.py      # Attention-based Transformer model
│
├── utils/
│   ├── data_factory.py     # Dataset generation & dataloaders
│   ├── evaluator.py        # RMSE, MAPE, MASE metrics
│   └── visualizer.py       # Diagnostic and forecast plots
│
└── results/                # Generated evaluation artifacts
```

---

# 🎯 Key Takeaways

* End-to-end forecasting system (not a notebook experiment)
* Explicit, rigorous baseline comparison
* Custom self-attention implementation
* Bayesian hyperparameter optimization
* GPU-ready training with mixed precision
* Proper time-series evaluation (MASE-focused reliability)

---

## How to Run

```bash
pip install -r requirements.txt
python main.py
```
---

## 📬 Contact & Connect :  **LinkedIn:** [Silambarasan](https://dub.sh/koItrbj)

---









