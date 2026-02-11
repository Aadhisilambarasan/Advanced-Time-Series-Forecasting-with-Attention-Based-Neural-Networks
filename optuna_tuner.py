import optuna
import torch
from models.transformer import TemporalTransformer
from trainer import train_model
from utils.data_factory import create_dataloaders

def objective(trial):
    # Hyperparameter search space
    d_model = trial.suggest_categorical("d_model", [32, 64])
    n_heads = trial.suggest_categorical("n_heads", [2, 4])
    num_layers = trial.suggest_int("num_layers", 1, 2)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)

    train_loader, val_loader, _, _, _ = create_dataloaders()

    # Dynamic Dimension Check
    batch_data, _ = next(iter(train_loader))
    detected_dim = batch_data.shape[2]

    model = TemporalTransformer(
        input_dim=detected_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dropout=dropout
    )

    _, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        epochs=15,
        lr=lr
    )

    return val_losses[-1]

# MAKE SURE THIS IS AT THE BOTTOM AND NOT INDENTED
def run_optimization():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    return study.best_params