import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def generate_energy_data(n_hours=24 * 365 * 2, seed=42):

    np.random.seed(seed)
    time = np.arange(n_hours)

    trend = 0.0003 * time
    daily = 10 * np.sin(2 * np.pi * time / 24)
    weekly = 5 * np.sin(2 * np.pi * time / (24 * 7))
    yearly = 15 * np.sin(2 * np.pi * time / (24 * 365))

    temperature = (
        20
        + 10 * np.sin(2 * np.pi * time / (24 * 365))
        + np.random.normal(0, 2, n_hours)
    )

    noise = np.random.normal(0, 2, n_hours)

    energy = (
        50
        + trend
        + daily
        + weekly
        + yearly
        - 0.5 * temperature
        + noise
    )

    df = pd.DataFrame({
        "energy": energy,
        "temperature": temperature,
        "hour": time % 24,
        "day_of_week": (time // 24) % 7,
        "month": (time // (24 * 30)) % 12
    })

    return df


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len, horizon):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.seq_len - self.horizon

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)


def create_dataloaders(seq_len=168, batch_size=32):

    df = generate_energy_data()

    features = df.drop(columns=["energy"]).values
    target = df["energy"].values.reshape(-1, 1)

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit(features[:train_end])
    scaler_y.fit(target[:train_end])

    X_scaled = scaler_X.transform(features)
    y_scaled = scaler_y.transform(target)

    train_dataset = TimeSeriesDataset(
        X_scaled[:train_end],
        y_scaled[:train_end],
        seq_len,
        1
    )

    val_dataset = TimeSeriesDataset(
        X_scaled[train_end:val_end],
        y_scaled[train_end:val_end],
        seq_len,
        1
    )

    test_dataset = TimeSeriesDataset(
        X_scaled[val_end:],
        y_scaled[val_end:],
        seq_len,
        1
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
        scaler_y,
        target[:train_end]
    )
