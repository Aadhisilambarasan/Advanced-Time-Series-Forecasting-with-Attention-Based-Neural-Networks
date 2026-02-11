import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=30,
    lr=5e-4,
    device=None,
    use_amp=True,
    grad_clip=1.0
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
        verbose=False
    )

    criterion = nn.MSELoss()

    scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_val_loss = float("inf")
    patience = 7
    counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        # ---------------------------
        # TRAINING
        # ---------------------------
        model.train()
        total_train_loss = 0.0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device).view(-1, 1)

            optimizer.zero_grad()

            with autocast(enabled=(use_amp and device.type == "cuda")):
                output = model(X)
                loss = criterion(output, y)

            scaler.scale(loss).backward()

            # Gradient clipping (important for Transformer stability)
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # ---------------------------
        # VALIDATION
        # ---------------------------
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X, y in val_loader:

                X = X.to(device)
                y = y.to(device).view(-1, 1)

                with autocast(enabled=(use_amp and device.type == "cuda")):
                    output = model(X)
                    loss = criterion(output, y)

                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # ---------------------------
        # EARLY STOPPING
        # ---------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    return train_losses, val_losses
