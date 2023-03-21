import sys

import torch
import torch.nn.functional as F

from test import *
from util.data import *
from util.time import *


def train(
    model=None,
    save_path="",
    config={},
    train_dataloader=None,
    val_dataloader=None,
    device=None,
    test_dataloader=None,
    test_dataset=None,
    dataset_name="swat",
    train_dataset=None,
):
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["decay"]
    )

    alpha = config["alpha"]
    beta = config["beta"]
    epochs = config["epoch"]

    patience = epochs // 5
    epochs_improved = 0

    train_loss_list = []
    min_loss = sys.float_info.max
    reduction = "mean"
    for epoch in range(epochs):
        model.train()

        total_loss = 0
        for (
            x,
            y,
            _,
            edge_index,
        ) in train_dataloader:
            x = torch.add(x, generate_gaussian_noise(x))
            x, y, edge_index = [item.float().to(device) for item in [x, y, edge_index]]

            optimizer.zero_grad()

            (
                x_hat,
                x_recon,
                _,
                mu,
                log_var,
                _,
                _,
                _,
                _,
                rhos,
                rho_hat,
            ) = model(x, y, edge_index)

            x_hat = x_hat.float().to(device)
            x_recon = x_recon.float().to(device)
            if (mu is not None) and (log_var is not None):
                mu = mu.float().to(device)
                log_var = log_var.float().to(device)

            loss_frcst = torch.sqrt(F.mse_loss(x_hat, y, reduction=reduction))
            loss_recon = F.mse_loss(x_recon, x, reduction=reduction)
            loss_recon += beta * kl_divergence(rhos, rho_hat)
            loss = alpha * loss_frcst + (1.0 - alpha) * loss_recon

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_loss_list.append(total_loss)

        train_loss_log = f"F: {alpha * loss_frcst:.4f} | R: {(1.0 - alpha) * loss_recon:.4f} | beta*KLD: {beta * kl_divergence(rhos, rho_hat)}"
        if val_dataloader is not None:
            val_loss, _ = test(model, val_dataloader, device, config=config)
            val_loss_log = f"V: {val_loss:.4f}"
            loss_log = f"[E {epoch + 1}/{epochs}] " + " | ".join(
                [val_loss_log, train_loss_log]
            )
            print(loss_log, flush=True)
            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = val_loss
                epochs_improved = 0
            else:
                epochs_improved += 1

            if epochs_improved >= patience:
                break

    return train_loss_list
