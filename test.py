import time

import torch
import torch.nn.functional as F

from util.data import *
from util.preprocess import *
from util.time import *


def test(model, dataloader, device, config={}):
    now = time.time()
    model.eval()

    test_len = len(dataloader)
    y_hat_list = []
    y_list = []
    y_label_list = []
    x_hat_list = []
    x_list = []
    x_label_list = []

    i = 0
    alpha = config["alpha"]
    beta = config["beta"]

    test_loss_list = []
    total_loss = 0
    reduction = "mean"
    for (
        x,
        y,
        labels,
        edge_index,
    ) in dataloader:
        x, y, edge_index = [item.float().to(device) for item in [x, y, edge_index]]
        with torch.no_grad():
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

            predicted_y = x_hat
            y_labels = labels.unsqueeze(1).repeat(1, predicted_y.shape[1])
            if len(y_hat_list) <= 0:
                y_hat_list = predicted_y
                y_list = y
                y_label_list = y_labels
            else:
                y_hat_list = torch.cat((y_hat_list, predicted_y), dim=0)
                y_list = torch.cat((y_list, y), dim=0)
                y_label_list = torch.cat((y_label_list, y_labels), dim=0)

            predicted_x = x_recon
            x_labels = labels.unsqueeze(1).repeat(1, predicted_x.shape[1])
            if len(x_hat_list) <= 0:
                x_hat_list = predicted_x
                x_list = x
                x_label_list = x_labels
            else:
                x_hat_list = torch.cat((x_hat_list, predicted_x), dim=0)
                x_list = torch.cat((x_list, x), dim=0)
                x_label_list = torch.cat((x_label_list, x_labels), dim=0)

        loss = alpha * loss_frcst + (1.0 - alpha) * loss_recon

        total_loss += loss.item()
        test_loss_list.append(loss.item())

        i += 1
        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    y_hat_list = y_hat_list.tolist()
    y_list = y_list.tolist()
    y_label_list = y_label_list.tolist()
    x_hat_list = x_hat_list.tolist()
    x_list = x_list.tolist()
    x_label_list = x_label_list.tolist()
    val_loss = sum(test_loss_list) / len(test_loss_list)
    return val_loss, {
        "forecasting": [y_hat_list, y_list, y_label_list],
        "reconstruction": [x_hat_list, x_list, x_label_list],
    }
