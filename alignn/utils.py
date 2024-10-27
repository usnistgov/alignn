"""Shared pydantic settings configuration."""

import json
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
from pydantic_settings import BaseSettings as PydanticBaseSettings
import torch
import pickle as pk
import os


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"


def plot_learning_curve(
    results_dir: Union[str, Path], key: str = "mae", plot_train: bool = False
):
    """Plot learning curves based on json history files."""
    if isinstance(results_dir, str):
        results_dir = Path(results_dir)

    with open(results_dir / "history_val.json", "r") as f:
        val = json.load(f)

    p = plt.plot(val[key], label=results_dir.name)

    if plot_train:
        # plot the training trace in the same color, lower opacity
        with open(results_dir / "history_train.json", "r") as f:
            train = json.load(f)

        c = p[0].get_color()
        plt.plot(train[key], alpha=0.5, c=c)

    plt.xlabel("epochs")
    plt.ylabel(key)

    return train, val


def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output, tmp_output_dir="out"):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(
        sc.transform(y_pred.cpu().numpy()), device=y_pred.device
    )
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=y.device)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer


def print_train_val_loss(
    e,
    running_loss,
    running_loss1,
    running_loss2,
    running_loss3,
    running_loss4,
    val_loss,
    val_loss1,
    val_loss2,
    val_loss3,
    val_loss4,
    train_ep_time,
    val_ep_time,
    saving_msg="",
):
    """Train loss header."""
    header = ("{:<12} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}").format(
        "Train Loss:",
        "Epoch",
        "Total",
        "Graph",
        "Atom",
        "Grad",
        "Stress",
        "Time",
    )
    print(header)

    # Train loss values
    train_row = (
        "{:<12} {:<8} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} "
        "{:<10.2f}"
    ).format(
        "",
        e,
        running_loss,
        running_loss1,
        running_loss2,
        running_loss3,
        running_loss4,
        train_ep_time,
    )
    print(train_row)

    # Validation loss header
    header = ("{:<12} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}").format(
        "Val Loss:",
        "Epoch",
        "Total",
        "Graph",
        "Atom",
        "Grad",
        "Stress",
        "Time",
    )
    print(header)

    # Validation loss values
    val_row = (
        "{:<12} {:<8} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} "
        "{:<10.2f} {:<10}"
    ).format(
        "",
        e,
        val_loss,
        val_loss1,
        val_loss2,
        val_loss3,
        val_loss4,
        val_ep_time,
        saving_msg,
    )
    print(val_row)
