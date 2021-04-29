"""Ignite training cli."""

import json
import os
import shutil
from pathlib import Path

# from typing import Any, Dict, Optional, Union
from typing import Optional

import torch
import typer

from alignn.config import TrainingConfig
from alignn.profile import profile_dgl
from alignn.train import train_dgl


def cli(
    config: Optional[Path] = typer.Argument(None),
    progress: bool = False,
    checkpoint_dir: Path = Path("/tmp/models"),
    store_outputs: bool = False,
    tensorboard: bool = False,
    profile: bool = False,
):
    """ALIGNN training cli.

    config: path to json config file (conform to TrainingConfig)
    progress: enable tqdm console logging
    tensorboard: enable tensorboard logging
    profile: run profiling script for one epoch instead of training
    """
    model_dir = config.parent

    if config is None:
        model_dir = os.getcwd()
        config = TrainingConfig(epochs=10, n_train=32, n_val=32, batch_size=16)

    elif config.is_file():
        model_dir = config.parent
        with open(config, "r") as f:
            config = json.load(f)
            config = TrainingConfig(**config)

    if profile:
        profile_dgl(config)
        return

    hist = train_dgl(
        config,
        progress=progress,
        checkpoint_dir=checkpoint_dir,
        store_outputs=store_outputs,
        log_tensorboard=tensorboard,
    )

    # print(model_dir)
    # with open(model_dir / "metrics.json", "w") as f:
    #     json.dump(hist, f)

    torch.save(hist, model_dir / "metrics.pt")

    with open(model_dir / "fullconfig.json", "w") as f:
        json.dump(json.loads(config.json()), f, indent=2)

    # move temporary checkpoint data into model_dir
    for checkpoint in checkpoint_dir.glob("*.pt"):
        shutil.copy(checkpoint, model_dir / checkpoint.name)


if __name__ == "__main__":
    typer.run(cli)
