"""Ignite training cli."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import typer

from jarvisdgl.config import TrainingConfig
from jarvisdgl.train import train_dgl


def cli(
    config: Optional[Path] = typer.Argument(None),
    progress: bool = False,
    store_outputs: bool = False,
    tensorboard: bool = False,
):
    """Jarvis-dgl training cli.

    config: path to json config file (conform to TrainingConfig)
    progress: enable tqdm console logging
    tensorboard: enable tensorboard logging
    """
    if config is None:
        model_dir = os.getcwd()
        config = TrainingConfig(epochs=10, n_train=32, n_val=32, batch_size=16)

    elif config.is_file():
        model_dir = config.parent
        with open(config, "r") as f:
            config = json.load(f)
            config = TrainingConfig(**config)

    hist = train_dgl(
        config,
        progress=progress,
        store_outputs=store_outputs,
        log_tensorboard=tensorboard,
    )

    print(model_dir)
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(hist, f)

    with open(model_dir / "fullconfig.json", "w") as f:
        json.dump(json.loads(config.json()), f, indent=2)


if __name__ == "__main__":
    typer.run(cli)
