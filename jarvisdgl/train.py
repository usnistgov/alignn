"""Ignite training script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python jarvisdgl/train.py`
then `tensorboard --logdir tb_logs/test` to monitor results...
"""

from functools import partial
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np
import ignite
import torch
from ignite.contrib.handlers.stores import EpochOutputStore
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import (
    OutputHandler,
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn

from jarvisdgl import data, models
from jarvisdgl.config import TrainingConfig


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def train_dgl(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    progress: bool = False,
    log_tensorboard: bool = False,
):
    """Training entry point for DGL networks.

    `config` should conform to jarvisdgl.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    if type(config) is dict:
        config = TrainingConfig(**config)

    deterministic = False
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    # torch config
    torch.set_default_dtype(torch.float32)

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")

    prepare_batch = partial(data.prepare_dgl_batch, device=device)

    # use input standardization for all real-valued feature sets
    standardize = True
    if config.atom_features.value == "mit":
        standardize = False

    train_loader, val_loader = data.get_train_val_loaders(
        target=config.target.value,
        n_train=config.n_train,
        n_val=config.n_val,
        batch_size=config.batch_size,
        atom_features=config.atom_features.value,
        standardize=standardize,
    )

    # define network, optimizer, scheduler
    if model is None:
        net = models.CGCNN(
            atom_input_features=config.atom_input_features,
            conv_layers=config.conv_layers,
            edge_features=config.edge_features,
            node_features=config.node_features,
            logscale=config.logscale,
        )
    else:
        net = model

    net.to(device)

    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)

    if config.optimizer.value == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.value == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )

    if config.scheduler.value == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler.value == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            max_momentum=0.92,
            base_momentum=0.88,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
        )

    # select configured loss function
    criteria = {"mse": nn.MSELoss(), "l1": nn.L1Loss()}
    criterion = criteria[config.criterion.value]

    # set up training engine and evaluators
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}
    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=data.prepare_dgl_batch,
        device=device,
        deterministic=deterministic,
    )

    evaluator = create_supervised_evaluator(
        net, metrics=metrics, prepare_batch=prepare_batch, device=device
    )
    train_evaluator = create_supervised_evaluator(
        net, metrics=metrics, prepare_batch=prepare_batch, device=device
    )
    eos = EpochOutputStore()
    eos.attach(evaluator)

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    if progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
        "predictions": [],
        "targets": [],
        "all_state": [],
        "EOS": [],
    }

    # collect evaluation performance
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        """Print training and validation metrics to console."""
        train_evaluator.run(train_loader)
        evaluator.run(val_loader)

        tmetrics = train_evaluator.state.metrics
        vmetrics = evaluator.state.metrics
        for metric in metrics.keys():
            history["train"][metric].append(tmetrics[metric])
            history["validation"][metric].append(vmetrics[metric])
        history["EOS"] = eos.data

    # optionally log results to tensorboard
    if log_tensorboard:

        tb_logger = TensorboardLogger(log_dir="tb_logs/test")
        for tag, evaluator in [
            ("training", train_evaluator),
            ("validation", evaluator),
        ]:
            tb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag=tag,
                metric_names=["loss", "mae"],
                global_step_transform=global_step_from_engine(trainer),
            )

    # train the model!
    trainer.run(train_loader, max_epochs=config.epochs)
    test_loss = evaluator.state.metrics["loss"]

    # if log_tensorboard:
    #    tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
    #    tb_logger.close()

    return history


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config, progress=True)
