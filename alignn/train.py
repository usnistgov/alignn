"""Ignite training script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python jarvisdgl/train.py`
then `tensorboard --logdir tb_logs/test` to monitor results...
"""

from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Union

import os, random, ignite
import numpy as np
import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.stores import EpochOutputStore
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
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn

# from jarvis.core.graphs import get_train_val_loaders

# from jarvisdgl.config import TrainingConfig
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from jarvis.db.figshare import data as jdata
from jarvis.core.graphs import StructureDataset
from alignn.all_models import (
    GCNSimple,
    CGCNNSimple,
    ALIGNNSimple,
    ALIGNNEdge,
    ALIGNNCF,
    ZeroInflatedGammaLoss,
)
from torch.utils.data.sampler import SubsetRandomSampler
from jarvis.core.graphs import prepare_line_graph_batch as prepare_dgl_batch


class RecursiveNamespace:
    """Module for calling dict values using dot."""

    @staticmethod
    def map_entry(entry):
        """Map entry data."""
        if isinstance(entry, dict):
            print("entry", entry)
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        """Initialize class."""
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))
            else:  # this is the only addition
                setattr(self, key, val)


def get_train_val_test_loaders(
    dataset_name="dft_2d",
    target="formation_energy_peratom",
    id_tag="jid",
    atom_features="cgcnn",
    neighbor_strategy="k-nearest",
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    train_size=None,
    val_size=None,
    test_size=None,
    filename="tsample",
    save_dataloader=True,
    batch_size: int = 8,
    split_seed=123,
    # target_mult_factor=1.0,
    pin_memory=False,
    num_workers=1,
):
    """Help function to set up Jarvis train and val dataloaders."""
    train_sample = filename + "_train.data"
    val_sample = filename + "_val.data"
    test_sample = filename + "_test.data"

    if (
        os.path.exists(train_sample)
        and os.path.exists(val_sample)
        and os.path.exists(test_sample)
        and save_dataloader
    ):
        print("Loading from saved file...")
        print("Make sure all the DataLoader params are same.")
        print("This module is made for debugging only.")
        train_loader = torch.load(train_sample)
        val_loader = torch.load(val_sample)
        test_loader = torch.load(test_sample)
        # print("train", len(train_loader.dataset))
        # print("val", len(val_loader.dataset))
        # print("test", len(test_loader.dataset))
    else:

        if train_ratio is not None and train_size is not None:
            raise ValueError(
                "Provide either ratio based splits or size based splits."
            )

        # if target_mult_factor is not None:
        #    targets = target_mult_factor * targets
        d = jdata(dataset_name)
        total_size = len(d)
        if (
            train_ratio is None
            and val_ratio is not None
            and test_ratio is not None
        ):
            if train_ratio is None:
                assert val_ratio + test_ratio < 1
                train_ratio = 1 - val_ratio - test_ratio
                print(
                    "Using rest of the dataset except the test and val sets."
                )
            else:
                assert train_ratio + val_ratio + test_ratio <= 1
        indices = list(range(total_size))
        if train_size is None:
            train_size = int(train_ratio * total_size)
        if test_size is None:
            test_size = int(test_ratio * total_size)
        if val_size is None:
            val_size = int(val_ratio * total_size)

        structures, targets, jv_ids = [], [], []
        for row in d:
            if row[target] != "na":
                structures.append(row["atoms"])
                targets.append(row[target])
                jv_ids.append(row[id_tag])
        structures = np.array(structures)
        targets = np.array(targets)
        jv_ids = np.array(jv_ids)

        tids = np.arange(len(structures))
        random.seed(split_seed)
        random.shuffle(tids)
        if train_size + val_size + test_size > len(structures):
            raise ValueError("Check total number of samples.")

        tid_train = tids[0:train_size]
        tid_val = tids[-(val_size + test_size) : -test_size]
        tid_test = tids[-test_size:]  # noqa:E203

        X_train = structures[tid_train]
        X_val = structures[tid_val]
        X_test = structures[tid_test]

        y_train = targets[tid_train]
        y_val = targets[tid_val]
        y_test = targets[tid_test]

        id_train = jv_ids[tid_train]
        id_val = jv_ids[tid_val]
        id_test = jv_ids[tid_test]

        train_data = StructureDataset(
            X_train,
            y_train,
            ids=id_train,
            atom_features=atom_features,
            neighbor_strategy=neighbor_strategy,
        )

        val_data = StructureDataset(
            X_val,
            y_val,
            ids=id_val,
            atom_features=atom_features,
            neighbor_strategy=neighbor_strategy,
            transform=train_data.transform,
        )

        test_data = StructureDataset(
            X_test,
            y_test,
            ids=id_test,
            atom_features=atom_features,
            neighbor_strategy=neighbor_strategy,
            transform=train_data.transform,
        )

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_data.collate,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_data.collate,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=val_data.collate,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if save_dataloader:
            torch.save(train_loader, train_sample)
            torch.save(val_loader, val_sample)
            torch.save(test_loader, test_sample)
    print("train", len(train_loader.dataset))
    print("val", len(val_loader.dataset))
    print("test", len(test_loader.dataset))
    # from jarvis.core.graphs import prepare_line_graph_batch

    return train_loader, val_loader, test_loader
    # return train_loader, val_loader, prepare_line_graph_batch


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


def setup_optimizer(params, config={}):
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


def train_dgl(
    config={},
    model: nn.Module = None,
    progress: bool = True,
    checkpoint_dir: Path = Path("./"),
    store_outputs: bool = True,
    log_tensorboard: bool = False,
):
    """Training entry point for DGL networks.

    `config` should conform to jarvisdgl.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    if isinstance(config, dict):
        try:
            config = RecursiveNamespace.map_entry(config)
            # config = TrainingConfig(**config)
        except Exception as exp:
            print("Check here", exp)
    deterministic = False
    if config.random_seed is not None:
        deterministic = True
        ignite.utils.manual_seed(config.random_seed)

    # torch config
    torch.set_default_dtype(torch.float32)

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # from jarvis.core.graphs import prepare_dgl_batch

    prepare_batch = partial(prepare_dgl_batch, device=device)

    # use input standardization for all real-valued feature sets
    # standardize = True
    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        dataset_name=config.dataset,
        target=config.target,
        id_tag=config.id_tag,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        train_ratio=config.train_ratio,
        test_ratio=config.test_ratio,
        val_ratio=config.val_ratio,
        filename=config.model.filename,
        batch_size=config.batch_size,
        save_dataloader=config.save_dataloader,
    )

    # define network, optimizer, scheduler
    if config.model.name == "cgcnn_simple":
        net = CGCNNSimple(config.model)
    elif config.model.name == "alignn_simple":
        net = ALIGNNSimple(config.model)
    elif config.model.name == "alignn_edge":
        net = ALIGNNEdge(config.model)
    elif config.model.name == "alignn_cf":
        net = ALIGNNCF(config.model)
    elif config.model.name == "gcn_simple":
        net = GCNSimple(config.model)
    else:
        raise ValueError(
            "Not implemented yet.",
            config.model.name,
            "choose from: cgcnn_simple,alignn_simple,alignn_edge,alignn_cf,gcn_simple",
        )
    net.to(device)

    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)
    ##config.scheduler = "none"
    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )
    """
    elif config.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            #max_lr=config.learning_rate,
            epochs=config.epochs,
            max_momentum=0.92,
            base_momentum=0.88,
            #total_steps=int(len(train_loader))
            #steps_per_epoch= int(len(train_loader)),
            steps_per_epoch= 2*int(len(train_loader)),
        )
    """

    # select configured loss function
    criteria = {
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "poisson": nn.PoissonNLLLoss(log_input=False, full=True),
        "zig": ZeroInflatedGammaLoss(),
    }
    criterion = criteria[config.criterion]

    # set up training engine and evaluators
    metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}
    if config.criterion == "zig":

        def zig_prediction_transform(x):
            output, y = x
            return criterion.predict(output), y

        metrics = {
            "loss": Loss(criterion),
            "mae": MeanAbsoluteError(
                output_transform=zig_prediction_transform
            ),
        }

    trainer = create_supervised_trainer(
        net,
        optimizer,
        criterion,
        prepare_batch=prepare_dgl_batch,
        device=device,
        deterministic=deterministic,
    )

    evaluator = create_supervised_evaluator(
        net, metrics=metrics, prepare_batch=prepare_dgl_batch, device=device
    )
    train_evaluator = create_supervised_evaluator(
        net, metrics=metrics, prepare_batch=prepare_dgl_batch, device=device
    )

    # ignite event handlers:
    trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())

    # apply learning rate scheduler
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )
    evaluator.add_event_handler(
        Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
    )

    # model checkpointing
    to_save = {
        "model": net,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "trainer": trainer,
    }
    handler = Checkpoint(
        to_save,
        DiskSaver(checkpoint_dir, create_dir=True, require_empty=False),
        n_saved=2,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)

    if progress:
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})
        # pbar.attach(evaluator, output_transform=lambda x: {"loss": x})

    history = {
        "train": {m: [] for m in metrics.keys()},
        "validation": {m: [] for m in metrics.keys()},
    }

    if store_outputs:
        # log_results handler will save epoch output
        # in history["EOS"]
        eos = EpochOutputStore()
        eos.attach(evaluator)
        train_eos = EpochOutputStore()
        train_eos.attach(train_evaluator)

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

        if store_outputs:
            history["EOS"] = eos.data
            history["trainEOS"] = train_eos.data

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
    evaluator.run(test_loader)

    if log_tensorboard:
        test_loss = evaluator.state.metrics["loss"]
        tb_logger.writer.add_hparams(config, {"hparam/test_loss": test_loss})
        tb_logger.close()

    return history


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config, progress=True)
