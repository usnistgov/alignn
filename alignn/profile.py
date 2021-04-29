"""pytorch profiling script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python jarvisdgl/profile.py`
"""

from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union

# import numpy as np
import torch
import torch.profiler
from torch import nn
from tqdm import tqdm

from alignn import data, models
from alignn.config import TrainingConfig
from alignn.train import group_decay, setup_optimizer

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def profile_dgl(config: Union[TrainingConfig, Dict[str, Any]]):
    """Training entry point for DGL networks.

    `config` should conform to alignn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    if type(config) is dict:
        config = TrainingConfig(**config)

    lg_models = {"clgn", "alignn"}

    # use input standardization for all real-valued feature sets

    train_loader, val_loader, prepare_batch = data.get_train_val_loaders(
        target=config.target,
        n_train=config.n_train,
        n_val=config.n_val,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        line_graph=config.model.name in lg_models,
    )
    prepare_batch = partial(prepare_batch, device=device)

    # define network, optimizer, scheduler
    _model = {
        "cgcnn": models.CGCNN,
        "icgcnn": models.iCGCNN,
        "densegcn": models.DenseGCN,
        "clgn": models.CLGN,
        "alignn": models.ALIGNN,
    }
    model = _model.get(config.model.name)(config.model)
    model.to(device)

    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(model)
    optimizer = setup_optimizer(params, config)

    criterion = nn.MSELoss()

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("."),
        with_stack=True,
        profile_memory=True,
    ) as profiler:
        # train for one epoch
        for batch in tqdm(train_loader):
            g, y = prepare_batch(batch)
            pred = model(g)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            profiler.step()
