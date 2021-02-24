"""Ignite training script.

from the repository root, run
`PYTHONPATH=$PYTHONPATH:. python jarvisdgl/train.py`
then `tensorboard --logdir tb_logs/test` to monitor results...
"""
from dataclasses import dataclass

import torch
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import (
    OutputHandler,
    global_step_from_engine,
)
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.handlers import TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError, RunningAverage
from torch import nn

from jarvisdgl import data, models

# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


# @dataclass
# class TrainingConfig:
#     dataset: str = "dft_3d"
#     target: str = "formation_energy_peratom"
#     n_train: int = 1024
#     n_val: int = 1024
#     batch_size: int = 32
#     atom_features: str = "basic"


# load tiny prototyping datasets
n = 128
bs = 64
train_loader, val_loader = data.get_train_val_loaders(
    n_train=n, n_val=n, batch_size=bs, atom_features="basic", normalize=True
)

# define network, optimizer, scheduler
n_epochs = 250

parameters = {
    "atom_input_features": 11,
    "conv_layers": 3,
    "edge_features": 16,
    "node_features": 64,
}
net = models.CGCNN(**parameters)
net.to(device)

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-2,
    max_momentum=0.92,
    base_momentum=0.88,
    epochs=n_epochs,
    steps_per_epoch=len(train_loader),
)
criterion = nn.MSELoss()

# set up training engines
metrics = {"loss": Loss(criterion), "mae": MeanAbsoluteError()}
trainer = create_supervised_trainer(
    net,
    optimizer,
    criterion,
    prepare_batch=data.prepare_dgl_batch,
    device=device,
)
evaluator = create_supervised_evaluator(
    net, metrics=metrics, prepare_batch=data.prepare_dgl_batch, device=device
)
train_evaluator = create_supervised_evaluator(
    net, metrics=metrics, prepare_batch=data.prepare_dgl_batch, device=device
)

# apply learning rate scheduler
trainer.add_event_handler(
    Events.ITERATION_COMPLETED, lambda engine: scheduler.step()
)

trainer.add_event_handler(Events.EPOCH_COMPLETED, TerminateOnNan())


# collect evaluation performance
@trainer.on(Events.EPOCH_COMPLETED)
def log_results(engine):
    """Print training and validation metrics to console."""
    train_evaluator.run(train_loader)
    evaluator.run(val_loader)

    epoch = trainer.state.epoch
    tmetrics = train_evaluator.state.metrics
    vmetrics = evaluator.state.metrics
    tloss, vloss = tmetrics["loss"], vmetrics["loss"]
    tmae, vmae = tmetrics["mae"], vmetrics["mae"]

    print(f"Epoch: {epoch} loss: {tloss:.2f} ({vloss:.2f})")
    print(f"Epoch: {epoch} MAE: {tmae:.2f} ({vmae:.2f})")


# log results to tensorboard
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

trainer.run(train_loader, max_epochs=n_epochs)

test_loss = evaluator.state.metrics["loss"]
tb_logger.writer.add_hparams(parameters, {"hparam/test_loss": test_loss})

print(test_loss)
tb_logger.close()
