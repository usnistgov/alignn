""" train.py

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
from ignite.metrics import Loss, MeanAbsoluteError, RunningAverage
from torch import nn

from jarvisdgl import data, models

# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

# load tiny prototyping datasets
train_loader, val_loader = data.get_train_val_loaders(
    n_train=1024, n_val=1024, batch_size=32, atom_features="basic"
)

# define network, optimizer, scheduler
n_epochs = 100

parameters = {
    "atom_input_features": 11,
    "conv_layers": 3,
    "edge_features": 32,
    "node_features": 64,
}
net = models.CGCNN(**parameters)
net.to(device)

optimizer = torch.optim.AdamW(net.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-4,
    epochs=n_epochs,
    steps_per_epoch=len(train_loader),
)
criterion = nn.MSELoss()

# set up training engines
metrics = {"loss": Loss(criterion)}
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


# collect evaluation performance
@trainer.on(Events.EPOCH_COMPLETED)
def log_results(engine):
    train_evaluator.run(train_loader)
    evaluator.run(val_loader)

    tmetrics = train_evaluator.state.metrics
    print(f"Epoch: {trainer.state.epoch}  train loss: {tmetrics['loss']:.2f}")
    metrics = evaluator.state.metrics
    print(f"Epoch: {trainer.state.epoch}  val loss: {metrics['loss']:.2f}")


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
        metric_names=["loss"],
        global_step_transform=global_step_from_engine(trainer),
    )

trainer.run(train_loader, max_epochs=n_epochs)

test_loss = evaluator.state.metrics["loss"]
tb_logger.writer.add_hparams(parameters, {"hparam/test_loss": test_loss})

print(test_loss)
tb_logger.close()
