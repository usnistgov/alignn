"""Training script test suite."""
import time

import matplotlib.pyplot as plt
import numpy as np

from jarvisdgl.main import train_property_model
from jarvisdgl.train import train_dgl

plt.switch_backend("agg")
from sklearn.metrics import mean_absolute_error


def test_prop():
    """Test full training run with small batch size."""
    train_property_model(epochs=2, maxrows=16, batch_size=8)


config_d = {
    "dataset": "dft_3d",
    "target": "formation_energy_peratom",
    "random_seed": 123,
    "n_val": 7200,
    "n_train": 28800,
    "epochs": 100,
    "batch_size": 256,
    "weight_decay": 0,
    "learning_rate": 0.01,
    "criterion": "mse",
    "atom_features": "basic",
    "optimizer": "adamw",
    "conv_layers": 3,
    "edge_features": 16,
    "node_features": 64,
    "fc_layers": 1,
    "fc_features": 128,
    "output_features": 1,
    "logscale": False,
}

config_dd = {
    "dataset": "dft_3d",
    "target": "optb88vdw_bandgap",
    "random_seed": 123,
    "n_val": 100,  # 00,
    "n_train": 200,  # 00,
    "epochs": 3,  # 100,
    "batch_size": 50,  # 6,
    "weight_decay": 0,
    "learning_rate": 0.01,
    "criterion": "mse",
    "atom_features": "mit",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "conv_layers": 3,
    "edge_features": 41,
    "node_features": 64,
    "fc_layers": 1,
    "fc_features": 128,
    "output_features": 1,
    "logscale": False,
    "log_tensorboard": False,
}

config_d = {
    "dataset": "dft_3d",
    "target": "optb88vdw_bandgap",
    "random_seed": 123,
    "n_val": 7200,
    "n_train": 28800,
    "epochs": 100,
    "batch_size": 256,
    "weight_decay": 0,
    "learning_rate": 0.01,
    "criterion": "mse",
    "atom_features": "mit",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "conv_layers": 3,
    "edge_features": 41,
    "node_features": 64,
    "fc_layers": 1,
    "fc_features": 128,
    "output_features": 1,
    "logscale": False,
}


def test_cgcnn_ignite():
    """Test CGCNN end to end training."""
    config = dict(
        target="formation_energy_peratom",
        epochs=2,
        n_train=16,
        n_val=16,
        batch_size=8,
    )
    result = train_dgl(config)
    # print(result)
    x = []
    y = []
    for i in result["EOS"]:
        for j in i:
            x.append(j[0].cpu().numpy().tolist())
            y.append(j[1].cpu().numpy().tolist())
    plt.plot(x, y, ".")
    plt.savefig("compare.png")
    plt.close()


t1 = time.time()
test_cgcnn_ignite()
t2 = time.time()
print(t2 - t1)
