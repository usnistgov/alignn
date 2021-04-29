"""Training script test suite."""
import time

import matplotlib.pyplot as plt
import numpy as np

from alignn.train import train_dgl

plt.switch_backend("agg")
from sklearn.metrics import mean_absolute_error

config_megnet = {
    "version": "6812572f234019d91e5ed657d76b00254b965573",
    "dataset": "megnet",
    # "target": "optb88vdw_bandgap",
    "target": "e_form",
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "random_seed": 123,
    "n_train": 60000,
    "n_val": 5000,
    "n_test": 4539,
    "epochs": 250,
    "batch_size": 256,
    "weight_decay": 1e-05,
    "learning_rate": 0.01,
    "warmup_steps": 2000,
    "criterion": "mse",
    "id_tag": "id",
    "pin_memory": False,
    "num_workers": 0,
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "model": {
        "name": "dense_alignn",
        "alignn_layers": 7,
        "gcn_layers": 0,
        "atom_input_features": 92,
        "edge_input_features": 92,
        "triplet_input_features": 40,
        "embedding_features": 92,
        "initial_features": 92,
        "bottleneck_features": 92,
        "residual": False,
        "growth_rate": 128,
        "output_features": 1,
        "norm": "layernorm",
        "link": "identity",
        "zero_inflated": False,
    },
}


config_3d = {
    "version": "6812572f234019d91e5ed657d76b00254b965573",
    "dataset": "dft_3d",
    "target": "optb88vdw_bandgap",
    # "target": "formation_energy_peratom",
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "random_seed": 123,
    "n_val": 7200,
    "n_test": 7000,
    "n_train": 28800,
    "epochs": 100,
    "batch_size": 256,
    "weight_decay": 1e-05,
    "learning_rate": 0.01,
    "warmup_steps": 2000,
    "criterion": "mse",
    "optimizer": "adamw",
    "scheduler": "none",
    "model": {
        "name": "dense_alignn",
        "alignn_layers": 7,
        "gcn_layers": 0,
        "atom_input_features": 92,
        "edge_input_features": 92,
        "triplet_input_features": 40,
        "embedding_features": 92,
        "initial_features": 92,
        "bottleneck_features": 92,
        "residual": False,
        "growth_rate": 128,
        "output_features": 1,
        "norm": "layernorm",
        "link": "identity",
        "zero_inflated": False,
    },
}

config_2d = {
    "version": "6812572f234019d91e5ed657d76b00254b965573",
    "dataset": "dft_2d",
    "target": "formation_energy_peratom",
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "random_seed": 123,
    "n_val": 72,
    "n_test": 72,
    "n_train": 288,
    "epochs": 3,
    "batch_size": 25,
    "weight_decay": 1e-05,
    "learning_rate": 0.01,
    "warmup_steps": 2000,
    "criterion": "mse",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "model": {
        "name": "dense_alignn",
        "alignn_layers": 7,
        "gcn_layers": 0,
        "atom_input_features": 92,
        "edge_input_features": 92,
        "triplet_input_features": 40,
        "embedding_features": 92,
        "initial_features": 92,
        "bottleneck_features": 92,
        "residual": False,
        "growth_rate": 128,
        "output_features": 1,
        "norm": "layernorm",
        "link": "identity",
        "zero_inflated": False,
    },
}


def test_cgcnn_ignite():
    """Test CGCNN end to end training."""
    config = config_2d  # megnet
    result = train_dgl(config)
    # result = train_dgl(config_2d)
    # result = train_dgl(config_gap)
    # print("EOS", result["EOS"])
    print("train=", result["train"])
    print("validation=", result["validation"])
    x = []
    y = []
    for i in result["EOS"]:
        x.append(i[0].cpu().numpy().tolist())
        y.append(i[1].cpu().numpy().tolist())
    x = np.array(x, dtype="float").flatten()
    y = np.array(y, dtype="float").flatten()
    f = open("res", "w")
    for i, j in zip(x, y):
        line = str(i) + "," + str(j) + "\n"
        f.write(line)
    f.close()
    plt.plot(x, y, ".")
    plt.xlabel("DFT")
    plt.ylabel("ML")
    plt.savefig("compare.png")
    plt.close()


t1 = time.time()
test_cgcnn_ignite()
t2 = time.time()
print(t2 - t1)
