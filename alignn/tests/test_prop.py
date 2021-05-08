"""Training script test suite."""
import time

import matplotlib.pyplot as plt
import numpy as np

from alignn.train import train_dgl

plt.switch_backend("agg")
from sklearn.metrics import mean_absolute_error

config1 = {
    "dataset": "dft_3d",
    # "target": "e_form",
    "target": "formation_energy_peratom",
    # "target": "optb88vdw_bandgap",
    # "n_train": 60000,
    # "n_test": 4237,
    # "n_val": 5000,
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "epochs": 250,
    "use_canonize": True,
    "save_dataloader": True,
    "pin_memory": False,
    "num_workers": 4,
    "batch_size": 256,  # 6,
    "weight_decay": 1e-05,
    "learning_rate": 0.01,
    "criterion": "mse",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "filename": "ddsample",
    "use_canonize": False,
    "model": {
        "name": "alignn",
    },
}

config = {
    "dataset": "dft_3d",
    "target": "optb88vdw_bandgap",
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "random_seed": 123,
    "save_dataloader": True,
    "n_val": 7200,
    "n_test": 7000,
    "n_train": 28800,
    "epochs": 100,
    "filename": "dsample",
    "batch_size": 256,
    "weight_decay": 1e-05,
    "learning_rate": 0.01,
    "criterion": "mse",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "model": {
        "name": "dense_alignn",
        "alignn_layers": 3,
        "gcn_layers": 0,
        "initial_features": 128,
        "growth_rate": 64,
        "residual": True,
        "bottleneck_features": 128,
        "atom_input_features": 92,
        "edge_input_features": 40,
        "triplet_input_features": 40,
        "embedding_features": 128,
        "output_features": 1,
        "norm": "layernorm",
        "link": "identity",
        "zero_inflated": False,
    },
}


def test_models():
    """Test CGCNN end to end training."""
    t1 = time.time()
    result = train_dgl(config1)
    t2 = time.time()
    print("Toal time:", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    x = []
    y = []
    for i in result["EOS"]:
        x.append(i[0].cpu().numpy().tolist())
        y.append(i[1].cpu().numpy().tolist())
    x = np.array(x, dtype="float").flatten()
    y = np.array(y, dtype="float").flatten()
    plt.plot(x, y, ".")
    plt.xlabel("DFT")
    plt.ylabel("ML")
    plt.savefig("compare.png")
    plt.close()


test_models()
