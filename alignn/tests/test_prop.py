"""Training script test suite."""
import time

import matplotlib.pyplot as plt
import numpy as np

from alignn.train import train_dgl

plt.switch_backend("agg")
from sklearn.metrics import mean_absolute_error


config = {
    "dataset": "dft_2d",
    "target": "formation_energy_peratom",
    # "target": "optb88vdw_bandgap",
    "n_train": 50,
    "n_test": 25,
    "n_val": 25,
    "num_workers": 0,
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "epochs": 2,
    "save_dataloader": False,
    "batch_size": 10,
    "weight_decay": 1e-05,
    "learning_rate": 0.01,
    "criterion": "mse",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "num_workers": 4,
    "model": {
        "name": "alignn",
    },
}


def test_models():
    """Test CGCNN end to end training."""
    config["model"]["name"] = "dense_alignn"
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Toal time:", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "alignn"
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "cgcnn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "densegcn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "icgcnn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "alignn_cgcnn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    # Classification
    config["model"]["name"] = "dense_alignn"
    config["classification_threshold"] = 0.0
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Toal time:", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "alignn"
    config["classification_threshold"] = 0.0
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "cgcnn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    config["classification_threshold"] = 0.0
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    config["model"]["name"] = "alignn_cgcnn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    config["classification_threshold"] = 0.0
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()

    """

    config["model"]["name"] = "simplegcn"
    config["write_predictions"] = False
    config["save_dataloader"] = False
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    print("train=", result["train"])
    print("validation=", result["validation"])
    print()
    print()
    print()
    """
    """
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
    """


# test_models()
