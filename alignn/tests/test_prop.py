"""Training script test suite."""
import time

import matplotlib.pyplot as plt
import numpy as np

from alignn.train import train_dgl

plt.switch_backend("agg")
from sklearn.metrics import mean_absolute_error

config_qm9 = {
    "dataset": "qm9",
    "target": "gap",
    "id_tag": "id",
    "random_seed": 123,
    "neighbor_strategy": "k-nearest",
    "train_ratio": None,  # 00,
    "val_ratio": None,  # 00,
    "test_ratio": None,  # 00,
    "train_size": 110000,
    "test_size": 13885,
    "val_size": 10000,
    "epochs": 250,  # 100,
    "batch_size": 500,  # 6,
    "weight_decay": 0,
    "learning_rate": 0.01,
    "criterion": "mse",
    "atom_features": "cgcnn",
    "optimizer": "adamw",
    "filename": "sample",
    "scheduler": "none",
    "save_dataloader": True,
    "model": {
        "name": "alignn_edge",
        "filename": "sample",
        "node_input_features": 92,
        "hidden_features": 92,
        "edge_input_features": 128,
        "embedding_features": 92,
        "triplet_input_features": 40,
        "alignn_layers": 3,
        "gcn_layers": 3,
        "output_features": 1,
    },
}


config_megnet = {
    "dataset": "megnet",
    "target": "e_form",
    "id_tag": "id",
    "random_seed": 123,
    "neighbor_strategy": "k-nearest",
    "train_ratio": None,  # 00,
    "val_ratio": None,  # 00,
    "test_ratio": None,  # 00,
    "train_size": 60000,
    "test_size": 4539,
    "val_size": 5000,
    "epochs": 250,  # 100,
    "batch_size": 500,  # 6,
    "weight_decay": 0,
    "learning_rate": 0.01,
    "criterion": "mse",
    "atom_features": "cgcnn",
    "optimizer": "adamw",
    "filename": "sample",
    "scheduler": "none",
    "save_dataloader": True,
    "model": {
        "name": "alignn_edge",
        "filename": "sample",
        "node_input_features": 92,
        "hidden_features": 92,
        "edge_input_features": 128,
        "embedding_features": 92,
        "triplet_input_features": 40,
        "alignn_layers": 3,
        "gcn_layers": 3,
        "output_features": 1,
    },
}
config_2d = {
    "dataset": "dft_2d",
    "target": "optb88vdw_bandgap",
    "id_tag": "jid",
    "random_seed": 123,
    "neighbor_strategy": "k-nearest",
    "train_ratio": None,  # 00,
    "val_ratio": None,  # 00,
    "test_ratio": None,  # 00,
    "train_size": 800,
    "test_size": 100,
    "val_size": 100,
    "epochs": 3,  # 100,
    "batch_size": 10,  # 6,
    "weight_decay": 0,
    "learning_rate": 0.01,
    "criterion": "mse",
    "atom_features": "cgcnn",
    "optimizer": "adamw",
    "id_tag": "jid",
    "filename": "sample",
    "scheduler": "none",
    "save_dataloader": True,
    "model": {
        "name": "alignn_edge",
        "filename": "sample",
        "node_input_features": 92,
        "hidden_features": 92,
        "edge_input_features": 128,
        "embedding_features": 92,
        "triplet_input_features": 40,
        "alignn_layers": 3,
        "gcn_layers": 3,
        "output_features": 1,
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
    x = np.array(x).flatten()
    y = np.array(y).flatten()
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
