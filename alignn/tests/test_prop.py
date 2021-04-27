"""Training script test suite."""
import time

import matplotlib.pyplot as plt
import numpy as np

from alignn.train import train_dgl

plt.switch_backend("agg")
from sklearn.metrics import mean_absolute_error



config_2d={
  "dataset": "dft_2d",
  "target": "optb88vdw_bandgap",
  "atom_features": "cgcnn",
  "neighbor_strategy": "k-nearest",
  "random_seed": 123,
  "n_val": 200,
  "n_train": 800,
  "epochs": 2,
  "batch_size": 25,
  "weight_decay": 1e-05,
  "learning_rate": 0.01,
  "warmup_steps": 2000,
  "criterion": "mse",
  "pin_memory": True,
  "workers":2,
  "id_tag":"jid",
  "optimizer": "adamw",
  "scheduler": "onecycle",
  "model": {
    "name": "dense_alignn",
    "initial_features": 92,
    "atom_input_features": 92,
    "edge_input_features": 92,
    "embedding_features": 92,
    "triplet_input_features": 40,
    "alignn_layers": 3,
    "gcn_layers": 3,
    "growth_rate": 32,
    "output_features": 1,
    "link": "logit",
    "zero_inflated": False
  }
}
config_megnet={
  "dataset": "megnet",
  "target": "gap pbe",
  "atom_features": "cgcnn",
  "neighbor_strategy": "k-nearest",
  "random_seed": 123,
  "n_val": 5000,
  "n_train": 60000,
  "epochs": 2,
  "batch_size": 256,
  "weight_decay": 1e-05,
  "learning_rate": 0.01,
  "warmup_steps": 2000,
  "criterion": "mse",
  "pin_memory": True,
  "workers":2,
  "id_tag":"id",
  "optimizer": "adamw",
  "scheduler": "onecycle",
  "model": {
    "name": "dense_alignn",
    "initial_features": 92,
    "atom_input_features": 92,
    "edge_input_features": 92,
    "embedding_features": 92,
    "triplet_input_features": 40,
    "alignn_layers": 3,
    "gcn_layers": 3,
    "growth_rate": 32,
    "output_features": 1,
    "link": "logit",
    "zero_inflated": False
  }
}
def test_cgcnn_ignite():
    """Test CGCNN end to end training."""
    config = dict(
        dataset="dft_2d",
        target="formation_energy_peratom",
        epochs=2,
        n_train=16,
        n_val=16,
        batch_size=8,
    )
    config = config_2d  # megnet
    print(config)
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
