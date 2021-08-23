"""tune_hypers.py.

set up random hyperparameter search
with Asynchronous Hyperband trial scheduler.
"""

import os

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from alignn.train import train_dgl

config = {
    "random_seed": 123,
    # "target": "formation_energy_per_atom",
    "target": "optb88vdw_bandgap",
    "epochs": 100,
    "n_test": 32,
    "num_workers": 4,
    "pin_memory": True,
    "output_dir": "test",
    "cache_dir": "/wrk/bld/alignn/data",
    "progress": True,
    "write_checkpoint": True,
    "write_predictions": False,
    "tune": True,
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(1e-7, 1e-3),
    "batch_size": tune.choice([16, 32, 64, 128, 256]),
    "model": {
        "name": "alignn",
        "alignn_layers": tune.randint(0, 4),
        "gcn_layers": tune.randint(0, 4),
        "edge_input_features": tune.qlograndint(8, 128, 8),
        "triplet_input_features": tune.qlograndint(8, 128, 8),
        "embedding_features": tune.qlograndint(16, 128, 16),
        "hidden_features": tune.choice([16, 32, 64, 128, 256, 512]),
        "link": tune.choice(["identity", "log"]),
    },
}


if __name__ == "__main__":

    # ray.init(local_mode=True)
    ray.init(dashboard_port=os.environ["UID"])

    analysis = tune.run(
        train_dgl,
        num_samples=10,
        resources_per_trial={"cpu": 4, "gpu": 1},
        scheduler=ASHAScheduler(metric="mae", mode="min", grace_period=5),
        config=config,
        local_dir="./ray_results",
    )
    print(analysis)
