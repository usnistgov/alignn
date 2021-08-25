"""tune_hypers.py.

set up random hyperparameter search
with Asynchronous Hyperband trial scheduler.
"""

import os

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from alignn.train import train_dgl

# coarse: with ASHA
# tune learning rate in 1e-4, 1e-2
# decay in 1e-7, 1e-4
# tune hidden features in 128, 512?

# finer: full training runs
# tune learning rate in 3e-4, 1e-2
# decay in 1e-7, 1e-4
# tune hidden features in 200, 256, 400

# fine:
# learning rate: 3e-4, 3e-2
# decay: 3e-6, 3e-3
# fix hidden size to 512
# fix embedding features to 32

config = {
    "random_seed": 123,
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
    "learning_rate": tune.loguniform(3e-4, 3e-2),
    "weight_decay": tune.loguniform(3e-6, 3e-3),
    "batch_size": 256,
    "model": {
        "name": "alignn",
        "alignn_layers": 2,
        "gcn_layers": 2,
        "edge_input_features": 32,
        "triplet_input_features": 32,
        "embedding_features": 32,
        "hidden_features": 512,
        "link": "identity",
    },
}


if __name__ == "__main__":

    # ray.init(local_mode=True)
    ray.init(dashboard_port=os.environ["UID"])

    analysis = tune.run(
        train_dgl,
        num_samples=100,
        resources_per_trial={"cpu": 4, "gpu": 1},
        metric="mae",
        mode="min",
        # scheduler=ASHAScheduler(metric="mae", mode="min", grace_period=15),
        config=config,
        local_dir="./ray_results",
    )
    print(analysis)
