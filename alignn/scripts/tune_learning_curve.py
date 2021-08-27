"""tune_alignn_config.py.

schedule training runs to compare ALIGNN configuration
Is squeeze-expand compression helpful for perf/speed?
Does ALIGNN layer component ordering matter?
"""

import os

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from alignn.train import train_dgl

config = {
    "target": tune.grid_search(["optb88vdw_bandgap", "pair-triplet"]),
    "epochs": 200,
    "n_train": tune.grid_search([1024, 2048, 4096, 8192, 16384, 32768, 44578]),
    "n_val": 5500,
    "n_test": 5500,
    "num_workers": 8,
    "pin_memory": True,
    "output_dir": "test",
    "cache_dir": "/wrk/bld/alignn/data",
    "progress": True,
    "write_checkpoint": True,
    "write_predictions": False,
    "tune": True,
    "learning_rate": 3.25e-3,
    "weight_decay": 3e-4,
    "batch_size": 128,
    "model": {
        "name": "alignn",
        "alignn_layers": 2,
        "alignn_order": "pair-triplet",
        "squeeze_ratio": 1.0,
        "gcn_layers": 3,
        "edge_input_features": 16,
        "triplet_input_features": 40,
        "embedding_features": 128,
        "hidden_features": 256,
    },
}


if __name__ == "__main__":

    # ray.init(local_mode=True)
    ray.init(dashboard_port=os.environ["UID"])

    analysis = tune.run(
        train_dgl,
        num_samples=5,
        resources_per_trial={"cpu": 8, "gpu": 1},
        metric="mae",
        mode="min",
        # scheduler=ASHAScheduler(metric="mae", mode="min", grace_period=15),
        config=config,
        local_dir="./ray_ablation",
    )
    print(analysis)
