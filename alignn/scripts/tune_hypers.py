"""tune_hypers.py.

set up random hyperparameter search
with Asynchronous Hyperband trial scheduler.
"""

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from alignn.train import train_dgl

config = {
    "random_seed": 123,
    # "target": "formation_energy_per_atom",
    "target": "optb88vdw_bandgap",
    "epochs": 10,
    "n_train": 32,
    "n_val": 32,
    "n_test": 32,
    "batch_size": 32,
    "num_workers": 1,
    "output_dir": "test",
    "progress": True,
    "write_checkpoint": True,
    "write_predictions": False,
    "tune": True,
    "learning_rate": tune.loguniform(1e-4, 1e-1),
}


if __name__ == "__main__":

    # ray.init(local_mode=True)

    analysis = tune.run(
        train_dgl,
        num_samples=2,
        scheduler=ASHAScheduler(metric="mae", mode="min", grace_period=5),
        config=config,
        local_dir="./ray_results",
    )
    print(analysis)
