"""Training script test suite."""

import time
import matplotlib.pyplot as plt
import numpy as np
from alignn.train import train_dgl
from alignn.pretrained import get_prediction
from alignn.pretrained import get_multiple_predictions
from sklearn.metrics import mean_absolute_error
import os
from jarvis.core.atoms import Atoms
from alignn.train_alignn import train_for_folder
from jarvis.db.figshare import get_jid_data
from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
import torch
from jarvis.db.jsonutils import loadjson, dumpjson
from alignn.config import TrainingConfig

world_size = int(torch.cuda.device_count())

plt.switch_backend("agg")

config = {
    "dataset": "dft_2d",
    "target": "formation_energy_peratom",
    # "target": "optb88vdw_bandgap",
    "n_train": 4,
    "n_test": 4,
    "n_val": 4,
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "epochs": 2,
    "batch_size": 2,
    "model": {
        "name": "ealignn_atomwise",
        "calculate_gradient": False,
        "energy_mult_natoms": False,
        "atom_input_features": 92,
    },
}

config = TrainingConfig(**config)


def test_models():
    test_clean()

    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("Total time", t2 - t1)
    # print("train=", result["train"])
    # print("validation=", result["validation"])
    print()
    print()
    print()
    test_clean()
    config.classification_threshold = 0.0
    # config.model.classification = True
    t1 = time.time()
    # result = train_dgl(config,model=None)
    t2 = time.time()
    print("Total time", t2 - t1)
    # print("train=", result["train"])
    # print("validation=", result["validation"])
    print()
    print()
    print()
    test_clean()


def test_alignn_train_regression():
    # Regression
    cmd = "rm -rf *train_data *test_data *val_data"
    os.system(cmd)
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/sample_data/")
    )
    config = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../examples/sample_data/config_example.json",
        )
    )
    tmp = loadjson(config)
    tmp["filename"] = "AA"
    dumpjson(data=tmp, filename=config)
    train_for_folder(
        local_rank=0, world_size=world_size, root_dir=root_dir, config_name=config
    )


def test_alignn_train_regression_multi_out():
    cmd = "rm -rf *train_data *test_data *val_data"
    os.system(cmd)
    # Regression multi-out
    root_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../examples/sample_data_multi_prop/"
        )
    )
    config = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../examples/sample_data/config_example.json",
        )
    )
    tmp = loadjson(config)
    tmp["filename"] = "BB"
    dumpjson(data=tmp, filename=config)
    train_for_folder(
        local_rank=0, world_size=world_size, root_dir=root_dir, config_name=config
    )


def test_alignn_train_classification():
    cmd = "rm -rf *train_data *test_data *val_data"
    os.system(cmd)
    # Classification
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/sample_data/")
    )
    config = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../examples/sample_data/config_example.json",
        )
    )
    tmp = loadjson(config)
    tmp["filename"] = "A"
    dumpjson(data=tmp, filename=config)
    train_for_folder(
        local_rank=0,
        world_size=world_size,
        root_dir=root_dir,
        config_name=config,
        classification_threshold=0.01,
    )


def test_alignn_train_ff():
    cmd = "rm -rf *train_data *test_data *val_data"
    os.system(cmd)
    # FF
    root_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../examples/sample_data_ff/")
    )
    config = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../examples/sample_data_ff/econfig_example_atomwise.json",
        )
    )
    tmp = loadjson(config)
    tmp["filename"] = "B"
    dumpjson(data=tmp, filename=config)
    train_for_folder(
        local_rank=0, world_size=world_size, root_dir=root_dir, config_name=config
    )
    cmd = "rm *.pt *.csv *.json *range"
    os.system(cmd)


def test_clean():
    cmd = "rm *.pt *.traj *.csv *.json *range"
    os.system(cmd)


test_clean()
# test_pretrained()
# test_models()
# test_alignn_train_ff()
# test_alignn_train_regression_multi_out()

# test_alignn_train_classification()
# test_alignn_train()
# test_minor_configs()
# test_runtime_training()
# test_alignn_train()
# test_calculator()
