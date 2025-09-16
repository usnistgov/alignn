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
        "name": "alignn_atomwise",
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


def test_pretrained():
    box = [[2.715, 2.715, 0], [0, 2.715, 2.715], [2.715, 0, 2.715]]
    coords = [[0, 0, 0], [0.25, 0.2, 0.25]]
    elements = ["Si", "Si"]
    Si = Atoms(lattice_mat=box, coords=coords, elements=elements)
    prd = get_prediction(atoms=Si)
    print(prd)
    cmd1 = "python alignn/pretrained.py"
    os.system(cmd1)
    get_multiple_predictions(atoms_array=[Si, Si])
    cmd1 = "rm *.json"
    os.system(cmd1)


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
            "../examples/sample_data_ff/config_example_atomwise.json",
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


def test_calculator():
    atoms = Atoms.from_dict(
        get_jid_data(dataset="dft_3d", jid="JVASP-32")["atoms"]
    )
    model_path = default_path()
    calc = AlignnAtomwiseCalculator(path=model_path)
    ase_atoms = atoms.ase_converter()
    ase_atoms.calc = calc
    energy = ase_atoms.get_potential_energy()
    forces = ase_atoms.get_forces()
    stress = ase_atoms.get_stress()
    print("energy", energy)
    print("max(forces.flatten()),3)", max(forces.flatten()))
    print("max(stress.flatten()),3)", max(stress.flatten()))
    # assert round(energy,3)==round(-60.954999923706055,3)
    # assert round(max(forces.flatten()),2)==round(0.08332983,2)
    # assert round(max(stress.flatten()),2)==round(0.002801671050217803,2)


def test_del_files():
    fnames = [
        "temp",
        "ase_nve.traj",
        "ase_nvt_langevin.traj",
        "ase_nvt_andersen.traj",
        "opt.log",
        "opt.traj",
        "alignn_ff.log",
        "dataset_data_range",
        "pred_data.json",
        "prediction_results_train_set.csv",
        "multi_out_predictions.json",
        "checkpoint_2.pt",
        "checkpoint_3.pt",
        "prediction_results_test_set.csv",
        "mad",
        "ids_train_val_test.json",
        "train_data_data_range",
        "val_data_data_range",
        "test_data_data_range",
        "config.json",
        "history_train.json",
        "current_model.pt",
        "best_model.pt",
        "Train_results.json",
        "Val_results.json",
        "history_val.json",
        "Test_results.json",
        "Test_results.json",
        "last_model.pt",
        "temp",
        "alignn/jv_formation_energy_peratom_alignn.zip",
        "alignn/jv_optb88vdw_total_energy_alignn.zip",
    ]
    for i in fnames:
        cmd = "rm -r " + i
        os.system(cmd)
    cmd = "rm -r *train_data *val_data *test_data"
    os.system(cmd)


test_clean()
#test_pretrained()
# test_models()
# test_alignn_train_ff()
# test_alignn_train_regression_multi_out()

# test_alignn_train_classification()
# test_alignn_train()
# test_minor_configs()
# test_runtime_training()
# test_alignn_train()
# test_calculator()
