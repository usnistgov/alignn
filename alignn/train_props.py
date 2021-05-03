"""Helper function for high-throughput GNN trainings."""
import matplotlib.pyplot as plt

# import numpy as np
import time
from alignn.train import train_dgl

# from sklearn.metrics import mean_absolute_error
plt.switch_backend("agg")

jv_3d_props = [
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "mbj_bandgap",
    "slme",
    "magmom_oszicar",
    "spillage",
    "kpoint_length_unit",
    "encut",
    "optb88vdw_total_energy",
    "epsx",
    "epsy",
    "epsz",
    "mepsx",
    "mepsy",
    "mepsz",
    "max_ir_mode",
    "min_ir_mode",
    "n-Seebeck",
    "p-Seebeck",
    "n-powerfact",
    "p-powerfact",
    "ncond",
    "pcond",
    "nkappa",
    "pkappa",
    "ehull",
    "exfoliation_energy",
    "dfpt_piezo_max_dielectric",
    "dfpt_piezo_max_eij",
    "dfpt_piezo_max_dij",
]


def train_prop_model(
    prop="", dataset="dft_3d", write_predictions=True, name="alignn"
):
    """Train models for a dataset and a property."""
    config = {
        "dataset": dataset,
        "target": prop,
        "epochs": 250,  # 00,#00,
        "batch_size": 128,  # 0,
        "weight_decay": 1e-05,
        "learning_rate": 0.01,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "write_predictions": False,
        "num_workers": 0,
        "model": {
            "name": name,
        },
    }
    if dataset == "megnet":
        config["id_tag"] = "id"
        if prop == "e_form" or prop == "gap pbe":
            config["n_train"] = 60000
            config["n_val"] = 5000
            config["n_test"] = 4237
            config["batch_size"] = 64
    if dataset == "qm9":
        config["id_tag"] = "id"
        config["n_train"] = 110000
        config["n_val"] = 10000
        config["n_test"] = 13885
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("train=", result["train"])
    print("validation=", result["validation"])
    print("Toal time:", t2 - t1)
    print()
    print()
    print()
