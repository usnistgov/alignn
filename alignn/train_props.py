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
    prop="",
    dataset="dft_3d",
    write_predictions=True,
    name="alignn",
    save_dataloader=False,
    train_ratio=None,
    val_ratio=None,
    test_ratio=None,
    learning_rate=0.01,
    batch_size=None,
    scheduler=None,
):
    """Train models for a dataset and a property."""
    if scheduler is None:
        scheduler = "onecycle"
    if batch_size is None:
        batch_size = 128
    config = {
        "dataset": dataset,
        "target": prop,
        "epochs": 300,  # 00,#00,
        "batch_size": batch_size,  # 0,
        "weight_decay": 1e-05,
        "learning_rate": learning_rate,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": scheduler,
        "save_dataloader": save_dataloader,
        "pin_memory": False,
        "write_predictions": write_predictions,
        "num_workers": 0,
        "model": {
            "name": name,
        },
    }
    if train_ratio is not None:
        config["train_ratio"] = train_ratio
        if val_ratio is None:
            raise ValueError("Enter val_ratio.")

        if test_ratio is None:
            raise ValueError("Enter test_ratio.")
        config["val_ratio"] = val_ratio
        config["test_ratio"] = test_ratio
    if dataset == "jv_3d":
        if batch_size is None:
            batch_size = 128
        config["batch_size"] = batch_size
        # config["save_dataloader"]=True
        config["num_workers"] = 4
        config["pin_memory"] = False
        # config["learning_rate"] = 0.001
        # config["epochs"] = 300

    if dataset == "megnet":
        if batch_size is None:
            batch_size = 128
        config["id_tag"] = "id"
        if prop == "e_form" or prop == "gap pbe":
            config["n_train"] = 60000
            config["n_val"] = 5000
            config["n_test"] = 4237
            # config["learning_rate"] = 0.01
            config["batch_size"] = batch_size
            # config["epochs"] = 300
            config["num_workers"] = 4
    if dataset == "qm9":
        if batch_size is None:
            batch_size = 128
        config["id_tag"] = "id"
        config["n_train"] = 110000
        config["n_val"] = 10000
        config["n_test"] = 13885
        config["batch_size"] = batch_size
        # config["cutoff"] = 5.0
        # config["max_neighbors"] = 9
        if prop in ["homo", "lumo", "gap", "zpve", "U0", "U", "H", "G"]:
            config["target_multiplication_factor"] = 27.211386024367243
    t1 = time.time()
    result = train_dgl(config)
    t2 = time.time()
    print("train=", result["train"])
    print("validation=", result["validation"])
    print("Toal time:", t2 - t1)
    print()
    print()
    print()
