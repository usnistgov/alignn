"""Helper function for high-throughput GNN trainings."""
import matplotlib.pyplot as plt

# import numpy as np
import time
from alignn.train import train_dgl

# from sklearn.metrics import mean_absolute_error
plt.switch_backend("agg")


def train_prop_model(
    prop="",
    dataset="dft_3d",
    write_predictions=True,
    name="alignn",
    save_dataloader=False,
    train_ratio=None,
    classification_threshold=None,
    val_ratio=None,
    test_ratio=None,
    learning_rate=0.001,
    batch_size=None,
    scheduler=None,
    n_epochs=None,
    id_tag=None,
    num_workers=None,
    weight_decay=None,
    alignn_layers=None,
    gcn_layers=None,
    edge_input_features=None,
    triplet_input_features=None,
    embedding_features=None,
    hidden_features=None,
    output_features=None,
    random_seed=None,
    n_early_stopping=None,
    cutoff=None,
    max_neighbors=None,
):
    """Train models for a dataset and a property."""
    if scheduler is None:
        scheduler = "onecycle"
    if batch_size is None:
        batch_size = 64
    if n_epochs is None:
        n_epochs = 300
    if num_workers is None:
        num_workers = 0
    config = {
        "dataset": dataset,
        "target": prop,
        "epochs": n_epochs,  # 00,#00,
        "batch_size": batch_size,  # 0,
        "weight_decay": 1e-05,
        "learning_rate": learning_rate,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": scheduler,
        "save_dataloader": save_dataloader,
        "pin_memory": False,
        "write_predictions": write_predictions,
        "num_workers": num_workers,
        "classification_threshold": classification_threshold,
        "model": {"name": name, },
    }
    if n_early_stopping is not None:
        config["n_early_stopping"] = n_early_stopping
    if cutoff is not None:
        config["cutoff"] = cutoff
    if max_neighbors is not None:
        config["max_neighbors"] = max_neighbors
    if weight_decay is not None:
        config["weight_decay"] = weight_decay
    if alignn_layers is not None:
        config["model"]["alignn_layers"] = alignn_layers
    if gcn_layers is not None:
        config["model"]["gcn_layers"] = gcn_layers
    if edge_input_features is not None:
        config["model"]["edge_input_features"] = edge_input_features
    if hidden_features is not None:
        config["model"]["hidden_features"] = hidden_features
    if embedding_features is not None:
        config["model"]["embedding_features"] = embedding_features
    if output_features is not None:
        config["model"]["output_features"] = output_features
    if random_seed is not None:
        config["random_seed"] = random_seed
    # if model_name is not None:
    #    config['model']['name']=model_name

    if id_tag is not None:
        config["id_tag"] = id_tag
    if train_ratio is not None:
        config["train_ratio"] = train_ratio
        if val_ratio is None:
            raise ValueError("Enter val_ratio.")

        if test_ratio is None:
            raise ValueError("Enter test_ratio.")
        config["val_ratio"] = val_ratio
        config["test_ratio"] = test_ratio
    if dataset == "jv_3d":
        # config["save_dataloader"]=True
        config["num_workers"] = 4
        config["pin_memory"] = False
        # config["learning_rate"] = 0.001
        # config["epochs"] = 300

    if dataset == "mp_3d_2020":
        config["id_tag"] = "id"
        config["num_workers"] = 0
    if dataset == "megnet2":
        config["id_tag"] = "id"
        config["num_workers"] = 0
    if dataset == "megnet":
        config["id_tag"] = "id"
        if prop == "e_form" or prop == "gap pbe":
            config["n_train"] = 60000
            config["n_val"] = 5000
            config["n_test"] = 4239
            # config["learning_rate"] = 0.01
            # config["epochs"] = 300
            config["num_workers"] = 4
    if dataset == "oqmd_3d_no_cfid":
        config["id_tag"] = "_oqmd_entry_id"
        config["num_workers"] = 0
    if dataset == "hmof" and prop == "co2_absp":
        config["model"]["output_features"] = 5
    if dataset == "edos_pdos":
        if prop == "edos_up":
            config["model"]["output_features"] = 300
        elif prop == "pdos_elast":
            config["model"]["output_features"] = 200
        else:
            raise ValueError("Target not available.")
    if dataset == "qm9_std_jctc":
        config["id_tag"] = "id"
        config["n_train"] = 110000
        config["n_val"] = 10000
        config["n_test"] = 10829

        # config["batch_size"] = 64
        config["cutoff"] = 5.0
        config["standard_scalar_and_pca"] = False

    if dataset == "qm9_dgl":
        config["id_tag"] = "id"
        config["n_train"] = 110000
        config["n_val"] = 10000
        config["n_test"] = 10831
        config["standard_scalar_and_pca"] = False
        config["batch_size"] = 64
        config["cutoff"] = 5.0
        if config["target"] == "all":
            config["model"]["output_features"] = 12

        # config["max_neighbors"] = 9

    if dataset == "hpov":
        config["id_tag"] = "id"
    if dataset == "qm9":
        config["id_tag"] = "id"
        config["n_train"] = 110000
        config["n_val"] = 10000
        config["n_test"] = 13885
        config["batch_size"] = batch_size
        config["cutoff"] = 5.0
        config["max_neighbors"] = 9
        # config['atom_features']='atomic_number'
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
