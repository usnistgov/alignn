#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""
import os
import numpy as np
import sys
from alignn.data import get_train_val_loaders
from alignn.train import train_dgl
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse

parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network"
)
parser.add_argument(
    "--root_dir",
    default="./",
    help="Folder with id_props.csv, structure files",
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--keep_data_order",
    default=False,
    help="Whether to randomly shuffle samples, True/False",
)

parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)

parser.add_argument(
    "--batch_size", default=None, help="Batch size, generally 64"
)

parser.add_argument(
    "--epochs", default=None, help="Number of epochs, generally 300"
)

parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)


def train_for_folder(
    root_dir="examples/sample_data",
    config_name="config.json",
    keep_data_order=False,
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    target_key="total_energy",
    atomwise_key="forces",
    gradwise_key="forces",
    stresswise_key="stresses",
    file_format="poscar",
    subtract_mean=True,
    normalize_with_natoms=False,
    output_dir=None,
):
    """Train for a folder."""
    dat = loadjson(os.path.join(root_dir, "id_prop.json"))
    config = loadjson(config_name)
    config = TrainingConfig(**config)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    config.keep_data_order = keep_data_order
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    train_grad = True
    train_stress = True
    train_atom = True
    target_atomwise = None  # "atomwise_target"
    target_grad = None  # "atomwise_grad"
    target_stress = None  # "stresses"

    if config.model.atomwise_weight == 0:
        train_atom = False
    if config.model.gradwise_weight == 0:
        train_grad = False
    if config.model.stresswise_weight == 0:
        train_stress = False
    mem = []
    enp = []
    if subtract_mean:
        for i in dat:
            i["energy_per_atom"] = i[
                target_key
            ]  # / len(i["atoms"]["elements"])
            mem.append(i)
            enp.append(i["energy_per_atom"])
        mean_energy = np.array(enp).mean()
        print("mean_energy", mean_energy)
    dataset = []
    for i in mem:
        info = {}
        if subtract_mean:
            info["target"] = i["energy_per_atom"] - mean_energy
        else:
            info["target"] = i[target_key]
        if normalize_with_natoms:
            info["target"] = info["target"] / len(i["atoms"]["elements"])

        if train_atom:
            target_atomwise = "atomwise_target"
            info["atomwise_target"] = i[atomwise_key]  # such as charges
        if train_grad:
            target_grad = "atomwise_grad"
            info["atomwise_grad"] = i[gradwise_key]  # - mean_force
        if train_stress:
            info["stresses"] = i[stresswise_key]  # - mean_force
            target_stress = "stresses"

        info["atoms"] = i["atoms"]
        info["jid"] = i["jid"]
        dataset.append(info)

    n_outputs = []
    multioutput = False
    lists_length_equal = True
    line_graph = False
    alignn_models = {
        "alignn",
        # "alignn_layernorm",
        "alignn_atomwise",
    }
    if config.model.name == "clgn":
        line_graph = True
    if config.model.name == "cgcnn":
        line_graph = True
    if config.model.name == "icgcnn":
        line_graph = True
    if config.model.name in alignn_models and config.model.alignn_layers > 0:
        line_graph = True

    if multioutput:
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]

    # print ('n_outputs',n_outputs[0])
    if multioutput and classification_threshold is not None:
        raise ValueError("Classification for multi-output not implemented.")
    if multioutput and lists_length_equal:
        config.model.output_features = len(n_outputs[0])
    else:
        # TODO: Pad with NaN
        if not lists_length_equal:
            raise ValueError("Make sure the outputs are of same size.")
        else:
            config.model.output_features = 1
    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        target="target",
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        line_graph=line_graph,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
    )

    train_dgl(
        config,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
    )

    # train_data = get_torch_dataset(


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    train_for_folder(
        root_dir=args.root_dir,
        config_name=args.config_name,
        keep_data_order=args.keep_data_order,
        classification_threshold=args.classification_threshold,
        output_dir=args.output_dir,
        batch_size=(args.batch_size),
        epochs=(args.epochs),
        file_format=(args.file_format),
    )
