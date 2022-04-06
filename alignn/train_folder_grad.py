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
    file_format="poscar",
    output_dir=None,
):
    """Train for a folder."""
    # config_dat=os.path.join(root_dir,config_name)
    dat = loadjson(os.path.join(root_dir, "id_prop.json"))
    # id_prop_dat = os.path.join(root_dir, "id_prop.csv")
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
    # with open(id_prop_dat, "r") as f:
    #    reader = csv.reader(f)
    #    data = [row for row in reader]
    # dict_keys(['jids', 'tags', 'energies', 'atoms', 'forces', 'stresses'])
    mean_energy = np.array(dat["energies"]).mean()
    mean_force = np.array(dat["forces"]).mean()
    print("mean_energy", mean_energy)
    print("mean_force", mean_force)
    dataset = []
    for ii, jj, kk, ff in zip(
        dat["ids"], dat["atoms"], dat["energies"], dat["forces"]
    ):
        info = {}
        info["target"] = kk - mean_energy
        info["atoms"] = jj
        info["atomwise_target"] = ff - mean_force
        info["atomwise_grad"] = ff - mean_force
        info["jid"] = ii
        dataset.append(info)
    # Assuming all the atomwise_target data are of same length
    # atomwise_grad in of size 3 each for each node
    """
    if "atomwise_target" in info:
        print ("atomwise_target size",len(info["atomwise_target"][0]))
        config.model.atomwise_output_features=len(info["atomwise_target"])
    """

    n_outputs = []
    multioutput = False
    lists_length_equal = True
    line_graph = False
    alignn_models = {
        "alignn",
        "dense_alignn",
        "alignn_cgcnn",
        "alignn_layernorm",
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
        target_atomwise="atomwise_target",
        target_grad="atomwise_grad",
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
