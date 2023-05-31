#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""
import os

# import numpy as np
import sys
from alignn.data import get_train_val_loaders
from alignn.train import train_dgl
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig

# from alignn.models.alignn import ALIGNN, ALIGNNConfig
import torch
import time

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


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
    default=True,
    help="Whether to randomly shuffle samples",
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
    "--target_key",
    default="total_energy",
    help="Name of the key for graph level data such as total_energy",
)

parser.add_argument(
    "--id_key",
    default="jid",
    help="Name of the key for graph level id such as id",
)

parser.add_argument(
    "--force_key",
    default="forces",
    help="Name of key for gradient level data such as forces, (Natoms x p)",
)

parser.add_argument(
    "--atomwise_key",
    default="forces",
    help="Name of key for atomwise level data: forces, charges (Natoms x p)",
)


parser.add_argument(
    "--stresswise_key",
    default="stresses",
    help="Name of the key for stress (3x3) level data such as forces",
)


parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)


parser.add_argument(
    "--restart_model_path",
    default=None,
    help="Checkpoint file path for model",
)


def train_for_folder(
    root_dir="examples/sample_data",
    config_name="config.json",
    keep_data_order=False,
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    id_key="jid",
    target_key="total_energy",
    atomwise_key="forces",
    gradwise_key="forces",
    stresswise_key="stresses",
    file_format="poscar",
    restart_model_path=None,
    # subtract_mean=False,
    # normalize_with_natoms=False,
    output_dir=None,
):
    """Train for a folder."""
    dat = loadjson(os.path.join(root_dir, "id_prop.json"))
    config_dict = loadjson(config_name)
    config = TrainingConfig(**config_dict)
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

    train_grad = False
    train_stress = False
    if config.model.gradwise_weight != 0:
        train_grad = True
    if config.model.stresswise_weight != 0:
        train_stress = True
    train_atom = False
    if config.model.atomwise_weight != 0:
        train_atom = True

    if config.model.atomwise_weight == 0:
        train_atom = False
    if config.model.gradwise_weight == 0:
        train_grad = False
    if config.model.stresswise_weight == 0:
        train_stress = False
    target_atomwise = None  # "atomwise_target"
    target_grad = None  # "atomwise_grad"
    target_stress = None  # "stresses"

    # mem = []
    # enp = []
    dataset = []
    for i in dat:
        info = {}
        info["target"] = i[target_key]
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
        info["jid"] = i[id_key]
        dataset.append(info)
    print("len dataset", len(dataset))
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

    model = None
    if restart_model_path is not None:
        print("Restarting the model training:", restart_model_path)
        if config.model.name == "alignn_atomwise":
            tmp = ALIGNNAtomWiseConfig(
                name="alignn_atomwise",
                output_features=config.model.output_features,
                alignn_layers=config.model.alignn_layers,
                atomwise_weight=config.model.atomwise_weight,
                stresswise_weight=config.model.stresswise_weight,
                graphwise_weight=config.model.graphwise_weight,
                gradwise_weight=config.model.gradwise_weight,
                gcn_layers=config.model.gcn_layers,
                atom_input_features=config.model.atom_input_features,
                edge_input_features=config.model.edge_input_features,
                triplet_input_features=config.model.triplet_input_features,
                embedding_features=config.model.embedding_features,
            )
            print("Rest config", tmp)
            # for i,j in config_dict['model'].items():
            #    print ('i',i)
            #    tmp.i=j
            # print ('tmp1',tmp)
            model = ALIGNNAtomWise(tmp)  # config.model)
            # model = ALIGNNAtomWise(ALIGNNAtomWiseConfig(
            #    name="alignn_atomwise",
            #    output_features=1,
            #    graphwise_weight=1,
            #    alignn_layers=4,
            #    gradwise_weight=10,
            #    stresswise_weight=0.01,
            #    atomwise_weight=0,
            #      )
            #    )
            print("model", model)
            model.load_state_dict(
                torch.load(restart_model_path, map_location=device)
            )
            model.to(device)

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
    # print('config.neighbor_strategy',config.neighbor_strategy)
    # import sys
    # sys.exit()
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

    t1 = time.time()
    train_dgl(
        config,
        model=model,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
    )
    t2 = time.time()
    print("Time taken (s)", t2 - t1)

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
        target_key=(args.target_key),
        id_key=(args.id_key),
        atomwise_key=(args.atomwise_key),
        gradwise_key=(args.force_key),
        stresswise_key=(args.stresswise_key),
        restart_model_path=(args.restart_model_path),
        # subtract_mean=(args.subtract_mean),
        # normalize_with_natoms=(args.normalize_with_natoms),
        file_format=(args.file_format),
    )
