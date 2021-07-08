"""Module to train multi-properties from a folder."""
import csv
import os
import sys
from jarvis.core.atoms import Atoms

# from jarvis.core.graphs import Graph
# from alignn.models.alignn import ALIGNN
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
    help="Folder with id_multi_prop.csv, poscars/cifs and config*.json",
)
parser.add_argument(
    "--config_name",
    default="config_example_regrssion.json",
    help="Name of the config file",
)


def train_for_folder_multi_prop(
    root_dir="examples/sample_data_multi_prop", config_name="config.json"
):
    """Train for a multi-prop model from folder."""
    # config_dat=os.path.join(root_dir,config_name)
    id_prop_dat = os.path.join(root_dir, "id_multi_prop.csv")
    config = loadjson(config_name)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    dataset = []
    n_outputs = []
    for i in data:
        info = {}
        poscar_name = i[0]
        poscar_path = os.path.join(root_dir, poscar_name)
        atoms = Atoms.from_poscar(poscar_path)
        info["atoms"] = atoms.to_dict()
        info["jid"] = poscar_name
        info["target"] = [float(j) for j in i[1:]]
        n_outputs.append(info["target"])
        dataset.append(info)
    lists_are_length_equaled = False not in [
        len(i) == len(n_outputs[0]) for i in n_outputs
    ]
    if lists_are_length_equaled:
        config.model.output_features = len(n_outputs[0])
    else:
        # TODO: Pad with NaN
        raise ValueError("Make sure the outputs are of same size.")

    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        target=config.target,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
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
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
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
    train_for_folder_multi_prop(
        root_dir=args.root_dir, config_name=args.config_name
    )
