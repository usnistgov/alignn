# conda activate megnet
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import numpy as np
from pymatgen.core.structure import Structure
import csv
import os
import random
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings(
    "ignore",
    ".*do not.*",
)


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
):
    """Get train, val, test IDs."""
    if (
        train_ratio is None
        and val_ratio is not None
        and test_ratio is not None
    ):
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print("Using rest of the dataset except the test and val sets.")
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
    # indices = list(range(total_size))
    if n_train is None:
        n_train = int(train_ratio * total_size)
    if n_test is None:
        n_test = int(test_ratio * total_size)
    if n_val is None:
        n_val = int(val_ratio * total_size)
    ids = list(np.arange(total_size))
    if not keep_data_order:
        random.seed(split_seed)
        random.shuffle(ids)
    if n_train + n_val + n_test > total_size:
        raise ValueError(
            "Check total number of samples.",
            n_train + n_val + n_test,
            ">",
            total_size,
        )

    # shuffle consistently with https://github.com/txie-93/cgcnn/data.py
    # i.e. shuffle the index in place with standard library random.shuffle
    # first obtain only valid indices

    # test_size = round(N * 0.2)

    # full train/val test split
    id_train = ids[:n_train]
    id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
    id_test = ids[-n_test:]
    return id_train, id_val, id_test


nfeat_bond = 100
r_cutoff = 5
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = 0.5
graph_converter = CrystalGraph(cutoff=r_cutoff)

# Model training
# Here, `structures` is a list of pymatgen Structure objects.
# `targets` is a corresponding list of properties.
root_dir = "alignn/alignn/examples/sample_data"
root_dir = (
    "CGCNN/formation_energy_peratom/cgcnn/"
    + "data_dir_dft_3d_formation_energy_peratom"
)
root_dir = (
    "/home/knc6/Software/version/alignn/alignn/tests/ALL_DATASETS/CGCNN/qm9_U0"
)

root_dir = "ALL_DATASETS/CGCNN/qm9_U0/cgcnn/data_dir_qm9_std_jctc_U0"
idp = os.path.join(root_dir, "id_prop.csv")

structures = []
targets = []

with open(idp, "r") as f:
    reader = csv.reader(f)
    data = [row for row in reader]


id_train, id_val, id_test = get_id_train_val_test(
    total_size=len(data),
)

print("Loading structures.")
for i in id_train:
    fname = os.path.join(root_dir, data[i][0]) + ".cif"  # check path name
    s = Structure.from_file(fname)
    t = float(data[i][1])
    structures.append(s)
    targets.append(t)
print(np.array(targets))
# new_structure=Structure.from_file('POSCAR')
# structures=[new_structure]
# targets=[1]
model = MEGNetModel(
    graph_converter=graph_converter,
    centers=gaussian_centers,
    width=gaussian_width,
)
graphs_valid = []
targets_valid = []
structures_invalid = []
for s, p in zip(structures, targets):
    try:
        graph = model.graph_converter.convert(s)
        graphs_valid.append(graph)
        targets_valid.append(p)
    except Exception:
        structures_invalid.append(s)

print("Training model.")
# train the model using valid graphs and targets
model.train_from_graphs(graphs_valid, targets_valid, epochs=300)
# model.train(structures, targets, epochs=10)
# print ('Testing model.')
new_structures = []
new_targets = []
preds = []
for i in id_test:
    fname = os.path.join(root_dir, data[i][0]) + ".cif"
    s = Structure.from_file(fname)
    t = float(data[i][1])
    new_structures.append(s)
    new_targets.append(t)
    # Predict the property of a new structure
    pred_target = model.predict_structure(s)
    preds.append(pred_target)
print(
    "Test MAE:",
    mean_absolute_error(np.array(new_targets), np.array(preds)),
)
# print (pred_target,type(pred_target[0]))
