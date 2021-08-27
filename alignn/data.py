"""Jarvis-dgl data loaders and DGLGraph utilities."""

import math
import os

# from sklearn.pipeline import Pipeline
import pickle as pk
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph, StructureDataset
from jarvis.db.figshare import data as jdata
from jarvis.db.jsonutils import dumpjson

# from sklearn.decomposition import PCA  # ,KernelPCA
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# use pandas progress_apply
tqdm.pandas()


QM9_TARGETS = [
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "U0",
    "U",
    "H",
    "G",
    "Cv",
]


def load_dataset(
    name: str = "dft_3d",
    target=None,
    limit: Optional[int] = None,
    classification_threshold: Optional[float] = None,
):
    """Load jarvis data."""
    d = jdata(name)
    data = []
    for i in d:
        if i[target] != "na" and not math.isnan(i[target]):
            if classification_threshold is not None:
                if i[target] <= classification_threshold:
                    i[target] = 0
                elif i[target] > classification_threshold:
                    i[target] = 1
                else:
                    raise ValueError(
                        "Check classification data type.",
                        i[target],
                        type(i[target]),
                    )
            data.append(i)
    d = data
    if limit is not None:
        d = d[:limit]
    d = pd.DataFrame(d)
    # d = d.replace("na", np.nan)
    return d


# np.mean(mean_absolute_deviation(x,axis=0))
def mean_absolute_deviation(data, axis=None):
    """Get Mean absolute deviation."""
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def load_graphs(
    df: pd.DataFrame,
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cache_dir: Optional[Path] = Path("data/graphs"),
    use_canonize: bool = False,
):
    """Construct crystal graphs.

    Load only atomic number node features
    and bond displacement vector edge features.

    Resulting graphs have scheme e.g.
    ```
    Graph(num_nodes=12, num_edges=156,
          ndata_schemes={'atom_features': Scheme(shape=(1,)}
          edata_schemes={'r': Scheme(shape=(3,)})
    ```
    """

    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=False,
            use_canonize=use_canonize,
        )

    n_samples, _ = df.shape

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cachefile = cache_dir / f"{name}-{n_samples}_{neighbor_strategy}.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs, labels = dgl.load_graphs(str(cachefile))
    else:
        graphs = df["atoms"].progress_apply(atoms_to_graph).values
        if cachefile is not None:
            dgl.save_graphs(str(cachefile), graphs.tolist())

    return graphs


def get_id_train_val_test(
    total_size=1000,
    split_seed=123,
    cv_seed=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    n_train=None,
    n_test=None,
    n_val=None,
    keep_data_order=False,
    shuffle_train_val=False,
):
    """Get train, val, test array indices."""
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

    if shuffle_train_val:
        # for cross-validation splitting
        # merge train and val ids, and re-shuffle
        cv_ids = np.r_[id_train, id_val]
        random.seed(cv_seed)
        random.shuffle(cv_ids)
        id_train = cv_ids[:n_train]
        id_val = cv_ids[n_train:]

    return id_train, id_val, id_test


def get_torch_dataset(
    dataset: pd.DataFrame,
    graphs: Sequence[dgl.DGLGraph],
    atom_features: str = "cgcnn",
    target: str = "formation_energy_peratom",
    id_tag: str = "jid",
    line_graph: bool = True,
    classification=False,
    output_dir=".",
    tmp_name="dataset",
):
    """Get Torch Dataset."""
    df = pd.DataFrame(dataset)
    # print("df", df)
    vals = df[target].values
    print("data range", np.max(vals), np.min(vals))
    f = open(os.path.join(output_dir, tmp_name + "_data_range"), "w")
    line = "Max=" + str(np.max(vals)) + "\n"
    f.write(line)
    line = "Min=" + str(np.min(vals)) + "\n"
    f.write(line)
    f.close()

    data = StructureDataset(
        df,
        graphs,
        target=target,
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=id_tag,
        classification=classification,
    )
    return data


def load_cached_dataloaders(basename, pin_memory=False, workers=0):
    """Load cached pytorch dataloaders and set options that may change."""
    train_sample = Path(f"{basename}_train.data")
    val_sample = Path(f"{basename}_val.data")
    test_sample = Path(f"{basename}_test.data")

    if train_sample.exists() and val_sample.exists() and test_sample.exists():
        print("Loading from saved file...")
        print("Make sure all the DataLoader params are same.")
        print("This module is made for debugging only.")
        train_loader = torch.load(train_sample)
        val_loader = torch.load(val_sample)
        test_loader = torch.load(test_sample)

        train_loader.pin_memory = pin_memory
        val_loader.pin_memory = pin_memory
        test_loader.pin_memory = pin_memory

        train_loader.num_workers = workers
        val_loader.num_workers = workers
        test_loader.num_workers = workers

        return (
            train_loader,
            val_loader,
            test_loader,
            train_loader.dataset.prepare_batch,
        )


def get_train_val_loaders(
    dataset: str = "dft_3d",
    dataset_array=[],
    target: str = "formation_energy_peratom",
    atom_features: str = "cgcnn",
    neighbor_strategy: str = "k-nearest",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    shuffle_train_val=False,
    batch_size: int = 5,
    standardize: bool = False,
    line_graph: bool = True,
    split_seed: int = 123,
    workers: int = 0,
    pin_memory: bool = True,
    save_dataloader: bool = False,
    filename: str = "sample",
    id_tag: str = "jid",
    use_canonize: bool = False,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    classification_threshold: Optional[float] = None,
    target_multiplication_factor: Optional[float] = None,
    standard_scalar_and_pca=False,
    keep_data_order=False,
    output_features=1,
    output_dir=None,
    cache_dir=None,
):
    """Help function to set up JARVIS train and val dataloaders.

    shuffle_train_val: True for cross-validation; randomizes train/val split
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if save_dataloader:
        dataloaders = load_cached_dataloaders(
            filename, pin_memory=pin_memory, workers=workers
        )
        if dataloaders is not None:
            return dataloaders

    if not dataset_array:
        try:
            cachefile = Path(cache_dir) / f"{dataset}.pkl"

            if cachefile.exists():
                d = pd.read_pickle(cachefile)
            else:
                d = pd.DataFrame(jdata(dataset))
                d = d.replace("na", np.nan)
                d.to_pickle(cachefile)

        except (TypeError, FileNotFoundError):
            # cache_dir not set -> TypeError
            # cachefile not writeable -> FileNotFoundError
            pass
    else:
        d = pd.DataFrame(dataset_array)
        d = d.replace("na", np.nan)

    # load graphs with just atomic number attributes
    # load atom feature vectors at StructureDataset construction
    graph_cache = None
    if cache_dir is not None:
        graph_cache = Path(cache_dir) / "graphs"

    graphs = load_graphs(
        d,
        name=dataset,
        neighbor_strategy=neighbor_strategy,
        use_canonize=use_canonize,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        cache_dir=graph_cache,
    )

    if dataset == "qm9_dgl" and target == "all":
        target = QM9_TARGETS
    else:
        d = d[~d[target].isna()]

    # mutate target values in place
    if target_multiplication_factor:
        d[target] *= target_multiplication_factor

    if classification_threshold is not None:
        print(f"Using {classification_threshold} for classifying {target}")
        d[target] = d[target] > classification_threshold

    # id_test = ids[-test_size:]
    # if standardize:
    #    data.setup_standardizer(id_train)
    id_train, id_val, id_test = get_id_train_val_test(
        total_size=len(d),
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        keep_data_order=keep_data_order,
        shuffle_train_val=shuffle_train_val,
    )

    # save DFT ids
    ids_train_val_test = {
        "id_train": d[id_tag].iloc[id_train].tolist(),
        "id_val": d[id_tag].iloc[id_val].tolist(),
        "id_test": d[id_tag].iloc[id_test].tolist(),
    }
    dumpjson(
        data=ids_train_val_test,
        filename=os.path.join(output_dir, "ids_train_val_test.json"),
    )

    # load graphs and dataframe separately,
    # then just slice them both the same way.

    if standard_scalar_and_pca:
        sc = StandardScaler()

        Y_train = d[target].iloc[id_train]

        if Y_train.ndim == 1:
            sc.fit(Y_train.values.reshape(-1, 1))
            Y_std = sc.transform(d[target].reshape(-1, 1))
            d[target] = Y_std.squeeze()
        else:
            sc.fit(Y_train)
            d[target] = sc.transform(d[target])

        print(f"Mean: {sc.mean_}")
        print(f"Variance: {sc.var_}")

        pk.dump(sc, open(os.path.join(output_dir, "sc.pkl"), "wb"))

        # pc = PCA(n_components=output_features)
        # pipe = Pipeline(
        #    [
        #        ("scale", StandardScaler()),
        #        ("reduce_dims", PCA(n_components=output_features)),
        #    ]
        # )

        # pc = PCA(n_components=10)
        # pc.fit(y_data)
        # pk.dump(pc, open("pca.pkl", "wb"))

    if classification_threshold is None:

        try:
            mad = mean_absolute_deviation(d[target])
        except Exception as exp:
            print("Cannot write mad", exp)
            mad = None

        print("MAX val:", d[target].max())
        print("MIN val:", d[target].min())
        print("MAD:", mad)

        with open(os.path.join(output_dir, "mad"), "w") as f:
            print(f"MAX val: {d[target].max()}", file=f)
            print(f"MIN val: {d[target].min()}", file=f)
            print(f"MAD val: {mad}", file=f)

        # Random model precited value
        x_bar = d[target].iloc[id_train].mean()
        baseline_mae = mean_absolute_error(
            d[target].iloc[id_test], x_bar * np.ones(len(id_test))
        )
        print("Baseline MAE:", baseline_mae)

    classification = classification_threshold is not None

    # make sure to reset pandas index on subsets or
    # pytorch will fail when converting target columns to tensor
    train_data = StructureDataset(
        d.iloc[id_train].reset_index(),
        [graphs[id] for id in id_train],
        target=target,
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=id_tag,
        classification=classification,
    )

    val_data = StructureDataset(
        d.iloc[id_val].reset_index(),
        [graphs[id] for id in id_val],
        target=target,
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=id_tag,
        classification=classification,
    )

    test_data = StructureDataset(
        d.iloc[id_test].reset_index(),
        [graphs[id] for id in id_test],
        target=target,
        atom_features=atom_features,
        line_graph=line_graph,
        id_tag=id_tag,
        classification=classification,
    )

    collate_fn = train_data.collate
    if line_graph:
        collate_fn = train_data.collate_line_graph

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    if save_dataloader:
        torch.save(train_loader, f"{filename}_train.data")
        torch.save(val_loader, f"{filename}_val.data")
        torch.save(test_loader, f"{filename}_val.data")

    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))

    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.prepare_batch,
    )
