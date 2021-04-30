"""Jarvis-dgl data loaders and DGLGraph utilities."""

import random
from pathlib import Path
from typing import Optional

# from typing import Dict, List, Optional, Set, Tuple

import os
import torch
import dgl
import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph, StructureDataset
from jarvis.db.figshare import data as jdata
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# use pandas progress_apply
tqdm.pandas()


def load_dataset(
    name: str = "dft_3d", target=None, limit: Optional[int] = None
):
    """Load jarvis data."""
    d = jdata(name)
    data = []
    for i in d:
        if i[target] != "na":
            data.append(i)
    d = data
    if limit is not None:
        d = d[:limit]
    d = pd.DataFrame(d)
    d = d.replace("na", np.nan)
    return d


def load_graphs(
    df: pd.DataFrame,
    name: str = "dft_3d",
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8,
    max_neighbors: int = 12,
    cachedir: Optional[Path] = Path("data"),
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

    if cachedir is not None:
        cachefile = cachedir / f"{name}-{neighbor_strategy}.bin"
    else:
        cachefile = None

    if cachefile is not None and cachefile.is_file():
        graphs, labels = dgl.load_graphs(str(cachefile))
    else:
        graphs = df["atoms"].progress_apply(atoms_to_graph).values
        if cachefile is not None:
            dgl.save_graphs(str(cachefile), graphs.tolist())

    return graphs


def get_train_val_loaders(
    dataset: str = "dft_3d",
    target: str = "formation_energy_peratom",
    atom_features: str = "atomic_number",
    neighbor_strategy: str = "k-nearest",
    n_train=None,
    n_val=None,
    n_test=None,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    batch_size: int = 8,
    standardize: bool = False,
    line_graph: bool = False,
    split_seed: int = 123,
    workers: int = 0,
    pin_memory: bool = True,
    save_dataloader: bool = True,
    filename: str = "sample",
    id_tag: str = "jid",
    use_canonize: bool = False,
):
    """Help function to set up Jarvis train and val dataloaders."""
    train_sample = filename + "_train.data"
    val_sample = filename + "_val.data"
    test_sample = filename + "_test.data"

    if (
        os.path.exists(train_sample)
        and os.path.exists(val_sample)
        and os.path.exists(test_sample)
        and save_dataloader
    ):
        print("Loading from saved file...")
        print("Make sure all the DataLoader params are same.")
        print("This module is made for debugging only.")
        train_loader = torch.load(train_sample)
        val_loader = torch.load(val_sample)
        test_loader = torch.load(test_sample)
        # print("train", len(train_loader.dataset))
        # print("val", len(val_loader.dataset))
        # print("test", len(test_loader.dataset))
    else:
        df = load_dataset(name=dataset, target=target, limit=None)
        vals = df[target].values
        print("data range", np.max(vals), np.min(vals))
        graphs = load_graphs(
            df,
            name=dataset,
            neighbor_strategy=neighbor_strategy,
            use_canonize=use_canonize,
        )

        data = StructureDataset(
            df,
            graphs,
            target=target,
            atom_features=atom_features,
            line_graph=line_graph,
            id_tag=id_tag,
        )

        total_size = len(data.labels)
        if (
            train_ratio is None
            and val_ratio is not None
            and test_ratio is not None
        ):
            if train_ratio is None:
                assert val_ratio + test_ratio < 1
                train_ratio = 1 - val_ratio - test_ratio
                print(
                    "Using rest of the dataset except the test and val sets."
                )
            else:
                assert train_ratio + val_ratio + test_ratio <= 1
        # indices = list(range(total_size))
        if n_train is None:
            n_train = int(train_ratio * total_size)
        if n_test is None:
            n_test = int(test_ratio * total_size)
        if n_val is None:
            n_val = int(val_ratio * total_size)
        ids = np.arange(total_size)

        random.seed(split_seed)
        random.shuffle(ids)
        if n_train + n_val + n_test > total_size:
            raise ValueError("Check total number of samples.")

        # shuffle consistently with https://github.com/txie-93/cgcnn/data.py
        # i.e. shuffle the index in place with standard library random.shuffle
        # first obtain only valid indices

        # test_size = round(N * 0.2)

        # full train/val test split
        id_train = ids[:n_train]
        id_val = ids[-(n_val + n_test) : -n_test]  # noqa:E203
        id_test = ids[-n_test:]
        # id_test = ids[-test_size:]
        # if standardize:
        #    data.setup_standardizer(id_train)

        train_data = Subset(data, id_train)
        val_data = Subset(data, id_val)
        test_data = Subset(data, id_test)
        # print ('id_train',id_train)
        # train_data = data[id_train]
        # val_data = data[id_val]
        # test_data = data[id_test]
        # id_train = ids[:n_train]
        # id_val = ids[-(n_val + n_test) : -n_test]
        # # id_test = ids[:-n_test]

        collate_fn = data.collate
        if line_graph:
            collate_fn = data.collate_line_graph

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
            drop_last=True,
            num_workers=workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin_memory,
        )
        if save_dataloader:
            torch.save(train_loader, train_sample)
            torch.save(val_loader, val_sample)
            torch.save(test_loader, test_sample)
    print("n_train:", len(train_loader.dataset))
    print("n_val:", len(val_loader.dataset))
    print("n_test:", len(test_loader.dataset))
    return (
        train_loader,
        val_loader,
        test_loader,
        train_loader.dataset.dataset.prepare_batch,
    )
