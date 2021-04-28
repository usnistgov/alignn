"""Jarvis-dgl data loaders and DGLGraph utilities."""

import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph, StructureDataset
from jarvis.db.figshare import data as jdata
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# use pandas progress_apply
tqdm.pandas()


def load_dataset(name: str = "dft_3d", limit: Optional[int] = None):
    """Load jarvis data."""
    d = jdata(name)
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
    n_train: int = 32,
    n_val: int = 32,
    n_test: int = 32,
    batch_size: int = 8,
    standardize: bool = False,
    line_graph: bool = False,
    split_seed: int = 123,
    workers: int = 4,
    pin_memory: bool = True,
):
    """Help function to set up Jarvis train and val dataloaders."""
    df = load_dataset(dataset, limit=None)
    graphs = load_graphs(
        df,
        name=dataset,
        neighbor_strategy=neighbor_strategy,
    )

    data = StructureDataset(
        df,
        graphs,
        target=target,
        atom_features=atom_features,
        line_graph=line_graph,
    )

    # shuffle consistently with https://github.com/txie-93/cgcnn/data.py
    # i.e. shuffle the index in place with standard library random.shuffle
    # first obtain only valid indices
    (ids,) = torch.where(torch.isfinite(data.labels))
    random.seed(split_seed)
    random.shuffle(ids)

    N = len(ids)
    train_size = round(N * 0.6)
    val_size = round(N * 0.2)
    # test_size = round(N * 0.2)

    # full train/val test split
    id_train = ids[:train_size]
    id_val = ids[train_size : train_size + val_size]  # noqa:E203
    # id_test = ids[-test_size:]

    id_train = id_train[:n_train]
    id_val = id_val[:n_val]

    if standardize:
        data.setup_standardizer(id_train)

    train_data = Subset(data, id_train)
    val_data = Subset(data, id_val)

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

    return train_loader, val_loader, data.prepare_batch
