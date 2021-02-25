"""Jarvis-dgl data loaders and DGLGraph utilities."""
from typing import List, Tuple

import dgl
import numpy as np
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.db.figshare import data as jdata
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


def prepare_dgl_batch(batch, device=None, non_blocking=False):
    """Send batched dgl graph to device."""
    g, t = batch
    batch = (g.to(device), t.to(device))

    return batch


def dgl_crystal(
    atoms: Atoms,
    primitive: bool = False,
    cutoff: float = 8,
    enforce_c_size: float = 5,
    atom_features="atomic_number",
):
    """Get DGLGraph from atoms, go through jarvis.core.graph."""
    feature_sets = ("atomic_number", "basic", "cfid")
    if atom_features not in feature_sets:
        raise NotImplementedError(
            f"atom features must be one of {feature_sets}"
        )

    jgraph = Graph.from_atoms(
        atoms,
        features=atom_features,
        get_prim=primitive,
        max_cut=cutoff,
        enforce_c_size=enforce_c_size,
    )

    # weight is currently
    #  `adj = variance * np.exp(-bond_distance / lengthscale)`
    g = dgl.from_networkx(jgraph.to_networkx(), edge_attrs=["weight"])
    g.edata["bondlength"] = g.edata["weight"].type(torch.FloatTensor)
    del g.edata["weight"]

    g.ndata["atom_features"] = torch.tensor(jgraph.node_attributes).type(
        torch.FloatTensor
    )

    return g


class Standardize(torch.nn.Module):
    """Standardize atom_features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Register featurewise mean and standard deviation."""
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, g: dgl.DGLGraph):
        """Apply standardization to atom_features."""
        g = g.local_var()
        h = g.ndata.pop("atom_features")
        g.ndata["atom_features"] = (h - self.mean) / self.std
        return g


class StructureDataset(torch.utils.data.Dataset):
    """Dataset of crystal DGLGraphs."""

    def __init__(
        self,
        structures,
        targets,
        maxrows=np.inf,
        atom_features="atomic_number",
        transform=None,
    ):
        """Initialize the class."""
        self.graphs = []
        self.labels = []
        for idx, (structure, target) in enumerate(zip(structures, targets)):
            if idx >= maxrows:
                break

            a = Atoms.from_dict(structure)
            g = dgl_crystal(a, atom_features=atom_features)

            self.graphs.append(g)
            self.labels.append(target)

        self.labels = torch.tensor(self.labels)
        self.transform = transform

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get StructureDataset sample."""
        g = self.graphs[idx]
        label = self.labels[idx]

        if self.transform:
            g = self.transform(g)

        return g, label

    def setup_standardizer(self):
        """Atom-wise feature standardization transform."""
        x = torch.cat([g.ndata["atom_features"] for g in self.graphs])
        self.atom_feature_mean = x.mean(0)
        self.atom_feature_std = x.std(0)

        self.transform = Standardize(
            self.atom_feature_mean, self.atom_feature_std
        )

    @staticmethod
    def collate(samples: List[Tuple[dgl.DGLGraph, torch.Tensor]]):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)


def get_train_val_loaders(
    dataset: str = "dft_3d",
    target: str = "formation_energy_peratom",
    atom_features: str = "atomic_number",
    n_train: int = 32,
    n_val: int = 32,
    batch_size: int = 8,
    standardize: bool = False,
):
    """Help function to set up Jarvis train and val dataloaders."""
    d = jdata(dataset)

    structures, targets = [], []
    for row in d:
        if row[target] != "na":
            structures.append(row["atoms"])
            targets.append(row[target])

    X_train, X_test, y_train, y_test = train_test_split(
        structures, targets, test_size=0.33, random_state=int(37)
    )

    train_data = StructureDataset(
        X_train, y_train, atom_features=atom_features, maxrows=n_train
    )
    if standardize:
        train_data.setup_standardizer()

    val_data = StructureDataset(
        X_test,
        y_test,
        atom_features=atom_features,
        maxrows=n_val,
        transform=train_data.transform,
    )

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_data.collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=val_data.collate,
        drop_last=True,
    )

    return train_loader, val_loader
