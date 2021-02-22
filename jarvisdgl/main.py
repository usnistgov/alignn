"""Module to train DGL graph for Atoms."""

# !pip install dgl==0.4.3 jarvis-tools==2021.2.3
from torch.utils.data import DataLoader
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms
from jarvis.core.atoms import get_supercell_dims
import dgl
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from dgl.nn import GraphConv
from dgl.nn import AvgPooling
import numpy as np
import torch.utils.data
import torch
from sklearn.model_selection import train_test_split
import networkx as nx

config = {
    "chem_type": "basic",
    "verbose": True,
    "use_gp": True,
    "use_dask": False,
    "zero_diag": True,
    "dropout_rate": 0,
    "lengthscale": 1.0,
    "variance": 1.0,
    "bn_size": 2,
    "gcn_layers": 16,
    "growth_rate": 24,
    "fc_layers": 8,
    "fc_features": 16,
    "batch_size": 16,
    "base_lr": 1.0e-2,
    "chem_type": "basic",
    "node_atomwise_angle_dist": True,
    "node_atomwise_rdf": True,
    "min_cell_size": 5,
    "get_prim": False,
    "base_wd": 1.0e-3,
    "n_epochs": 60,
    "test_size": 0.2,
}


def dgl_crystal(
    atoms: Atoms, primitive: bool = False, cutoff: float = 8, enforce_c_size: float = 5
):
    """Get DGLGraph from atoms."""

    if primitive:
        atoms = atoms.get_primitive_atoms

    dim = get_supercell_dims(atoms=atoms, enforce_c_size=enforce_c_size)
    atoms = atoms.make_supercell(dim)
    dist = atoms.raw_distance_matrix
    dist[dist > cutoff] = 0

    D = nx.DiGraph(dist)

    g = dgl.from_networkx(D, edge_attrs=["weight"])
    g.edata["bondlength"] = g.edata["weight"]
    # del g.edata['weight']
    g.ndata["atomic_number"] = torch.tensor(atoms.atomic_numbers, dtype=torch.int8)
    return g


class SimpleGCN(nn.Module):
    """Module for simple GCN."""

    def __init__(self, in_features=1, conv_layers=2, width=32):
        """Initialize class with number og input features, conv layers."""
        super().__init__()
        self.layer1 = GraphConv(in_features, width)
        self.layer2 = GraphConv(width, 1)
        self.readout = AvgPooling()

    def forward(self, g):
        """Provide forward function."""
        features = g.ndata["atomic_number"]
        features = features.view(-1, features.size()[-1]).T
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        x = self.readout(g, x)
        return torch.squeeze(x)


class StructureDataset(torch.utils.data.Dataset):
    """Module for generating DGL dataset."""

    def __init__(self, structures, targets):
        """Initialize the class."""
        self.graphs = []
        self.labels = []
        for i, j in zip(structures, targets):
            if len(self.labels) < 65:
                a = Atoms.from_dict(i)
                # graph=Graph.from_atoms(a).to_networkx()
                # g1=dgl.from_networkx(graph)
                # atom_features = "atomic_number"

                g2 = dgl_crystal(a)

                self.graphs.append(g2)
                self.labels.append(j)

        self.labels = torch.tensor(self.labels)

    def __len__(self):
        """Get length."""
        return self.labels.shape[0]

    def __getitem__(self, idx):
        """Get item."""
        return self.graphs[idx], self.labels[idx]

    @staticmethod
    def collate(samples):
        """Provide input `samples` is a pair of lists (graphs, labels)."""
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)


def train_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch=0,
):
    """Train model."""
    train_loss = []
    model.train()
    for g, target in train_loader:
        output = model(g)
        loss = criterion(output, target)
        # print (loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss)
    return train_loss


def evaluate(
    test_loader,
    model,
    criterion,
    optimizer,
    epoch=0,
):
    """Evaluate model."""
    test_loss = []
    model.eval()
    for g, target in test_loader:
        output = model(g)
        loss = criterion(output, target)
        # print (loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        test_loss.append(loss)
    return test_loss


def train_property_model(prop="optb88vdw_bandgap", dataset_name="dft_3d"):
    """Train property model."""
    dataset = jdata(dataset_name)
    structures = []
    targets = []
    for i in dataset:
        if i[prop] != "na":
            structures.append(i["atoms"])
            targets.append(i[prop])

    X_train, X_test, y_train, y_test = train_test_split(
        structures, targets, test_size=0.33, random_state=int(37)
    )

    train_data = StructureDataset(X_train, y_train)
    val_data = StructureDataset(X_test, y_test)

    # use a regular pytorch dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=True,
        collate_fn=train_data.collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=True,
        collate_fn=val_data.collate,
        drop_last=True,
    )

    model = SimpleGCN()
    criterion = torch.nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # hist = {"val_loss": [], "train_loss": []}
    t_loss = []
    # v_loss = []
    for epoch_idx in range(config["n_epochs"]):
        train_loss = train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch=epoch_idx,
        )
        val_loss = evaluate(val_loader, model, criterion, optimizer, epoch=epoch_idx)
        # print (train_loss, type(train_loss),val_loss,type(val_loss))
        t_loss.append(np.mean(np.array([j.data for j in train_loss])))
        val_loss = [j.data for j in val_loss]
        print("t_loss,v_loss", epoch_idx, (train_loss[-1]), val_loss[-1])


if __name__ == "__main__":
    train_property_model()
