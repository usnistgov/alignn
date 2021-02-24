"""Module to train DGL graph for Atoms."""

# !pip install dgl==0.4.3 jarvis-tools==2021.2.3
import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from dgl.nn import AvgPooling, GraphConv
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.db.figshare import data as jdata
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader

from jarvisdgl.models import CGCNN

torch.set_default_dtype(torch.float32)

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
    atoms: Atoms,
    primitive: bool = False,
    cutoff: float = 8,
    enforce_c_size: float = 5,
):
    """Get DGLGraph from atoms, go through jarvis.core.graph."""
    jgraph = Graph.from_atoms(
        atoms,
        features="basic",
        get_prim=primitive,
        max_cut=cutoff,
        enforce_c_size=enforce_c_size,
    )

    # weight is currently
    #  `adj = variance * np.exp(-bond_distance / lengthscale)`
    g = dgl.from_networkx(jgraph.to_networkx(), edge_attrs=["weight"])
    g.edata["bondlength"] = g.edata["weight"].type(torch.FloatTensor)
    del g.edata["weight"]

    g.ndata["atom_features"] = torch.tensor(
        jgraph.node_attributes, dtype=torch.float32
    )

    return g


class SimpleGCN(nn.Module):
    """Module for simple GCN."""

    def __init__(self, in_features=1, conv_layers=2, width=32):
        """Initialize class with number of input features, conv layers."""
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

    def __init__(self, structures, targets, maxrows=np.inf):
        """Initialize the class."""
        self.graphs = []
        self.labels = []
        for idx, (i, j) in enumerate(zip(structures, targets)):
            if idx >= maxrows:
                break

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
    scheduler,
    epoch=0,
):
    """Train model."""
    train_loss = []
    model.train()
    for g, target in train_loader:
        output = model(g)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss.append(loss.item())
    return train_loss


def evaluate(
    test_loader,
    model,
    criterion,
    epoch=0,
):
    """Evaluate model."""
    model.eval()

    test_loss = []
    for g, target in test_loader:
        with torch.no_grad():
            output = model(g)
            loss = criterion(output, target)
            test_loss.append(loss)

    return test_loss


def train_property_model(
    prop="optb88vdw_bandgap",
    dataset_name="dft_3d",
    epochs=config.get("n_epochs", 10),
    maxrows=1024,
    batch_size=config.get("batch_size", 32),
):
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

    train_data = StructureDataset(X_train, y_train, maxrows=maxrows)
    val_data = StructureDataset(X_test, y_test, maxrows=maxrows)

    print(len(train_data))
    print(len(val_data))

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

    # model = SimpleGCN()
    model = CGCNN(atom_input_features=11, logscale=False)
    criterion = torch.nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-2,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    # hist = {"val_loss": [], "train_loss": []}
    t_loss = []
    # v_loss = []
    for epoch_idx in range(epochs):
        train_loss = train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch=epoch_idx,
        )
        val_loss = evaluate(val_loader, model, criterion, epoch=epoch_idx)
        # print (train_loss, type(train_loss),val_loss,type(val_loss))
        t_loss.append(np.mean(train_loss))
        val_loss = [j.data for j in val_loss]
        print("t_loss,v_loss", epoch_idx, (train_loss[-1]), val_loss[-1])


if __name__ == "__main__":
    train_property_model(prop="formation_energy_peratom")
