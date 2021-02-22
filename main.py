#!pip install jarvis-tools dgl
from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
%matplotlib inline
import networkx as nx
import dgl
import torch
from dgl.nn.pytorch import GraphConv
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from dgl import DGLGraph
import torch
import random

dft_3d = data('dft_3d')
#dft_2d = data('dft_2d')
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)



class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)



class Subset(object):
    """Subset of a dataset at specified indices
    Code adapted from PyTorch.

    Parameters
    ----------
    dataset
        dataset[i] should return the ith datapoint
    indices : list
        List of datapoint indices to construct the subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, item):
        """Get the datapoint indexed by item

        Returns
        -------
        tuple
            datapoint
        """
        return self.dataset[self.indices[item]]

    def __len__(self):
        """Get subset size

        Returns
        -------
        int
            Number of datapoints in the subset
        """
        return len(self.indices)



x=[]
y=[]

for i in dft_3d:
 if len(x)<500:
  #g=dgl.from_networkx(Graph.from_atoms(atoms=Atoms.from_dict(i['atoms'])).to_networkx())
  gg=Graph.from_atoms(atoms=Atoms.from_dict(i['atoms']))
  dd=DGLGraph()

  dd.add_nodes(len(gg.nodes))
  dd.add_edges([i[0] for i in gg.edges],[i[1] for i in gg.edges])
  dd.edata['h']=torch.tensor(gg.edge_attributes)
  gap=i['optb88vdw_bandgap']
  if gap>0.05:
    label=1
  else:
    label=0
  x.append(dd)
  y.append(label)



# Create training and test sets.
trainset = Subset(x,y)#MiniGCDataset(320, 10, 20)
testset = Subset(x,y)#MiniGCDataset(80, 10, 20)
# Use PyTorch's DataLoader and the collate function
# defined before.
data_loader = DataLoader(trainset, batch_size=32, shuffle=True,collate_fn=collate)

# Create model
model = Classifier(1, 256, 2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(80):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)
