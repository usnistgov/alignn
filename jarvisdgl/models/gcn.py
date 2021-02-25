"""A baseline graph convolution network dgl implementation."""
import torch
from dgl.nn import AvgPooling, GraphConv
from torch import nn
from torch.nn import functional as F


class SimpleGCN(nn.Module):
    """Module for simple GCN."""

    def __init__(self, in_features=1, conv_layers=2, width=32):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.layer1 = GraphConv(in_features, width)
        self.layer2 = GraphConv(width, 1)
        self.readout = AvgPooling()

    def forward(self, g):
        """Baseline SimpleGCN : start with `atom_features`."""
        features = g.ndata["atom_features"]
        features = features.view(-1, features.size()[-1]).T
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        x = self.readout(g, x)
        return torch.squeeze(x)
