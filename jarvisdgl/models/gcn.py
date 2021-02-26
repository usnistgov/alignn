"""A baseline graph convolution network dgl implementation."""
import dgl
import torch
from dgl.nn import AvgPooling, GraphConv
from torch import nn
from torch.nn import functional as F


class SimpleGCN(nn.Module):
    """Module for simple GCN."""

    def __init__(self, in_features=1, conv_layers=2, width=32):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.layer1 = GraphConv(in_features, width, allow_zero_in_degree=True)
        self.layer2 = GraphConv(width, 1, allow_zero_in_degree=True)
        self.readout = AvgPooling()

    def forward(self, g):
        """Baseline SimpleGCN : start with `atom_features`."""
        g = g.local_var()

        features = g.ndata["atom_features"]

        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        x = self.readout(g, x)

        return torch.squeeze(x)
