"""A baseline graph convolution network dgl implementation."""
import dgl
import torch
from dgl.nn import AvgPooling, GraphConv
from torch import nn
from torch.nn import functional as F

from jarvisdgl.config import SimpleGCNConfig


class SimpleGCN(nn.Module):
    """Module for simple GCN."""

    def __init__(
        self, config: SimpleGCNConfig = SimpleGCNConfig(name="simplegcn")
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.layer1 = GraphConv(config.atom_input_features, config.width)
        self.layer2 = GraphConv(config.width, config.output_features)
        self.readout = AvgPooling()

    def forward(self, g):
        """Baseline SimpleGCN : start with `atom_features`."""
        g = g.local_var()

        features = g.ndata["atom_features"]

        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        x = self.readout(g, x)

        return torch.squeeze(x)
