"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import List, Optional, Tuple

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling, CFConv
from torch import nn
from torch.nn import functional as F

from jarvisdgl.config import ALIGNNConfig
from jarvisdgl.models.cgcnn import CGCNNConv
from jarvisdgl.models.utils import RBFExpansion


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        angle_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = CGCNNConv(
            node_features, edge_features, return_messages=True
        )
        self.edge_bottleneck = nn.Sequential(
            nn.Linear(node_features + edge_features, edge_features),
            nn.BatchNorm1d(edge_features),
            nn.Softplus(),
        )
        self.edge_update = CGCNNConv(edge_features, angle_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for CLGN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # CGCNN update on crystal graph
        x, m = self.node_update(g, x, y)

        # fuse messages to line graph inputs
        m = self.edge_bottleneck(torch.cat((y, m), 1))
        y = y + m

        # CGCNN update on line graph
        y = self.edge_update(lg, y, z)

        return x, y


class ALIGNN(nn.Module):
    """Line graph network."""

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        print(config)

        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )

        self.bn = nn.BatchNorm1d(config.node_features)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0, vmax=8.0, bins=config.edge_features, lengthscale=0.5
            ),
            nn.Linear(config.edge_features, 64),
            nn.BatchNorm1d(64),
            nn.Softplus(),
            nn.Linear(64, config.hidden_features),
            nn.BatchNorm1d(config.hidden_features),
            nn.Softplus(),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1, vmax=1.0, bins=config.angle_features, lengthscale=0.1
            ),
            nn.Linear(config.angle_features, 64),
            nn.BatchNorm1d(64),
            nn.Softplus(),
            nn.Linear(64, config.hidden_features),
            nn.BatchNorm1d(config.hidden_features),
            nn.Softplus(),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.node_features,
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.conv_layers)
            ]
        )
        self.final_conv = CGCNNConv(
            config.node_features, config.hidden_features
        )

        self.bn_final = nn.BatchNorm1d(config.node_features)

        self.readout = AvgPooling()

        self.fc = nn.Linear(config.node_features, config.output_features)

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, g: Tuple[dgl.DGLGraph, dgl.DGLGraph]):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        g, lg = g
        g = g.local_var()
        lg = lg.local_var()

        # initial node features: atom feature network...
        # conv-bn-relu
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)
        x = F.relu(self.bn(x))

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # angle features (fixed)
        z = self.angle_embedding(lg.edata.pop("h"))

        for alignn_layer in self.alignn_layers:
            x, y = alignn_layer(g, lg, x, y, z)

        # final node update
        x = self.final_conv(g, x, y)

        # norm-relu-pool-classify
        h = F.relu(self.bn_final(x))
        h = self.readout(g, h)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        return torch.squeeze(out)
