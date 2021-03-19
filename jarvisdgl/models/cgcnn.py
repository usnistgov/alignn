"""CGCNN: dgl implementation."""

from typing import Optional

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from dgl.nn import AvgPooling
from torch import nn

from jarvisdgl.config import CGCNNConfig
from jarvisdgl.models.utils import RBFExpansion


class CGCNNConv(nn.Module):
    """Xie and Grossman graph convolution function.

    10.1103/PhysRevLett.120.145301
    """

    def __init__(self, node_features: int = 64, edge_features: int = 32):
        """Initialize torch modules for CGCNNConv layer."""
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features

        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        in_feats = 2 * self.node_features + self.edge_features

        # edge interaction model (W_f)
        self.edge_interaction = nn.Sequential(
            nn.Linear(in_feats, self.node_features),
            nn.BatchNorm1d(self.node_features),
            nn.Sigmoid(),
        )

        # edge attention model (W_s)
        self.edge_update = nn.Sequential(
            nn.Linear(in_feats, self.node_features),
            nn.BatchNorm1d(self.node_features),
            nn.Softplus(),
        )

        # final batchnorm
        self.bn = nn.BatchNorm1d(self.node_features)

    def combine_edge_features(self, edges):
        """Edge update for CGCNNConv.

        concatenate source and destination node features with edge features
        then apply the edge update modulated by the edge interaction model
        """
        # form augmented edge features z_ij = [v_i, v_j, u_ij]
        z = torch.cat((edges.src["h"], edges.dst["h"], edges.data["h"]), dim=1)

        # multiply output of atom interaction net and edge attention net
        # i.e. compute the term inside the summation in eq 5
        # σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        return {"z": self.edge_interaction(z) * self.edge_update(z)}

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """CGCNN convolution defined in Eq 5.

        10.1103/PhysRevLett.120.14530
        """
        g = g.local_var()

        g.ndata["h"] = node_feats
        g.edata["h"] = edge_feats

        # apply the convolution term in eq. 5 (without residual connection)
        # storing the results in edge features `h`
        g.update_all(
            message_func=self.combine_edge_features,
            reduce_func=fn.sum("z", "h"),
        )

        # final batchnorm
        h = self.bn(g.ndata.pop("h"))

        # residual connection plus nonlinearity
        return F.softplus(node_feats + h)


class CGCNN(nn.Module):
    """CGCNN dgl implementation."""

    def __init__(self, config: CGCNNConfig = CGCNNConfig(name="cgcnn")):
        """Set up CGCNN modules."""
        super().__init__()

        self.rbf = RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_features)
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )

        self.conv_layers = nn.ModuleList(
            [
                CGCNNConv(config.node_features, config.edge_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.readout = AvgPooling()

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.Softplus()
        )

        if config.hurdle:
            # add latent Bernoulli variable model to zero out
            # predictions in non-negative regression model
            self.hurdle = True
            self.fc_hurdle = nn.Linear(config.fc_features, 1)
        else:
            self.hurdle = False

        self.fc_out = nn.Linear(config.fc_features, config.output_features)
        self.logscale = config.logscale

        if self.logscale:
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc_out.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """CGCNN function mapping graph to outputs."""
        g = g.local_var()

        # fixed edge features: RBF-expanded bondlengths
        edge_features = self.rbf(g.edata.pop("bondlength"))

        # initial node features: atom feature network...
        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(v)

        # CGCNN-Conv block: update node features
        for conv_layer in self.conv_layers:
            node_features = conv_layer(g, node_features, edge_features)

        # crystal-level readout
        features = self.readout(g, node_features)
        features = F.softplus(features)
        features = self.fc(features)
        features = F.softplus(features)

        out = self.fc_out(features)

        if self.logscale:
            out = torch.exp(out)

        if self.hurdle:
            logits = self.fc_hurdle(features)
            p = torch.sigmoid(logits)
            out = torch.where(p < 0.5, torch.zeros_like(out), out)

        return torch.squeeze(out)
