from typing import Optional

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from dgl.nn import AvgPooling
from torch import nn


class RBFExpansion(nn.Module):
    def __init__(
        self,
        vmin: float = 1,
        vmax: float = 5,
        bins: int = 10,
        lengthscale: Optional[float] = None,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )


class CGCNNConv(nn.Module):
    """Xie and Grossman graph convolution function
    10.1103/PhysRevLett.120.145301
    """

    def __init__(self, node_features: int = 64, edge_features: int = 32):
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
        """CGCNNConv edge update function

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
        """CGCNN convolution defined in Eq 5
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
    def __init__(
        self,
        atom_input_features: int = 1,
        node_features: int = 32,
        edge_features: int = 32,
        conv_layers: int = 3,
        fc_features: int = 64,
        output_features: int = 1,
        logscale=False,
    ):
        super().__init__()

        self.rbf = RBFExpansion(vmin=1, vmax=5, bins=edge_features)
        self.atom_embedding = nn.Linear(atom_input_features, node_features)

        self.conv_layers = nn.ModuleList(
            [
                CGCNNConv(node_features, edge_features)
                for _ in range(conv_layers)
            ]
        )

        self.readout = AvgPooling()

        self.fc = nn.Sequential(
            nn.Linear(node_features, fc_features), nn.Softplus()
        )

        self.fc_out = nn.Linear(fc_features, output_features)
        self.logscale = logscale

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        g = g.local_var()

        # fixed edge features: RBF-expanded bondlengths
        edge_features = self.rbf(g.edata.pop("bondlength"))

        # initial node features: atom feature network...
        v = g.ndata.pop("atom_features").type(torch.FloatTensor)

        node_features = self.atom_embedding(v)

        # CGCNN-Conv block: update node features
        for conv_layer in self.conv_layers:
            node_features = conv_layer(g, node_features, edge_features)

        # crystal-level readout
        features = self.readout(g, node_features)
        features = F.softplus(features)
        features = self.fc(features)

        out = self.fc_out(features)

        if self.logscale:
            out = torch.exp(out)

        return torch.squeeze(out)
