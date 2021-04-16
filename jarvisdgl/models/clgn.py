"""A prototype crystal line graph network dgl implementation."""
from typing import List, Optional

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling, CFConv
from torch import nn
from torch.nn import functional as F

from jarvisdgl.config import CLGNConfig
from jarvisdgl.models.utils import RBFExpansion


def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors."""
    # line graph edge: (a, b), (b, c)
    # `a -> b -> c`
    # use law of cosines to compute angles cosines
    # negate src bond so displacements are like `a <- b -> c`
    # cos(theta) = ba \dot bc / (||ba|| ||bc||)
    r1 = -edges.src["r"]
    r2 = edges.dst["r"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"h": bond_cosine}


class CLGNLayer(nn.Module):
    """Crystal line graph network layer."""

    def __init__(
        self,
        node_in_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        edge_out_feats: int,
        angle_in_feats: int,
        hidden_feats: int,
    ):
        """Initialize CLGN layer."""
        super().__init__()
        # self.bn = nn.BatchNorm1d(node_out_feats)

        self.project_node = nn.Linear(node_in_feats, hidden_feats)
        self.project_out = nn.Linear(hidden_feats, node_out_feats)
        self.project_edge = nn.Sequential(
            nn.Linear(edge_in_feats, hidden_feats),
            nn.Softplus(),
            nn.Linear(hidden_feats, hidden_feats),
        )

        # self.g_conv = CFConv(
        #     node_in_feats, edge_in_feats, hidden_feats, node_out_feats
        # )

        self.lg_conv = CFConv(
            edge_in_feats + hidden_feats,
            angle_in_feats,
            hidden_feats,
            edge_out_feats,
        )

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
        # node update
        # like CFConv, but save edge messages to fuse to line graph
        # https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/cfconv.html
        # x = self.g_conv(g, x, y)
        g.srcdata["hv"] = self.project_node(x)
        g.edata["he"] = self.project_edge(y)
        g.apply_edges(fn.u_mul_e("hv", "he", "m"))
        g.update_all(fn.copy_e("m", "m"), fn.sum("m", "hv"))
        x = self.project_out(g.ndata.pop("hv"))
        x = F.softplus(x)

        # edge update: CFConv
        # concatenate edge features and edge messages
        y = torch.cat((y, g.edata.pop("m")), 1)
        y = self.lg_conv(lg, y, z)
        y = F.softplus(y)

        return x, y


class CLGN(nn.Module):
    """Line graph network."""

    def __init__(self, config: CLGNConfig = CLGNConfig(name="clgn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        print(config)

        self.rbf = RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_features)
        self.angle_bf = RBFExpansion(
            vmin=-1, vmax=1.0, bins=config.angle_features
        )
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )

        self.bn = nn.BatchNorm1d(config.node_features)

        self.conv1 = CLGNLayer(
            config.node_features,
            config.node_features,
            config.edge_features,
            config.edge_features,
            config.angle_features,
            config.hidden_features,
        )
        self.conv2 = CLGNLayer(
            config.node_features,
            config.node_features,
            config.edge_features,
            config.edge_features,
            config.angle_features,
            config.hidden_features,
        )
        self.conv3 = CFConv(
            config.node_features,
            config.edge_features,
            config.hidden_features,
            config.node_features,
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

    def forward(self, g):
        """Baseline SimpleGCN : start with `atom_features`."""
        g, lg = g
        g = g.local_var()
        lg = lg.local_var()
        # lg = g.line_graph(shared=True)

        # obtain bond angle cosines from displacement vectors
        # store them in lg.edata["h"]
        # lg.apply_edges(compute_bond_cosines)

        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        edge_features = self.rbf(bondlength)

        # initial node features: atom feature network...
        # conv-bn-relu
        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(v)
        node_features = F.relu(self.bn(node_features))

        angle_features = self.angle_bf(lg.edata.pop("h"))
        x, y = self.conv1(g, lg, node_features, edge_features, angle_features)
        x, y = self.conv2(g, lg, x, y, angle_features)
        x = self.conv3(g, x, y)

        # norm-relu-pool-classify
        h = F.relu(self.bn_final(x))

        h = self.readout(g, h)

        out = self.fc(h)

        if self.link:
            out = self.link(out)

        return torch.squeeze(out)
