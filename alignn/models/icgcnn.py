"""CGCNN: dgl implementation."""

from typing import Tuple
import dgl
import dgl.function as fn

# import numpy as np
import torch
import torch.nn.functional as F
from dgl.nn import AvgPooling
from pydantic.typing import Literal
from torch import nn

from alignn.models.utils import RBFExpansion
from alignn.utils import BaseSettings


class ICGCNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.icgcnn."""

    name: Literal["icgcnn"]
    conv_layers: int = 3
    atom_input_features: int = 1
    edge_features: int = 16
    node_features: int = 64
    fc_layers: int = 1
    fc_features: int = 64
    output_features: int = 1

    # if logscale is set, apply `exp` to final outputs
    # to constrain predictions to be positive
    logscale: bool = False
    hurdle: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class CGCNNUpdate(nn.Module):
    """Helper nn.Module for CGCNN-style updates."""

    def __init__(self, in_features: int, out_features: int):
        """Set up CGCNN internal parameters."""
        super().__init__()

        # edge interaction model (W_f / W_1)
        self.conv = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Sigmoid(),
        )

        # edge attention model (W_s / W_2)
        self.screen = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        """Apply CGCNNConv-style update."""
        return self.conv(x) * self.screen(x)


class iCGCNNConv(nn.Module):
    """Park and Wolverton iCGCNN convolution.

    10.1103/PhysRevMaterials.4.063801

    In the papers, nodes are v_i, v_j, edges are u_ij
    In DGL, nodes are u (src) and v (dst), edges are e
    """

    def __init__(self, node_features: int = 64, edge_features: int = 32):
        """Initialize torch modules for iCGCNNConv layer."""
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features

        # iCGCNNConv has a node update and an edge update
        # each update has a pairwise and triplet interaction term

        # pairwise features:
        # z_ij = cat(v_i, v_j, u_ij)
        pair_sz = 2 * self.node_features + self.edge_features

        # triplet features:
        # z_ijl = cat(v_i, v_j, v_l, u_ij, u_il)
        triple_sz = 3 * self.node_features + 2 * self.edge_features

        # node update functions
        self.node_pair_update = CGCNNUpdate(pair_sz, self.node_features)
        self.node_triple_update = CGCNNUpdate(triple_sz, self.node_features)

        # edge update functions
        self.edge_pair_update = CGCNNUpdate(pair_sz, self.edge_features)
        self.edge_triple_update = CGCNNUpdate(triple_sz, self.edge_features)

        # final batchnorm
        self.node_bn = nn.BatchNorm1d(self.node_features)
        self.edge_bn = nn.BatchNorm1d(self.edge_features)

    def combine_edge_features(self, edges):
        """Edge update for iCGCNNConv.

        concatenate source and destination node features with edge features
        then apply the edge update modulated by the edge interaction model
        """
        # form augmented edge features z_ij = [v_i, v_j, u_ij]
        z = torch.cat((edges.src["h"], edges.dst["h"], edges.data["h"]), dim=1)

        return {"z_pair": z}

    def combine_triplet_features(self, edges):
        """Line graph edge update for iCGCNNConv."""
        z_ijl = torch.cat(
            (
                edges.src["src_h"],
                edges.src["dst_h"],
                edges.dst["dst_h"],
                edges.src["h"],
                edges.dst["h"],
            ),
            dim=1,
        )
        return {"z_triple": z_ijl}

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CGCNN convolution defined in Eq 1, 2, and 3.

        10.1103/PhysRevMaterials.4.063801

        This convolution function forms z_ij and z_ijl tensors
        and performs two aggregrations each:
        one to update node features and one to update edge features
        """
        g = g.local_var()

        g.ndata["h"] = node_feats
        g.edata["h"] = edge_feats

        # propagate node features to line graph nodes
        g.apply_edges(
            func=lambda edges: {
                "src_h": edges.src["h"],
                "dst_h": edges.dst["h"],
            }
        )

        # line graph edge == pairs of bonds (u,v), (v,w)
        # z_ijl = cat(src[src], src[dst], dst[dst])
        lg = dgl.line_graph(g, shared=True)

        # both node and edge updates need both z_ij and z_ijl
        # compute these separately with apply_edges
        # apply multiple linear maps within that message function
        # then propagate them with separate update_all call each

        # compute z_ij (n_pairs, pair_sz)
        g.apply_edges(self.combine_edge_features)
        z_pair = g.edata.pop("z_pair")

        # compute z_ijl_kk' (n_triples, triple_sz)
        lg.apply_edges(self.combine_triplet_features)
        z_triple = lg.edata.pop("z_triple")

        # node update: eqs 1 and 2
        # eq 1 (pre-reduction) (n_edges, node_sz)
        # reduces to (n_nodes, node_sz)
        h_node_pair = self.node_pair_update(z_pair)

        # eq 2 (pre-reduction) (n_triples, node_sz)
        # reduces to (n_nodes, node_sz)
        h_node_triple = self.node_triple_update(z_triple)

        # edge update: eq 3
        # eq 3 term 1 (n_edges, edge_sz)
        # no reduction needed
        h_edge_pair = self.edge_pair_update(z_pair)

        # eq 3 term 2 (pre-reduction) (n_triples, edge_sz)
        # reduces to (n_edges, edge_sz)
        h_edge_triple = self.edge_triple_update(z_triple)

        # aggregate triple features to edges, then edges to nodes
        lg.edata["h_node_triple"] = h_node_triple
        lg.edata["h_edge_triple"] = h_edge_triple

        # triple -> edge aggregation (i.e. LG edges to LG nodes)
        # partial summation in Eq 2 (sum over l, k')
        lg.update_all(
            fn.copy_e("h_node_triple", "h_node_triple"),
            fn.sum("h_node_triple", "h_node_triple"),
        )
        # sum over l, k' in Eq 3
        lg.update_all(
            fn.copy_e("h_edge_triple", "h_edge_triple"),
            fn.sum("h_edge_triple", "h_edge_triple"),
        )

        # further aggregate triplet features to nodes
        # complete summation in eq 2 (sum over j, k)
        g.edata["h_node_triple"] = lg.ndata.pop("h_node_triple")
        g.update_all(
            fn.copy_e("h_node_triple", "h_node_triple"),
            fn.sum("h_node_triple", "h_node_triple"),
        )

        # edge-wise reduction in eq 1 (sum over j,k)
        g.edata["h_node_pair"] = h_node_pair
        g.update_all(
            message_func=fn.copy_e("h_node_pair", "h_node_pair"),
            reduce_func=fn.sum("h_node_pair", "h_node_pair"),
        )

        # final batchnorm
        h_node = g.ndata.pop("h_node_pair") + g.ndata.pop("h_node_triple")
        h_node = self.node_bn(h_node)

        h_edge = h_edge_pair + lg.ndata.pop("h_edge_triple")
        h_edge = self.edge_bn(h_edge)

        # residual connection plus nonlinearity
        return F.softplus(node_feats + h_node), F.softplus(edge_feats + h_edge)


class iCGCNN(nn.Module):
    """iCGCNN dgl implementation."""

    def __init__(self, config: ICGCNNConfig = ICGCNNConfig(name="icgcnn")):
        """Set up CGCNN modules."""
        super().__init__()

        self.rbf = RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_features)
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )
        self.classification = config.classification
        self.conv_layers = nn.ModuleList(
            [
                iCGCNNConv(config.node_features, config.edge_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.readout = AvgPooling()

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.Softplus()
        )

        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(config.fc_features, config.output_features)

        self.logscale = config.logscale

    def forward(self, g) -> torch.Tensor:
        """CGCNN function mapping graph to outputs."""
        g, lg = g
        g = g.local_var()

        # fixed edge features: RBF-expanded bondlengths
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        h_edge = self.rbf(bondlength)

        # initial node features: atom feature network...
        v = g.ndata.pop("atom_features")
        h_node = self.atom_embedding(v)

        # CGCNN-Conv block: update node features
        for conv_layer in self.conv_layers:
            h_node, h_edge = conv_layer(g, h_node, h_edge)

        # crystal-level readout
        features = self.readout(g, h_node)
        features = F.softplus(features)
        features = self.fc(features)
        features = F.softplus(features)

        out = self.fc_out(features)

        if self.logscale:
            out = torch.exp(out)
        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)

        return torch.squeeze(out)
