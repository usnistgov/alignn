"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling

# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from alignn.models.utils import RBFExpansion
from alignn.utils import BaseSettings


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 4
    alignn_order: Literal["triplet-pair", "pair-triplet"] = "triplet-pair"
    squeeze_ratio: float = 0.5
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1
    norm: Literal["layernorm", "batchnorm"] = "layernorm"

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self,
        node_input_features: int,
        edge_input_features: int,
        output_features: int,
        residual: bool = True,
        norm = nn.BatchNorm1d,
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(
            node_input_features, output_features, bias=False
        )
        self.dst_gate = nn.Linear(
            node_input_features, output_features, bias=False
        )
        self.edge_gate = nn.Linear(
            edge_input_features, output_features, bias=False
        )
        self.bn_edges = norm(output_features)

        self.src_update = nn.Linear(
            node_input_features, output_features, bias=False
        )
        self.dst_update = nn.Linear(
            node_input_features, output_features, bias=False
        )
        self.bn_nodes = norm(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class BottleneckLayer(nn.Module):
    """MLP reduction bottleneck."""

    def __init__(self, in_features: int, reduction: float = 0.5):
        """Set up MLP bottleneck.

        in_features should be divisible by reduction^2
        """
        super().__init__()

        intermediate_features = int(in_features * reduction)
        bottleneck_features = int(intermediate_features * reduction)

        self.layers = nn.Sequential(
            MLPLayer(in_features, intermediate_features),
            MLPLayer(intermediate_features, bottleneck_features),
        )

    def forward(self, x):
        """Run the bottleneck layer."""
        return self.layers(x)


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        order: Literal["triplet-pair", "pair-triplet"] = "triplet-pair",
        reduction: float = 0.5,
        norm = nn.BatchNorm1d,
    ):
        """Set up ALIGNN parameters.

        `reduction` sets amount of compression in squeeze-expand component.
        currently supports reduction=0.5 (4x compression, 2 layers of 50%)
        setting reduction=1.0 fully disables this feature.
        """
        super().__init__()

        gcn_size = in_features
        self.order = order
        self.squeeze_expand = False

        # allow only 4x reduction or skip it...
        assert reduction in (0.5, 1.0)

        if reduction < 1.0:
            self.squeeze_expand = True
            self.bottleneck_size = int(in_features * reduction * reduction)
            gcn_size = self.bottleneck_size

            self.node_bottleneck = BottleneckLayer(in_features, reduction=0.5)
            self.pair_bottleneck = BottleneckLayer(in_features, reduction=0.5)
            self.triplet_bottleneck = BottleneckLayer(
                in_features, reduction=0.5
            )

            self.node_expand = MLPLayer(self.bottleneck_size, out_features)
            self.pair_expand = MLPLayer(self.bottleneck_size, out_features)
            self.triplet_expand = MLPLayer(self.bottleneck_size, out_features)

        # y: in_features
        # z: in_features
        self.edge_update = EdgeGatedGraphConv(
            gcn_size,
            gcn_size,
            gcn_size,
            residual=False,
            norm=norm,
        )

        # x: in_features
        # y: out_features
        self.node_update = EdgeGatedGraphConv(
            gcn_size,
            gcn_size,
            gcn_size,
            residual=False,
            norm=norm
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()

        # save inputs for residual connection
        x_in, y_in, z_in = x, y, z

        if self.squeeze_expand:
            x = self.node_bottleneck(x)
            y = self.pair_bottleneck(y)
            z = self.triplet_bottleneck(z)

        if self.order == "triplet-pair":
            m, z = self.edge_update(lg, y, z)
            x, y = self.node_update(g, x, m)

        elif self.order == "pair-triplet":
            x, m = self.node_update(g, x, y)
            y, z = self.edge_update(lg, m, z)

        if self.squeeze_expand:
            x = self.node_expand(x)
            y = self.pair_expand(y)
            z = self.triplet_expand(z)

        # residual connection
        x += x_in
        y += y_in
        z += z_in

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int, norm=nn.BatchNorm1d):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            norm(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        print(config)
        self.classification = config.classification

        norm = {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}[
            config.norm
        ]

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features, norm=norm
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features, norm=norm),
            MLPLayer(config.embedding_features, config.hidden_features, norm=norm),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features, norm=norm),
            MLPLayer(config.embedding_features, config.hidden_features, norm=norm),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                    order=config.alignn_order,
                    reduction=config.squeeze_ratio,
                    norm=norm,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features,
                    config.hidden_features,
                    config.hidden_features,
                    norm=norm,
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)
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

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)
