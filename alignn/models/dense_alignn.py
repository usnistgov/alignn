"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

# from typing import List, Optional, Tuple, Union
import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
from pydantic import root_validator
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from alignn.models.utils import RBFExpansion
from alignn.utils import BaseSettings


class DenseALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.dense_alignn."""

    name: Literal["dense_alignn"]
    alignn_layers: int = 3
    gcn_layers: int = 3
    atom_input_features: int = 92
    edge_input_features: int = 81
    triplet_input_features: int = 40
    embedding_features: int = 92
    initial_features: int = 92
    bottleneck_features: int = 92
    residual: bool = True
    growth_rate: int = 64
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1
    norm: Literal["batchnorm", "layernorm"] = "layernorm"

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False

    @root_validator()
    def ensure_residual_dimensions_match(cls, values):
        """Check that residual connections are allowed."""
        initial_features = values.get("initial_features")
        bottleneck_features = values.get("bottleneck_features")
        residual = values.get("residual")
        if residual:
            if initial_features != bottleneck_features:
                raise ValueError(
                    "input and bottleneck dims must match to use residuals."
                )

        return values

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
        norm=nn.BatchNorm1d,
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual

        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.norm_edges = norm(edge_input_features)
        self.src_gate = nn.Linear(
            node_input_features, output_features, bias=False
        )
        self.dst_gate = nn.Linear(
            node_input_features, output_features, bias=False
        )
        self.edge_gate = nn.Linear(
            edge_input_features, output_features, bias=False
        )

        self.norm_nodes = norm(node_input_features)
        self.src_update = nn.Linear(
            node_input_features, output_features, bias=False
        )
        self.dst_update = nn.Linear(
            node_input_features, output_features, bias=False
        )

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

        # pre-normalization, pre-activation
        # node and edge updates
        x = F.silu(self.norm_nodes(node_feats))
        y = F.silu(self.norm_edges(edge_feats))

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(x)
        g.ndata["e_dst"] = self.dst_gate(x)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        y = g.edata.pop("e_nodes") + self.edge_gate(y)

        g.edata["sigma"] = torch.sigmoid(y)
        g.ndata["Bh"] = self.dst_update(x)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(x) + g.ndata.pop("h")

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        residual: bool = False,
        norm=nn.BatchNorm1d,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.residual = residual
        self.node_update = EdgeGatedGraphConv(
            in_features, in_features, out_features, residual, norm
        )
        # y: out_features
        # z: in_features
        self.edge_update = EdgeGatedGraphConv(
            out_features, in_features, out_features, residual, norm
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
        # y_initial = y

        # Edge-gated graph convolution update on crystal graph
        # x, y are concatenated feature maps
        x, y = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        # y: growth_rate
        # z: concatenated feature map size
        y, z = self.edge_update(lg, y, z)

        # # residual edge connection around line graph convolution
        # y = y_initial + y

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(
        self, in_features: int, out_features: int, norm=nn.BatchNorm1d
    ):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.ModuleDict(
            {
                "linear": nn.Linear(in_features, out_features),
                "norm": norm(out_features),
                "activation": nn.SiLU(),
            }
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        for name, cpt in self.layer.items():
            x = cpt(x)
        return x


class DenseGCNBlock(nn.Module):
    """Dense block of gated graph convolution layers."""

    def __init__(
        self,
        n_layers: int = 3,
        input_features: int = 32,
        growth_rate: int = 32,
        output_features: int = 32,
        residual: bool = True,
        norm=nn.BatchNorm1d,
    ):
        """Densely-connected gated graph convolution layers."""
        super().__init__()
        self.residual = residual
        self.bottleneck_inputs = input_features + n_layers * growth_rate
        self.layers = nn.ModuleList()

        for idx in range(n_layers):
            in_features = input_features + idx * growth_rate
            self.layers.append(
                EdgeGatedGraphConv(
                    in_features,
                    in_features,
                    growth_rate,
                    residual=False,
                    norm=norm,
                )
            )

        self.bottleneck_x = nn.Sequential(
            norm(self.bottleneck_inputs),
            nn.SiLU(),
            nn.Linear(self.bottleneck_inputs, output_features, bias=False),
        )
        self.bottleneck_y = nn.Sequential(
            norm(self.bottleneck_inputs),
            nn.SiLU(),
            nn.Linear(self.bottleneck_inputs, output_features, bias=False),
        )

    def forward(self, g, x, y):
        """Gated GCN updates: update node, edge features."""
        x_identity = x
        y_identity = y
        xs, ys = [x], [y]
        for gcn_layer in self.layers:
            new_x, new_y = gcn_layer(g, torch.cat(xs, 1), torch.cat(ys, 1))
            xs.append(new_x)
            ys.append(new_y)

        x = self.bottleneck_x(torch.cat(xs, 1))
        y = self.bottleneck_y(torch.cat(ys, 1))

        if self.residual:
            x = x_identity + x
            y = y_identity + y

        return x, y


class DenseALIGNNBlock(nn.Module):
    """Dense block of ALIGNN updates."""

    def __init__(
        self,
        n_layers: int = 3,
        input_features: int = 32,
        growth_rate: int = 32,
        output_features: int = 32,
        residual: bool = True,
        norm=nn.BatchNorm1d,
    ):
        """Dense block of ALIGNN updates."""
        super().__init__()
        self.residual = residual
        self.bottleneck_inputs = input_features + n_layers * growth_rate

        self.layers = nn.ModuleList()
        for idx in range(n_layers):
            in_features = input_features + idx * growth_rate
            self.layers.append(
                ALIGNNConv(in_features, growth_rate, residual=False, norm=norm)
            )

        self.bottleneck_x = nn.Sequential(
            norm(self.bottleneck_inputs),
            nn.SiLU(),
            nn.Linear(self.bottleneck_inputs, output_features, bias=False),
        )
        self.bottleneck_y = nn.Sequential(
            norm(self.bottleneck_inputs),
            nn.SiLU(),
            nn.Linear(self.bottleneck_inputs, output_features, bias=False),
        )

    def forward(self, g, lg, x, y, z):
        """ALIGNN updates: update node, edge, triplet features.

        DenseNet style updates:
        maintain a list of x, y, z features
        and concatenate all previous feature maps
        to form input for each layer
        """
        x_identity = x
        xs = [x]
        y_identity = y
        ys = [y]
        # z_identity = z
        zs = [z]

        for alignn_layer in self.layers:
            new_x, new_y, new_z = alignn_layer(
                g, lg, torch.cat(xs, 1), torch.cat(ys, 1), torch.cat(zs, 1)
            )
            xs.append(new_x)
            ys.append(new_y)
            zs.append(new_z)

        x = self.bottleneck_x(torch.cat(xs, 1))
        y = self.bottleneck_y(torch.cat(ys, 1))

        # residual connections around graph dense graph convolution block
        if self.residual:
            x = x_identity + x
            y = y_identity + y

        return x, y


class DenseALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(
        self,
        config: DenseALIGNNConfig = DenseALIGNNConfig(name="dense_alignn"),
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        print(config)
        self.classification = config.classification
        norm = {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}[
            config.norm
        ]

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.initial_features, norm
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
                lengthscale=0.5,
            ),
            MLPLayer(
                config.edge_input_features, config.embedding_features, norm
            ),
            MLPLayer(config.embedding_features, config.initial_features, norm),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-np.pi,
                vmax=np.pi,
                bins=config.triplet_input_features,
            ),
            MLPLayer(
                config.triplet_input_features, config.embedding_features, norm
            ),
            MLPLayer(config.embedding_features, config.initial_features, norm),
        )

        if config.alignn_layers > 0:
            self.dense_alignn_block = DenseALIGNNBlock(
                n_layers=config.alignn_layers,
                input_features=config.initial_features,
                growth_rate=config.growth_rate,
                output_features=config.bottleneck_features,
                residual=config.residual,
                norm=norm,
            )
        else:
            self.dense_alignn_block = None

        initial_features = config.initial_features
        self.dense_gcn_block = DenseGCNBlock(
            n_layers=config.gcn_layers,
            input_features=initial_features,
            growth_rate=config.growth_rate,
            output_features=config.bottleneck_features,
            residual=config.residual,
            norm=norm,
        )

        self.readout = AvgPooling()

        if self.classification:
            self.fc = nn.Linear(config.bottleneck_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(
                config.bottleneck_features, config.output_features
            )

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

        # Kaiming initialization not working out
        # stick with default Glorot
        # self.apply(self.reset_parameters)

    @staticmethod
    def reset_parameters(m):
        """He initialization."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode="fan_out", nonlinearity="relu"
            )
            # nn.init.constant_(m.bias, 0)

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if self.dense_alignn_block is not None:
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

        x, y = self.dense_alignn_block(g, lg, x, y, z)
        x, y = self.dense_gcn_block(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)
        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)

        return torch.squeeze(out)
