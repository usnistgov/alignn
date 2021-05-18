"""CGCNN: dgl implementation."""

from typing import Tuple

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
from dgl.nn import AvgPooling
from pydantic.typing import Literal
from torch import nn

# import torch
from alignn.models.utils import RBFExpansion
from alignn.utils import BaseSettings


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


class ACGCNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["alignn_cgcnn"]
    conv_layers: int = 3
    atom_input_features: int = 92
    edge_features: int = 40
    node_features: int = 92
    fc_layers: int = 1
    fc_features: int = 256
    output_features: int = 1
    alignn_layers: int = 3
    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class ACGCNNConv(nn.Module):
    """Xie and Grossman graph convolution function.

    10.1103/PhysRevLett.120.145301
    """

    def __init__(
        self,
        node_features: int = 64,
        edge_features: int = 32,
        return_messages: bool = False,
    ):
        """Initialize torch modules for CGCNNConv layer."""
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.return_messages = return_messages

        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.linear_src = nn.Linear(node_features, 2 * node_features)
        self.linear_dst = nn.Linear(node_features, 2 * node_features)
        self.linear_edge = nn.Linear(edge_features, 2 * node_features)
        self.bn_message = nn.BatchNorm1d(2 * node_features)

        # final batchnorm
        self.bn = nn.BatchNorm1d(node_features)

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

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # compute edge messages -- coalesce W_f and W_s from the paper
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))
        g.ndata["h_src"] = self.linear_src(node_feats)
        g.ndata["h_dst"] = self.linear_dst(node_feats)
        g.apply_edges(fn.u_add_v("h_src", "h_dst", "h_nodes"))
        m = g.edata.pop("h_nodes") + self.linear_edge(edge_feats)
        m = self.bn_message(m)

        # split messages into W_f and W_s terms
        # multiply output of atom interaction net and edge attention net
        # i.e. compute the term inside the summation in eq 5
        # σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        h_f, h_s = torch.chunk(m, 2, dim=1)
        m = torch.sigmoid(h_f) * F.softplus(h_s)
        g.edata["m"] = m

        # apply the convolution term in eq. 5 (without residual connection)
        # storing the results in edge features `h`
        g.update_all(
            message_func=fn.copy_e("m", "z"),
            reduce_func=fn.sum("z", "h"),
        )

        # final batchnorm
        h = self.bn(g.ndata.pop("h"))

        # residual connection plus nonlinearity
        out = F.softplus(node_feats + h)

        if self.return_messages:
            return out, m

        return out


class ACGCNN(nn.Module):
    """CGCNN dgl implementation."""

    def __init__(
        self, config: ACGCNNConfig = ACGCNNConfig(name="alignn_cgcnn")
    ):
        """Set up CGCNN modules."""
        super().__init__()

        self.rbf = RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_features)
        self.abf = RBFExpansion(
            vmin=-np.pi / 2, vmax=np.pi / 2, bins=config.edge_features
        )
        # self.abf = RBFExpansion(vmin=-1, vmax=1, bins=config.edge_features)
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )
        self.classification = config.classification
        self.conv_layers1 = nn.ModuleList(
            [
                ACGCNNConv(config.node_features, config.edge_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.conv_layers2 = nn.ModuleList(
            [
                ACGCNNConv(config.edge_features, config.edge_features)
                for _ in range(config.conv_layers)
            ]
        )
        self.readout = AvgPooling()

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.Softplus()
        )

        if config.zero_inflated:
            # add latent Bernoulli variable model to zero out
            # predictions in non-negative regression model
            self.zero_inflated = True
            self.fc_nonzero = nn.Linear(config.fc_features, 1)
            self.fc_scale = nn.Linear(config.fc_features, 1)
            # self.fc_shape = nn.Linear(config.fc_features, 1)
            self.fc_scale.bias.data = torch.tensor(
                # np.log(2.1), dtype=torch.float
                2.1,
                dtype=torch.float,
            )
            if self.classification:
                raise ValueError(
                    "Classification not implemented with ZIG loss."
                )
        else:
            self.zero_inflated = False
            if self.classification:
                self.fc_out = nn.Linear(config.fc_features, 2)
                self.softmax = nn.LogSoftmax(dim=1)
            else:
                self.fc_out = nn.Linear(
                    config.fc_features, config.output_features
                )
        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(
                    np.log(avg_gap), dtype=torch.float
                )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, g) -> torch.Tensor:
        """CGCNN function mapping graph to outputs."""
        g, lg = g
        g = g.local_var()
        # lg = g.line_graph(shared=True)
        # lg.apply_edges(compute_bond_cosines)
        angle_features = self.abf(lg.edata.pop("h"))
        # fixed edge features: RBF-expanded bondlengths
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        edge_features = self.rbf(bondlength)

        # initial node features: atom feature network...
        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(v)

        # CGCNN-Conv block: update node features
        for conv_layer1, conv_layer2 in zip(
            self.conv_layers1, self.conv_layers2
        ):
            node_features = conv_layer1(g, node_features, edge_features)
            edge_features = conv_layer2(lg, edge_features, angle_features)

        # crystal-level readout
        features = self.readout(g, node_features)
        features = F.softplus(features)
        features = self.fc(features)
        features = F.softplus(features)

        if self.zero_inflated:
            logit_p = self.fc_nonzero(features)
            log_scale = self.fc_scale(features)
            # log_shape = self.fc_shape(features)

            # pred = (torch.sigmoid(logit_p)
            #         * torch.exp(log_scale)
            #         * torch.exp(log_shape))
            # out = torch.where(p < 0.5, torch.zeros_like(out), out)
            return (
                torch.squeeze(logit_p),
                torch.squeeze(log_scale),
                # torch.squeeze(log_shape),
            )

        else:
            out = self.fc_out(features)
            if self.link:
                out = self.link(out)
        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)


class ZeroInflatedGammaLoss(nn.modules.loss._Loss):
    """Zero inflated Gamma regression loss."""

    def predict(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        """Combine ZIG multi-part outputs to yield real-valued predictions."""
        # logit_p, log_scale, log_shape = inputs
        logit_p, log_scale = inputs
        return (
            torch.sigmoid(logit_p)
            * F.softplus(log_scale)
            # * torch.exp(log_scale)
            # * (1 + torch.exp(log_shape))
        )

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-inflated Gamma loss.

        binary crossentropy loss combined with Gamma negative log likelihood
        """
        # logit_p, log_scale, log_shape = inputs
        logit_p, log_scale = inputs

        bce_loss = F.binary_cross_entropy_with_logits(
            logit_p, target, reduction="sum"
        )

        indicator = target > 0
        # g_loss = F.mse_loss(
        #     log_scale[indicator],
        #     torch.log(target[indicator]), reduction="sum"
        # )
        # g_loss = F.mse_loss(
        #     torch.exp(log_scale[indicator]),
        # target[indicator], reduction="sum"
        # )
        g_loss = F.mse_loss(
            F.softplus(log_scale[indicator]),
            target[indicator],
            reduction="sum",
        )

        return (bce_loss + g_loss) / target.numel()
        # return bce_loss + torch.tensor(2.0) * g_loss.sum() / indicator.sum()
