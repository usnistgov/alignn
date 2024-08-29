"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""

from typing import Tuple, Union
from torch.autograd import grad
import dgl
import dgl.function as fn
import numpy as np
from dgl.nn import AvgPooling
import torch

# from dgl.nn.functional import edge_softmax
from typing import Literal
from torch import nn
from torch.nn import functional as F
from alignn.models.utils import RBFExpansion
from alignn.graphs import compute_bond_cosines
from alignn.utils import BaseSettings


class ALIGNNAtomWiseConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn_atomwise"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1
    grad_multiplier: int = -1
    calculate_gradient: bool = True
    atomwise_output_features: int = 0
    graphwise_weight: float = 1.0
    gradwise_weight: float = 0.0
    stresswise_weight: float = 0.0
    atomwise_weight: float = 0.0
    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    force_mult_natoms: bool = False
    energy_mult_natoms: bool = False
    include_pos_deriv: bool = False
    use_cutoff_function: bool = False
    inner_cutoff: float = 6  # Ansgtrom
    stress_multiplier: float = 1
    add_reverse_forces: bool = False  # will make True as default soon
    lg_on_fly: bool = False  # will make True as default soon
    batch_stress: bool = True
    multiply_cutoff: bool = False
    extra_features: int = 0
    exponent: int = 3

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


def cutoff_function_based_edges_old(r, inner_cutoff=4):
    """Apply smooth cutoff to pairwise interactions

    r: bond lengths
    inner_cutoff: cutoff radius

    inside cutoff radius, apply smooth cutoff envelope
    outside cutoff radius: hard zeros
    """
    ratio = r / inner_cutoff
    return torch.where(
        ratio <= 1,
        1 - 6 * ratio**5 + 15 * ratio**4 - 10 * ratio**3,
        torch.zeros_like(r),
    )


def cutoff_function_based_edges(r, inner_cutoff=4, exponent=3):
    """Apply smooth cutoff to pairwise interactions

    r: bond lengths
    inner_cutoff: cutoff radius

    inside cutoff radius, apply smooth cutoff envelope
    outside cutoff radius: hard zeros
    """
    ratio = r / inner_cutoff
    c1 = -(exponent + 1) * (exponent + 2) / 2
    c2 = exponent * (exponent + 2)
    c3 = -exponent * (exponent + 1) / 2
    envelope = (
        1
        + c1 * ratio**exponent
        + c2 * ratio ** (exponent + 1)
        + c3 * ratio ** (exponent + 2)
    )
    # r_cut = inner_cutoff
    # r_on = inner_cutoff+1

    # r_sq = r * r
    # r_on_sq = r_on * r_on
    # r_cut_sq = r_cut * r_cut
    # envelope = (r_cut_sq - r_sq)
    # ** 2 * (r_cut_sq + 2 * r_sq - 3 * r_on_sq)/ (r_cut_sq - r_on_sq) ** 3
    return torch.where(r <= inner_cutoff, envelope, torch.zeros_like(r))


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.LayerNorm(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.LayerNorm(output_features)

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


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

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
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNNAtomWise(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(
        self,
        config: ALIGNNAtomWiseConfig = ALIGNNAtomWiseConfig(
            name="alignn_atomwise"
        ),
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification
        self.config = config
        if self.config.gradwise_weight == 0:
            self.config.calculate_gradient = False
        # if self.config.atomwise_weight == 0:
        #    self.config.atomwise_output_features = None
        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()

        if config.extra_features != 0:
            self.readout_feat = AvgPooling()
            # Credit for extra_features work:
            # Gong et al., https://doi.org/10.48550/arXiv.2208.05039
            self.extra_feature_embedding = MLPLayer(
                config.extra_features, config.extra_features
            )
            # print('config.output_features',config.output_features)
            self.fc3 = nn.Linear(
                config.hidden_features + config.extra_features,
                config.output_features,
            )
            self.fc1 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )
            self.fc2 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )

        if config.atomwise_output_features > 0:
            # if config.atomwise_output_features is not None:
            self.fc_atomwise = nn.Linear(
                config.hidden_features, config.atomwise_output_features
            )

        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 1)
            self.softmax = nn.Sigmoid()
            # self.softmax = nn.LogSoftmax(dim=1)
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
        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            # print('features',features,features.shape)
            features = self.extra_feature_embedding(features)
        g = g.local_var()
        result = {}

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)
        r = g.edata["r"]
        if self.config.calculate_gradient:
            r.requires_grad_(True)
        bondlength = torch.norm(r, dim=1)
        # mask = bondlength >= self.config.inner_cutoff
        # bondlength[mask]=float(1.1)
        if self.config.lg_on_fly and len(self.alignn_layers) > 0:
            # re-compute bond angle cosines here to ensure
            # the three-body interactions are fully included
            # in the autograd graph. don't rely on dataloader/caching.
            lg.ndata["r"] = r  # overwrites precomputed r values
            lg.apply_edges(compute_bond_cosines)  # overwrites precomputed h
            z = self.angle_embedding(lg.edata.pop("h"))

        # r = g.edata["r"].clone().detach().requires_grad_(True)
        if self.config.use_cutoff_function:
            # bondlength = cutoff_function_based_edges(
            if self.config.multiply_cutoff:
                c_off = cutoff_function_based_edges(
                    bondlength,
                    inner_cutoff=self.config.inner_cutoff,
                    exponent=self.config.exponent,
                ).unsqueeze(dim=1)

                y = self.edge_embedding(bondlength) * c_off
            else:
                bondlength = cutoff_function_based_edges(
                    bondlength,
                    inner_cutoff=self.config.inner_cutoff,
                    exponent=self.config.exponent,
                )
                y = self.edge_embedding(bondlength)
        else:
            y = self.edge_embedding(bondlength)
        # y = self.edge_embedding(bondlength)
        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)
        # norm-activation-pool-classify
        out = torch.empty(1)
        if self.config.output_features is not None:
            h = self.readout(g, x)
            out = self.fc(h)
            if self.config.extra_features != 0:
                h_feat = self.readout_feat(g, features)
                # print('h_feat',h_feat)
                h = torch.cat((h, h_feat), 1)
                h = self.fc1(h)
                h = self.fc2(h)
                out = self.fc3(h)
                # print('out',out)
            else:
                out = torch.squeeze(out)
        atomwise_pred = torch.empty(1)
        if (
            self.config.atomwise_output_features > 0
            # self.config.atomwise_output_features is not None
            and self.config.atomwise_weight != 0
        ):
            atomwise_pred = self.fc_atomwise(x)
            # atomwise_pred = torch.squeeze(self.readout(g, atomwise_pred))
        forces = torch.empty(1)
        # gradient = torch.empty(1)
        stress = torch.empty(1)

        if self.config.calculate_gradient:
            if self.config.include_pos_deriv:
                # Not tested yet
                g.ndata["coords"].requires_grad_(True)
                dx = [g.ndata["coords"], r]
            else:
                dx = r

            if self.config.energy_mult_natoms:
                en_out = out * g.num_nodes()
            else:
                en_out = out

            # force calculation based on bond displacement vectors
            # autograd gives dE / d{r_{i->j}}
            pair_forces = (
                self.config.grad_multiplier
                * grad(
                    en_out,
                    dx,
                    grad_outputs=torch.ones_like(en_out),
                    create_graph=True,
                    retain_graph=True,
                )[0]
            )
            if self.config.force_mult_natoms:
                pair_forces *= g.num_nodes()

            # construct force_i = dE / d{r_i}
            # reduce over bonds to get forces on each atom

            # force_i contributions from r_{j->i} (in edges)
            g.edata["pair_forces"] = pair_forces
            g.update_all(
                fn.copy_e("pair_forces", "m"), fn.sum("m", "forces_ji")
            )
            if self.config.add_reverse_forces:
                # reduce over reverse edges too!
                # force_i contributions from r_{i->j} (out edges)
                # aggregate pairwise_force_contributions over reversed edges
                rg = dgl.reverse(g, copy_edata=True)
                rg.update_all(
                    fn.copy_e("pair_forces", "m"), fn.sum("m", "forces_ij")
                )

                # combine dE / d(r_{j->i}) and dE / d(r_{i->j})
                forces = torch.squeeze(
                    g.ndata["forces_ji"] - rg.ndata["forces_ij"]
                )
            else:
                forces = torch.squeeze(g.ndata["forces_ji"])

            if self.config.stresswise_weight != 0:
                # Under development, use with caution
                # 1 eV/Angstrom3 = 160.21766208 GPa
                # 1 GPa = 10 kbar
                # Following Virial stress formula, assuming inital velocity = 0
                # Save volume as g.gdta['V']?
                # print('pair_forces',pair_forces.shape)
                # print('r',r.shape)
                # print('g.ndata["V"]',g.ndata["V"].shape)
                if not self.config.batch_stress:
                    # print('Not batch_stress')
                    stress = (
                        -1
                        * 160.21766208
                        * (
                            torch.matmul(r.T, pair_forces)
                            # / (2 * g.edata["V"])
                            / (2 * g.ndata["V"][0])
                        )
                    )
                # print("stress1", stress, stress.shape)
                # print("g.batch_size", g.batch_size)
                else:
                    # print('Using batch_stress')
                    stresses = []
                    count_edge = 0
                    count_node = 0
                    for graph_id in range(g.batch_size):
                        num_edges = g.batch_num_edges()[graph_id]
                        num_nodes = 0
                        st = -1 * (
                            160.21766208
                            * torch.matmul(
                                r[count_edge : count_edge + num_edges].T,
                                pair_forces[
                                    count_edge : count_edge + num_edges
                                ],
                            )
                            / g.ndata["V"][count_node + num_nodes]
                        )

                        count_edge = count_edge + num_edges
                        num_nodes = g.batch_num_nodes()[graph_id]
                        count_node = count_node + num_nodes
                        # print("stresses.append",stresses[-1],stresses[-1].shape)
                        for n in range(num_nodes):
                            stresses.append(st)
                    # stress = (stresses)
                    stress = self.config.stress_multiplier * torch.cat(
                        stresses
                    )
                # print("stress2", stress, stress.shape)
                # virial = (
                #    160.21766208
                #    * 10
                #    * torch.einsum("ij, ik->jk", result["r"], result["dy_dr"])
                #    / 2
                # )  # / ( g.ndata["V"][0])
        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.max(out,dim=1)
            out = self.softmax(out)
        result["out"] = out
        result["grad"] = forces
        result["stresses"] = stress
        result["atomwise_pred"] = atomwise_pred
        # print(result)
        return result
