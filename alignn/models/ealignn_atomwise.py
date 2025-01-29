"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""

from typing import Tuple, Union
from torch.autograd import grad
import dgl
import dgl.function as fn

# import numpy as np
from dgl.nn import AvgPooling
import torch

# from dgl.nn.functional import edge_softmax
from typing import Literal
from torch import nn
from torch.nn import functional as F
from alignn.models.utils import (
    RBFExpansion,
    compute_cartesian_coordinates,
    compute_pair_vector_and_distance,
    MLPLayer,
    lightweight_line_graph,
    remove_net_torque,
)
from alignn.graphs import compute_bond_cosines
from alignn.utils import BaseSettings


class eALIGNNAtomWiseConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["ealignn_atomwise"]
    alignn_layers: int = 2
    gcn_layers: int = 2
    atom_input_features: int = 1
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 64
    output_features: int = 1
    calculate_gradient: bool = True
    atomwise_output_features: int = 0
    graphwise_weight: float = 1.0
    gradwise_weight: float = 1.0
    stresswise_weight: float = 0.0
    atomwise_weight: float = 0.0
    classification: bool = False
    energy_mult_natoms: bool = True  # Make it false for regression only
    remove_torque: bool = True
    inner_cutoff: float = 2.8  # Ansgtrom
    use_penalty: bool = True
    extra_features: int = 0
    penalty_factor: float = 0.1
    penalty_threshold: float = 1
    additional_output_features: int = 0
    additional_output_weight: float = 0
    stress_multiplier: float = 1
    # Extra
    grad_multiplier: int = -1
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    force_mult_natoms: bool = False
    energy_mult_natoms: bool = True
    include_pos_deriv: bool = False
    use_cutoff_function: bool = False
    add_reverse_forces: bool = True  # will make True as default soon
    lg_on_fly: bool = True  # will make True as default soon
    batch_stress: bool = True
    multiply_cutoff: bool = False
    exponent: int = 5


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


class eALIGNNAtomWise(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(
        self,
        config: eALIGNNAtomWiseConfig = eALIGNNAtomWiseConfig(
            name="ealignn_atomwise"
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

        if config.additional_output_features:
            self.fc_additional_output = nn.Linear(
                config.hidden_features, config.additional_output_features
            )
        if self.classification:
            self.fc = nn.Linear(config.hidden_features, 1)
            self.softmax = nn.Sigmoid()
            # self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc = nn.Linear(config.hidden_features, config.output_features)

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        # print('g',g,len(g))
        if len(g) == 3:
            g, lg, lat = g
            lg = lg.local_var()
        else:
            g, lat = g

        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            features = self.extra_feature_embedding(features)
        g = g.local_var()
        result = {}

        x = g.ndata.pop("atom_features")

        x = self.atom_embedding(x)
        r = g.edata["r"]
        if self.config.calculate_gradient:
            r.requires_grad_(True)
        bondlength = torch.norm(r, dim=1)
        if (self.config.alignn_layers) > 0:
            g.ndata["cart_coords"] = compute_cartesian_coordinates(g, lat)
            g.ndata["cart_coords"].requires_grad_(True)
            r, bondlength = compute_pair_vector_and_distance(g)
            g.edata["bondlength"] = bondlength
            g = lightweight_line_graph(
                g,
                feature_name="bondlength",
                filter_condition=lambda x: torch.gt(
                    x, self.config.inner_cutoff
                ),
            )
            r, bondlength = compute_pair_vector_and_distance(g)
            lg = g.line_graph(shared=True)
            lg.ndata["r"] = r  # overwrites precomputed r values
            lg.apply_edges(compute_bond_cosines)  # overwrites precomputed h
            z = self.angle_embedding(lg.edata.pop("h"))

        y = self.edge_embedding(bondlength)
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)
        # norm-activation-pool-classify
        out = torch.empty(1)
        additional_out = torch.empty(1)
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
            if self.config.additional_output_features > 0:
                additional_out = self.fc_additional_output(h)

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
        natoms = torch.tensor([gg.num_nodes() for gg in dgl.unbatch(g)]).to(
            g.device
        )
        en_out = out
        if self.config.energy_mult_natoms:
            en_out = out * natoms  # g.num_nodes()
        if self.config.use_penalty:
            penalty_factor = (
                self.config.penalty_factor
            )  # Penalty weight, tune as needed
            penalty_threshold = self.config.penalty_threshold  # 1 angstrom

            penalties = torch.where(
                bondlength < penalty_threshold,
                penalty_factor * (penalty_threshold - bondlength),
                torch.zeros_like(bondlength),
            )
            total_penalty = torch.sum(penalties)
            en_out += total_penalty

        if self.config.calculate_gradient:
            # force calculation based on bond displacement vectors
            # autograd gives dE / d{r_{i->j}}
            pair_forces = (
                -1
                * grad(
                    en_out,
                    r,
                    grad_outputs=torch.ones_like(en_out),
                    create_graph=True,
                    retain_graph=True,
                )[0]
            )
            pair_forces *= g.num_nodes()

            g.edata["pair_forces"] = pair_forces
            g.update_all(
                fn.copy_e("pair_forces", "m"), fn.sum("m", "forces_ji")
            )
            rg = dgl.reverse(g, copy_edata=True)
            rg.update_all(
                fn.copy_e("pair_forces", "m"), fn.sum("m", "forces_ij")
            )

            # combine dE / d(r_{j->i}) and dE / d(r_{i->j})
            forces = torch.squeeze(
                g.ndata["forces_ji"] - rg.ndata["forces_ij"]
            )
            if self.config.remove_torque:
                # print('forces1',forces,forces.shape)
                # print('natoms',natoms,natoms.shape)
                forces = remove_net_torque(g, forces, natoms)
                # print('forces2',forces,forces.shape)

            if self.config.stresswise_weight != 0:
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
                            pair_forces[count_edge : count_edge + num_edges],
                        )
                        / g.ndata["V"][count_node + num_nodes]
                    )

                    count_edge = count_edge + num_edges
                    num_nodes = g.batch_num_nodes()[graph_id]
                    count_node = count_node + num_nodes
                    stresses.append(st)
                stress = self.config.stress_multiplier * torch.stack(stresses)
        if self.classification:
            out = self.softmax(out)
        # print('out',out)
        result["out"] = out
        result["additional"] = additional_out
        result["grad"] = forces
        result["stresses"] = stress
        result["atomwise_pred"] = atomwise_pred
        return result
