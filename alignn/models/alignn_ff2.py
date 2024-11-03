from math import pi, sqrt
from typing import Tuple, Union
from torch.autograd import grad
import dgl
import dgl.function as fn
from dgl.nn import AvgPooling
import torch
from typing import Literal
from torch import nn
from torch.nn import functional as F
from alignn.models.utils import (
    RadialBesselFunction,
    RBFExpansion,
    RBFExpansionSmooth,
    BesselExpansion,
    SphericalHarmonicsExpansion,
    FourierExpansion,
    compute_pair_vector_and_distance,
    check_line_graph,
    cutoff_function_based_edges,
    compute_cartesian_coordinates,
    MLPLayer,
)
from alignn.graphs import compute_bond_cosines
from alignn.utils import BaseSettings
from dgl import GCNNorm


class ALIGNNFF2Config(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn_ff2"]
    alignn_layers: int = 2
    gcn_layers: int = 2
    atom_input_features: int = 1
    edge_input_features: int = 64
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 128
    output_features: int = 1
    grad_multiplier: int = -1
    calculate_gradient: bool = True
    atomwise_output_features: int = 0
    graphwise_weight: float = 1.0
    gradwise_weight: float = 1.0
    stresswise_weight: float = 0.0
    atomwise_weight: float = 0.0
    classification: bool = False
    batch_stress: bool = False
    use_cutoff_function: bool = True
    use_penalty: bool = True
    multiply_cutoff: bool = True
    inner_cutoff: float = 4.0  # Angstrom
    stress_multiplier: float = 1.0
    sigma: float = 0.2
    exponent: int = 4
    extra_features: int = 0


class GraphConv(nn.Module):
    """
    Custom Graph Convolution layer with smooth transformations on bond lengths and angles.
    """

    def __init__(
        self, in_feats, out_feats, activation=nn.SiLU(), hidden_features=64
    ):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(
            in_feats, out_feats
        )  # Linear transformation for features
        self.activation = activation
        self.edge_transform = nn.Linear(
            hidden_features, out_feats
        )  # For bond-length based transformation

    def forward(self, g, node_feats, bond_feats):
        """
        Forward pass with bond length handling for smooth transitions.
        """
        # Transform bond (edge) features
        # print('bond_feats',bond_feats.shape)
        bond_feats = self.edge_transform(bond_feats)

        # Message passing: message = transformed edge feature + node feature
        g.ndata["h"] = node_feats
        g.edata["e"] = bond_feats
        g.update_all(
            message_func=fn.u_add_e(
                "h", "e", "m"
            ),  # Add node and edge features
            reduce_func=fn.sum("m", "h"),  # Sum messages for each node
        )

        # Final node feature transformation
        node_feats = self.fc(g.ndata["h"])
        return self.activation(node_feats), bond_feats


class AtomGraphBlock(nn.Module):
    """
    Atom Graph Block that processes atom-centric features and uses GraphConv for updates.
    """

    def __init__(self, in_feats, out_feats, n_layers=2, hidden_features=64):
        super(AtomGraphBlock, self).__init__()
        self.layers = nn.ModuleList(
            [
                GraphConv(
                    in_feats if i == 0 else out_feats,
                    out_feats,
                    hidden_features=hidden_features,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, g, node_feats, bond_feats):
        for layer in self.layers:
            node_feats, bond_feats = layer(g, node_feats, bond_feats)
        return node_feats, bond_feats


class BondGraphBlock(nn.Module):
    """
    Bond Graph Block that applies additional processing on bond-based features.
    """

    def __init__(self, in_feats, out_feats, n_layers=2, hidden_features=64):
        super(BondGraphBlock, self).__init__()
        # self.fc = nn.Linear(in_feats, out_feats)  # Linear transformation for bond features
        # self.activation = activation
        self.layers = nn.ModuleList(
            [
                GraphConv(
                    in_feats if i == 0 else out_feats,
                    out_feats,
                    hidden_features=hidden_features,
                )
                for i in range(n_layers)
            ]
        )

    def forward(self, g, bond_feats, angle_feats):
        """
        Process bond features with smooth transformations.
        """
        # Transform bond features and apply smooth activation
        for layer in self.layers:
            bond_feats, angle_feats = layer(g, bond_feats, angle_feats)
        return bond_feats, angle_feats


class ALIGNNFF2(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(
        self,
        config: ALIGNNFF2Config = ALIGNNFF2Config(name="alignn_ff2"),
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
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
            RadialBesselFunction(
                max_n=config.edge_input_features, cutoff=config.inner_cutoff
            ),
            # RBFExpansionSmooth(num_centers=config.edge_input_features,  cutoff=config.inner_cutoff, sigma=config.sigma),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RadialBesselFunction(
                max_n=config.edge_input_features, cutoff=config.inner_cutoff
            ),
            # RBFExpansionSmooth(num_centers=config.triplet_input_features,  cutoff=1.0, sigma=config.sigma),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.atom_graph_layers = nn.ModuleList(
            [
                AtomGraphBlock(
                    config.hidden_features,
                    config.hidden_features,
                    n_layers=config.gcn_layers,
                    hidden_features=config.hidden_features,
                )
            ]
        )

        self.bond_graph_layers = nn.ModuleList(
            [
                BondGraphBlock(
                    config.hidden_features,
                    config.hidden_features,
                    n_layers=config.gcn_layers,
                    hidden_features=config.hidden_features,
                )
            ]
        )

        self.angle_graph_layers = nn.ModuleList(
            [
                BondGraphBlock(
                    config.hidden_features,
                    config.hidden_features,
                    hidden_features=config.hidden_features,
                )
            ]
        )

        self.gnorm = GCNNorm()

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

    def forward(self, g):
        result = {}
        if self.config.alignn_layers > 0:
            g, lg, lat = g
            lg = lg.local_var()
            # print('lattice',lattice,lattice.shape)
        else:
            g, lat = g

        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            features = self.extra_feature_embedding(features)
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        g = self.gnorm(g)
        # Compute and embed bond lengths
        g.ndata["cart_coords"] = compute_cartesian_coordinates(g, lat)
        if self.config.calculate_gradient:
            g.ndata["cart_coords"].requires_grad_(True)

        r, bondlength = compute_pair_vector_and_distance(g)
        bondlength = torch.norm(r, dim=1)
        y = self.edge_embedding(bondlength)

        # smooth_cutoff = polynomial_cutoff(
        #   bond_expansion, self.config.inner_cutoff, self.config.exponent
        # )
        # bond_expansion *= smooth_cutoff
        if self.config.use_cutoff_function:
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
        out = torch.empty(1)  # graph level output eg energy
        lg = g.line_graph(shared=True)
        lg.ndata["r"] = r
        lg.apply_edges(compute_bond_cosines)
        for atom_graph_layer in self.atom_graph_layers:
            x, y = atom_graph_layer(g, x, y)

        if self.config.output_features is not None:
            h = self.readout(g, x)
            out = self.fc(h)
            if self.config.extra_features != 0:
                h_feat = self.readout_feat(g, features)
                h = torch.cat((h, h_feat), 1)
                h = self.fc1(h)
                h = self.fc2(h)
                out = self.fc3(h)
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

        if self.config.use_penalty:
            penalty_factor = 500.0  # Penalty weight, tune as needed
            penalty_factor = 0.01  # Penalty weight, tune as needed
            penalty_threshold = 1.0  # 1 angstrom

            penalties = torch.where(
                bondlength < penalty_threshold,
                penalty_factor * (penalty_threshold - bondlength),
                torch.zeros_like(bondlength),
            )
            total_penalty = torch.sum(penalties)
            out += total_penalty

        if self.config.calculate_gradient:

            # en_out = torch.sum(out)*g.num_nodes()
            en_out = out  # *g.num_nodes()
            # en_out = (out) *g.num_nodes()
            grad_vars = [g.ndata["cart_coords"]]
            grads = grad(
                en_out,
                grad_vars,
                grad_outputs=torch.ones_like(en_out),
                create_graph=True,
                retain_graph=True,
            )
            forces_out = -1 * grads[0] * g.num_nodes()
            # forces_out = -1*grads[0]
            stresses = torch.eye(3)

        if self.classification:
            out = self.softmax(out)
        result["out"] = out
        result["grad"] = forces_out
        result["stresses"] = stress
        result["atomwise_pred"] = atomwise_pred
        return result
