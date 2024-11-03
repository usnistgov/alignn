from torch.autograd import grad
from math import pi
from typing import Any, Callable, Literal, cast
from collections.abc import Sequence
from torch.nn import Linear, Module
from jarvis.core.specie import get_element_full_names
import dgl.function as fn
from torch import Tensor, nn
from jarvis.core.atoms import Atoms
from alignn.graphs import Graph
from enum import Enum
import dgl
from pathlib import Path
import torch
from dgl import readout_nodes
import inspect
import json
import os
from alignn.utils import BaseSettings
from alignn.models.utils import (
    get_ewald_sum,
    get_atomic_repulsion,
    FourierExpansion,
    RadialBesselFunction,
    prune_edges_by_features,
    _create_directed_line_graph,
    compute_theta,
    create_line_graph,
    compute_pair_vector_and_distance,
    polynomial_cutoff,
)

torch.autograd.detect_anomaly()
DEFAULT_ELEMENTS = list(get_element_full_names().keys())


class ALIGNNeFFConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn_eff"]
    alignn_layers: int = 4
    calculate_gradient: bool = True
    output_features: int = 1
    atomwise_output_features: int = 0
    graphwise_weight: float = 1.0
    gradwise_weight: float = 20.0
    stresswise_weight: float = 0.0
    atomwise_weight: float = 0.0
    batch_stress: bool = True


class EFFLineGraphConv(nn.Module):

    def __init__(
        self,
        node_update_func: Module,
        node_out_func: Module,
        edge_update_func: Module | None,
        node_weight_func: Module | None,
    ):
        """
        Args:
            node_update_func: Update function for message between nodes (bonds)
            node_out_func: Output function for nodes (bonds), after message aggregation
            edge_update_func: edge update function (for angle features)
            node_weight_func: layer node weight function.
        """
        super().__init__()

        self.node_update_func = node_update_func
        self.node_out_func = node_out_func
        self.node_weight_func = node_weight_func
        self.edge_update_func = edge_update_func

    @classmethod
    def from_dims(
        cls,
        node_dims: list[int],
        edge_dims: list[int] | None = None,
        activation: Module | None = None,
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        node_weight_input_dims: int = 0,
    ):
        norm_kwargs = (
            {"batched_field": "edge"} if normalization == "graph" else None
        )

        node_update_func = GatedMLP_norm(
            in_feats=node_dims[0],
            dims=node_dims[1:],
            activation=activation,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            norm_kwargs=norm_kwargs,
        )
        node_out_func = nn.Linear(
            in_features=node_dims[-1], out_features=node_dims[-1], bias=False
        )

        node_weight_func = (
            nn.Linear(node_weight_input_dims, node_dims[-1])
            if node_weight_input_dims > 0
            else None
        )
        edge_update_func = (
            GatedMLP_norm(
                in_feats=edge_dims[0],
                dims=edge_dims[1:],
                activation=activation,
                normalization=normalization,
                normalize_hidden=normalize_hidden,
                norm_kwargs=norm_kwargs,
            )
            if edge_dims is not None
            else None
        )

        return cls(
            node_update_func=node_update_func,
            node_out_func=node_out_func,
            edge_update_func=edge_update_func,
            node_weight_func=node_weight_func,
        )

    def _edge_udf(self, edges: dgl.udf.EdgeBatch) -> dict[str, Tensor]:
        """Edge user defined update function.

        Update angle features (edges in bond graph)

        Args:
            edges: edge batch

        Returns:
            edge_update: edge features update
        """
        bonds_i = edges.src["features"]  # first bonds features
        bonds_j = edges.dst["features"]  # second bonds features
        angle_ij = edges.data["features"]
        atom_ij = edges.data["aux_features"]  # center atom features
        inputs = torch.hstack([bonds_i, angle_ij, atom_ij, bonds_j])
        messages_ij = self.edge_update_func(inputs, edges._graph)  # type: ignore
        return {"feat_update": messages_ij}

    def edge_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Perform edge update -> update angle features.

        Args:
            graph: bond graph (line graph of atom graph)

        Returns:
            edge_update: edge features update
        """
        graph.apply_edges(self._edge_udf)
        edge_update = graph.edata["feat_update"]
        return edge_update

    def node_update_(
        self, graph: dgl.DGLGraph, shared_weights: Tensor | None
    ) -> Tensor:
        """Perform node update -> update bond features.

        Args:
            graph: bond graph (line graph of atom graph)
            shared_weights: node message shared weights

        Returns:
            node_update: bond features update
        """
        src, dst = graph.edges()
        bonds_i = graph.ndata["features"][src]  # first bond feature
        bonds_j = graph.ndata["features"][dst]  # second bond feature
        angle_ij = graph.edata["features"]
        atom_ij = graph.edata["aux_features"]  # center atom features
        inputs = torch.hstack([bonds_i, angle_ij, atom_ij, bonds_j])

        messages = self.node_update_func(inputs, graph)

        # smooth out messages with layer-wise weights
        if self.node_weight_func is not None:
            rbf = graph.ndata["bond_expansion"]
            weights = self.node_weight_func(rbf)
            weights_i, weights_j = weights[src], weights[dst]
            messages = messages * weights_i * weights_j

        # smooth out messages with shared weights
        if shared_weights is not None:
            weights_i, weights_j = shared_weights[src], shared_weights[dst]
            messages = messages * weights_i * weights_j

        # message passing
        graph.edata["message"] = messages
        graph.update_all(
            fn.copy_e("message", "message"), fn.sum("message", "feat_update")
        )

        # update nodes
        node_update = self.node_out_func(
            graph.ndata["feat_update"]
        )  # the bond update

        return node_update

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Tensor,
        edge_features: Tensor,
        aux_edge_features: Tensor,
        shared_node_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        with graph.local_scope():
            graph.ndata["features"] = node_features
            graph.edata["features"] = edge_features
            graph.edata["aux_features"] = aux_edge_features

            # node (bond) update
            node_update = self.node_update_(graph, shared_node_weights)
            new_node_features = node_features + node_update
            graph.ndata["features"] = new_node_features

            # edge (angle) update (should angle update be done before node update?)
            if self.edge_update_func is not None:
                edge_update = self.edge_update_(graph)
                new_edge_features = edge_features + edge_update
                graph.edata["features"] = new_edge_features
            else:
                new_edge_features = edge_features

        return new_node_features, new_edge_features


class GatedMLP_norm(nn.Module):
    """An implementation of a Gated multi-layer perceptron constructed with MLP."""

    def __init__(
        self,
        in_feats: int,
        dims: Sequence[int],
        activation: nn.Module | None = None,
        activate_last: bool = True,
        use_bias: bool = True,
        bias_last: bool = True,
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        norm_kwargs: dict[str, Any] | None = None,
    ):
        """:param in_feats: Dimension of input features.
        :param dims: Architecture of neural networks.
        :param activation: non-linear activation module.
        :param activate_last: Whether applying activation to last layer or not.
        :param use_bias: Whether to use a bias in linear layers.
        :param bias_last: Whether applying bias to last layer or not.
        :param normalization: normalization name.
        :param normalize_hidden: Whether to normalize output of hidden layers.
        :param norm_kwargs: Keyword arguments for normalization layer.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims]
        self._depth = len(dims)
        self.use_bias = use_bias
        self.activate_last = activate_last

        activation = activation if activation is not None else nn.SiLU()
        self.layers = MLP_norm(
            self.dims,
            activation=activation,
            activate_last=True,
            use_bias=use_bias,
            bias_last=bias_last,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            norm_kwargs=norm_kwargs,
        )
        self.gates = MLP_norm(
            self.dims,
            activation,
            activate_last=False,
            use_bias=use_bias,
            bias_last=bias_last,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            norm_kwargs=norm_kwargs,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, graph=None) -> torch.Tensor:
        return self.layers(inputs, graph) * self.sigmoid(
            self.gates(inputs, graph)
        )


class EFFBondGraphBlock(nn.Module):
    """A EFF atom graph block as a sequence of operations involving a message passing layer over the bond graph."""

    def __init__(
        self,
        num_atom_feats: int,
        num_bond_feats: int,
        num_angle_feats: int,
        activation: Module,
        bond_hidden_dims: Sequence[int],
        angle_hidden_dims: Sequence[int] | None,
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        rbf_order: int = 0,
        bond_dropout: float = 0.0,
        angle_dropout: float = 0.0,
    ):
        """.

        Args:
            num_atom_feats: number of atom features
            num_bond_feats: number of bond features
            num_angle_feats: number of angle features
            activation: activation function
            bond_hidden_dims: dimensions of hidden layers of bond graph convolution
            angle_hidden_dims: dimensions of hidden layers of angle update function
                Default = None
            normalization: Normalization type to use in update functions. either "graph" or "layer"
                If None, no normalization is applied.
                Default = None
            normalize_hidden: Whether to normalize hidden features.
                Default = False
            rbf_order (int): RBF order specifying input dimensions for linear layer
                specifying message weights. If 0, no layer-wise weights are used.
                Default = 0
            bond_dropout (float): dropout probability for bond graph convolution.
                Default = 0.0
            angle_dropout (float): dropout probability for angle update function.
                Default = 0.0
        """
        super().__init__()

        node_input_dim = 2 * num_bond_feats + num_angle_feats + num_atom_feats
        node_dims = [node_input_dim, *bond_hidden_dims, num_bond_feats]
        edge_dims = (
            [node_input_dim, *angle_hidden_dims, num_angle_feats]
            if angle_hidden_dims is not None
            else None
        )

        self.conv_layer = EFFLineGraphConv.from_dims(
            node_dims=node_dims,
            edge_dims=edge_dims,
            activation=activation,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            node_weight_input_dims=rbf_order,
        )

        self.bond_dropout = (
            nn.Dropout(bond_dropout) if bond_dropout > 0.0 else nn.Identity()
        )
        self.angle_dropout = (
            nn.Dropout(angle_dropout) if angle_dropout > 0.0 else nn.Identity()
        )

    def forward(
        self,
        graph: dgl.DGLGraph,
        atom_features: Tensor,
        bond_features: Tensor,
        angle_features: Tensor,
        shared_node_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Perform convolution in BondGraph to update bond and angle features.

        Args:
            graph: bond graph (line graph of atom graph)
            atom_features: atom features
            bond_features: bond features
            angle_features: concatenated center atom and angle features
            shared_node_weights: shared node message weights

        Returns:
            tuple: update bond features, update angle features
        """
        node_features = bond_features[graph.ndata["bond_index"]]
        edge_features = angle_features
        aux_edge_features = atom_features[graph.edata["center_atom_index"]]

        bond_features_, angle_features = self.conv_layer(
            graph,
            node_features,
            edge_features,
            aux_edge_features,
            shared_node_weights,
        )

        bond_features_ = self.bond_dropout(bond_features_)
        angle_features = self.angle_dropout(angle_features)

        bond_features[graph.ndata["bond_index"]] = bond_features_

        return bond_features, angle_features


class EFFGraphConv(nn.Module):
    """A EFF atom graph convolution layer in DGL."""

    def __init__(
        self,
        node_update_func: Module,
        node_out_func: Module,
        edge_update_func: Module | None,
        node_weight_func: Module | None,
        edge_weight_func: Module | None,
        state_update_func: Module | None,
    ):
        """
        Args:
            node_update_func: Update function for message between nodes (atoms)
            node_out_func: Output function for nodes (atoms), after message aggregation
            edge_update_func: Update function for edges (bonds). If None is given, the
                edges are not updated.
            node_weight_func: Weight function for radial basis functions.
                If None is given, no layer-wise weights will be used.
            edge_weight_func: Weight function for radial basis functions
                If None is given, no layer-wise weights will be used.
            state_update_func: Update function for state feats.
        """
        super().__init__()
        self.include_state = state_update_func is not None
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_out_func = node_out_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    @classmethod
    def from_dims(
        cls,
        activation: Module,
        node_dims: Sequence[int],
        edge_dims: Sequence[int] | None = None,
        state_dims: Sequence[int] | None = None,
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        rbf_order: int = 0,
    ):
        """Create a EFFAtomGraphConv layer from dimensions.

        Args:
            activation: activation function
            node_dims: NN architecture for node update function given as a list of
                dimensions of each layer.
            edge_dims: NN architecture for edge update function given as a list of
                dimensions of each layer.
                Default = None
            state_dims: NN architecture for state update function given as a list of
                dimensions of each layer.
                Default = None
            normalization: Normalization type to use in update functions. either "graph" or "layer"
                If None, no normalization is applied.
                Default = None
            normalize_hidden: Whether to normalize hidden features.
                Default = False
            rbf_order (int): RBF order specifying input dimensions for linear layer
                specifying message weights. If 0, no layer-wise weights are used.
                Default = 0

        Returns:
            EFFAtomGraphConv
        """
        norm_kwargs = (
            {"batched_field": "edge"} if normalization == "graph" else None
        )

        node_update_func = GatedMLP_norm(
            in_feats=node_dims[0],
            dims=node_dims[1:],
            activation=activation,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            norm_kwargs=norm_kwargs,
        )
        node_out_func = nn.Linear(
            in_features=node_dims[-1], out_features=node_dims[-1], bias=False
        )
        node_weight_func = (
            nn.Linear(
                in_features=rbf_order, out_features=node_dims[-1], bias=False
            )
            if rbf_order > 0
            else None
        )
        edge_update_func = (
            GatedMLP_norm(
                in_feats=edge_dims[0],
                dims=edge_dims[1:],
                activation=activation,
                normalization=normalization,
                normalize_hidden=normalize_hidden,
                norm_kwargs=norm_kwargs,
            )
            if edge_dims is not None
            else None
        )
        edge_weight_func = (
            nn.Linear(
                in_features=rbf_order, out_features=edge_dims[-1], bias=False
            )
            if rbf_order > 0 and edge_dims is not None
            else None
        )
        state_update_func = (
            MLP(
                state_dims,
                activation,
                activate_last=True,
            )
            if state_dims is not None
            else None
        )

        return cls(
            node_update_func=node_update_func,
            node_out_func=node_out_func,
            edge_update_func=edge_update_func,
            node_weight_func=node_weight_func,
            edge_weight_func=edge_weight_func,
            state_update_func=state_update_func,
        )

    def _edge_udf(self, edges: dgl.udf.EdgeBatch) -> dict[str, Tensor]:
        """Edge user defined update function.

        Update for bond features (edges) in atom graph.

        Args:
            edges: edges in atom graph (ie bonds)

        Returns:
            edge_update: edge features update
        """
        atom_i = edges.src["features"]  # first atom features
        atom_j = edges.dst["features"]  # second atom features
        bond_ij = edges.data["features"]  # bond features
        if self.include_state:
            global_state = edges.data["global_state"]
            inputs = torch.hstack([atom_i, bond_ij, atom_j, global_state])
        else:
            inputs = torch.hstack([atom_i, bond_ij, atom_j])

        edge_update = self.edge_update_func(inputs, edges._graph)  # type: ignore
        if self.edge_weight_func is not None:
            rbf = edges.data["bond_expansion"]
            rbf = rbf.float()
            edge_update = edge_update * self.edge_weight_func(rbf)

        return {"feat_update": edge_update}

    def edge_update_(
        self, graph: dgl.DGLGraph, shared_weights: Tensor | None
    ) -> Tensor:
        """Perform edge update -> bond features.

        Args:
            graph: atom graph
            shared_weights: atom graph edge weights shared between convolution layers

        Returns:
            edge_update: edge features update
        """
        graph.apply_edges(self._edge_udf)
        edge_update = graph.edata["feat_update"]
        if shared_weights is not None:
            edge_update = edge_update * shared_weights
        return edge_update

    def node_update_(
        self, graph: dgl.DGLGraph, shared_weights: Tensor | None
    ) -> Tensor:
        """Perform node update -> atom features.

        Args:
            graph: DGL atom graph
            shared_weights: node message shared weights

        Returns:
            node_update: updated node features
        """
        src, dst = graph.edges()
        atom_i = graph.ndata["features"][src]  # first atom features
        atom_j = graph.ndata["features"][dst]  # second atom features
        bond_ij = graph.edata["features"]  # bond features

        if self.include_state:
            global_state = graph.edata["global_state"]
            inputs = torch.hstack([atom_i, bond_ij, atom_j, global_state])
        else:
            inputs = torch.hstack([atom_i, bond_ij, atom_j])

        messages = self.node_update_func(inputs, graph)

        # smooth out the messages with layer-wise weights
        if self.node_weight_func is not None:
            rbf = graph.edata["bond_expansion"]
            rbf = rbf.float()
            messages = messages * self.node_weight_func(rbf)

        # smooth out the messages with shared weights
        if shared_weights is not None:
            messages = messages * shared_weights

        # message passing
        graph.edata["message"] = messages
        graph.update_all(
            fn.copy_e("message", "message"), fn.sum("message", "feat_update")
        )

        # update nodes
        node_update = self.node_out_func(
            graph.ndata["feat_update"]
        )  # the bond update

        return node_update

    def state_update_(self, graph: dgl.DGLGraph, state_attr: Tensor) -> Tensor:
        """Perform attribute (global state) update.

        Args:
            graph: atom graph
            state_attr: global state features

        Returns:
            state_update: state features update
        """
        node_avg = dgl.readout_nodes(graph, feat="features", op="mean")
        inputs = torch.hstack([state_attr, node_avg])
        state_attr = self.state_update_func(inputs)  # type: ignore
        return state_attr

    def forward(
        self,
        graph: dgl.DGLGraph,
        node_features: Tensor,
        edge_features: Tensor,
        state_attr: Tensor,
        shared_node_weights: Tensor | None,
        shared_edge_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform sequence of edge->node->states updates.

        Args:
            graph: atom graph
            node_features: node features
            edge_features: edge features
            state_attr: state attributes
            shared_node_weights: shared node message weights
            shared_edge_weights: shared edge message weights

        Returns:
            tuple: updated node features, updated edge features, updated state attributes
        """
        with graph.local_scope():
            graph.ndata["features"] = node_features
            graph.edata["features"] = edge_features

            if self.include_state:
                graph.edata["global_state"] = dgl.broadcast_edges(
                    graph, state_attr
                )

            if self.edge_update_func is not None:
                edge_update = self.edge_update_(graph, shared_edge_weights)
                new_edge_features = edge_features + edge_update
                graph.edata["features"] = new_edge_features
            else:
                new_edge_features = edge_features

            node_update = self.node_update_(graph, shared_node_weights)
            new_node_features = node_features + node_update
            graph.ndata["features"] = new_node_features

            if self.include_state:
                state_attr = self.state_update_(graph, state_attr)  # type: ignore

        return new_node_features, new_edge_features, state_attr


class EFFAtomGraphBlock(nn.Module):
    """
    A EFF atom graph block as a sequence of operations
    involving a message passing layer over the atom graph.
    """

    def __init__(
        self,
        num_atom_feats: int,
        num_bond_feats: int,
        activation: Module,
        atom_hidden_dims: Sequence[int],
        bond_hidden_dims: Sequence[int] | None = None,
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        num_state_feats: int | None = None,
        rbf_order: int = 0,
        dropout: float = 0.0,
    ):
        """.

        Args:
            num_atom_feats: number of atom features
            num_bond_feats: number of bond features
            activation: activation function
            atom_hidden_dims: dimensions of atom convolution hidden layers
            bond_hidden_dims: dimensions of bond update hidden layers.
            normalization: Normalization type to use in update functions. either "graph" or "layer"
                If None, no normalization is applied.
                Default = None
            normalize_hidden: Whether to normalize hidden features.
                Default = False
            num_state_feats: number of state features if self.include_state is True
                Default = None
            rbf_order: RBF order specifying input dimensions for linear layer
                specifying message weights. If 0, no layer-wise weights are used.
                Default = False
            dropout: dropout probability.
                Default = 0.0
        """
        super().__init__()

        node_input_dim = 2 * num_atom_feats + num_bond_feats
        if num_state_feats is not None:
            node_input_dim += num_state_feats
            state_dims = [
                num_atom_feats + num_state_feats,
                *atom_hidden_dims,
                num_state_feats,
            ]
        else:
            state_dims = None
        node_dims = [node_input_dim, *atom_hidden_dims, num_atom_feats]
        edge_dims = (
            [node_input_dim, *bond_hidden_dims, num_bond_feats]
            if bond_hidden_dims is not None
            else None
        )

        self.conv_layer = EFFGraphConv.from_dims(
            activation=activation,
            node_dims=node_dims,
            edge_dims=edge_dims,
            state_dims=state_dims,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            rbf_order=rbf_order,
        )

        if normalization == "graph":
            self.atom_norm = GraphNorm(num_atom_feats, batched_field="node")
            self.bond_norm = GraphNorm(num_bond_feats, batched_field="edge")
        elif normalization == "layer":
            self.atom_norm = LayerNorm(num_atom_feats)
            self.bond_norm = LayerNorm(num_bond_feats)
        else:
            self.atom_norm = None
            self.bond_norm = None

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        graph: dgl.DGLGraph,
        atom_features: Tensor,
        bond_features: Tensor,
        state_attr: Tensor,
        shared_node_weights: Tensor | None,
        shared_edge_weights: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Perform sequence of bond(optional)->atom->states(optional) updates.

        Args:
            graph: atom graph
            atom_features: node features
            bond_features: edge features
            state_attr: state attributes
            shared_node_weights: node message weights shared amongst layers
            shared_edge_weights: edge message weights shared amongst layers
        """
        atom_features, bond_features, state_attr = self.conv_layer(
            graph=graph,
            node_features=atom_features,
            edge_features=bond_features,
            state_attr=state_attr,
            shared_node_weights=shared_node_weights,
            shared_edge_weights=shared_edge_weights,
        )
        # move skip connections here? dropout before skip connections?
        atom_features = self.dropout(atom_features)
        bond_features = self.dropout(bond_features)
        if self.atom_norm is not None:
            atom_features = self.atom_norm(atom_features, graph)
        if self.bond_norm is not None:
            bond_features = self.bond_norm(bond_features, graph)
        if state_attr is not None:
            state_attr = self.dropout(state_attr)

        return atom_features, bond_features, state_attr


class MLP_norm(nn.Module):
    """Multi-layer perceptron with normalization layer."""

    def __init__(
        self,
        dims: list[int],
        activation: nn.Module | None = None,
        activate_last: bool = False,
        use_bias: bool = True,
        bias_last: bool = True,
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        norm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Args:
            dims: Dimensions of each layer of MLP.
            activation: activation: Activation function.
            activate_last: Whether to apply activation to last layer.
            use_bias: Whether to use bias.
            bias_last: Whether to apply bias to last layer.
            normalization: normalization name. "graph" or "layer"
            normalize_hidden: Whether to normalize output of hidden layers.
            norm_kwargs: Keyword arguments for normalization layer.
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = nn.ModuleList()
        self.norm_layers = (
            nn.ModuleList() if normalization in ("graph", "layer") else None
        )
        self.activation = (
            activation if activation is not None else nn.Identity()
        )
        self.activate_last = activate_last
        self.normalize_hidden = normalize_hidden
        norm_kwargs = norm_kwargs or {}
        norm_kwargs = cast(dict, norm_kwargs)

        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=use_bias))
                if normalize_hidden and self.norm_layers is not None:
                    if normalization == "graph":
                        self.norm_layers.append(
                            GraphNorm(out_dim, **norm_kwargs)
                        )
                    elif normalization == "layer":
                        self.norm_layers.append(
                            LayerNorm(out_dim, **norm_kwargs)
                        )
            else:
                self.layers.append(
                    Linear(in_dim, out_dim, bias=use_bias and bias_last)
                )
                if self.norm_layers is not None:
                    if normalization == "graph":
                        self.norm_layers.append(
                            GraphNorm(out_dim, **norm_kwargs)
                        )
                    elif normalization == "layer":
                        self.norm_layers.append(
                            LayerNorm(out_dim, **norm_kwargs)
                        )

    def forward(self, inputs: torch.Tensor, g=None) -> torch.Tensor:
        """Applies all layers in turn.

        Args:
            inputs: input feature tensor.
            g: graph of model, needed for graph normalization

        Returns:
            output feature tensor.
        """
        x = inputs
        for i in range(self._depth - 1):
            x = self.layers[i](x)
            if self.norm_layers is not None and self.normalize_hidden:
                x = self.norm_layers[i](x, g)
            x = self.activation(x)

        x = self.layers[-1](x)
        if self.norm_layers is not None:
            x = self.norm_layers[-1](x, g)
        if self.activate_last:
            x = self.activation(x)
        return x


class ActivationFunction(Enum):
    """Enumeration of optional activation functions."""

    swish = nn.SiLU
    # sigmoid = nn.Sigmoid
    # tanh = nn.Tanh
    # softplus = nn.Softplus
    # softplus2 = SoftPlus2
    # softexp = SoftExponential


class ALIGNNeFF(nn.Module):
    """Main EFF model."""

    __version__ = 1

    def __init__(
        self,
        config: ALIGNNeFFConfig = ALIGNNeFFConfig(name="alignn_eff"),
        element_types: tuple[str, ...] | None = None,
        dim_atom_embedding: int = 64,
        dim_bond_embedding: int = 64,
        dim_angle_embedding: int = 64,
        dim_state_embedding: int | None = None,
        dim_state_feats: int | None = None,
        non_linear_bond_embedding: bool = False,
        non_linear_angle_embedding: bool = False,
        cutoff: float = 4.0,
        threebody_cutoff: float = 3.0,
        cutoff_exponent: int = 5,
        max_n: int = 9,
        max_f: int = 4,
        learn_basis: bool = True,
        num_blocks: int = 4,
        shared_bond_weights: (
            Literal["bond", "three_body_bond", "both"] | None
        ) = "both",
        layer_bond_weights: (
            Literal["bond", "three_body_bond", "both"] | None
        ) = None,
        atom_conv_hidden_dims: Sequence[int] = (64,),
        bond_update_hidden_dims: Sequence[int] | None = None,
        bond_conv_hidden_dims: Sequence[int] = (64,),
        angle_update_hidden_dims: Sequence[int] | None = (),
        conv_dropout: float = 0.0,
        final_mlp_type: Literal["gated", "mlp"] = "mlp",
        final_hidden_dims: Sequence[int] = (64, 64),
        final_dropout: float = 0.0,
        pooling_operation: Literal["sum", "mean"] = "sum",
        readout_field: Literal[
            "atom_feat", "bond_feat", "angle_feat"
        ] = "atom_feat",
        activation_type: str = "swish",
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        is_intensive: bool = False,
        num_targets: int = 1,
        num_site_targets: int = 1,
        task_type: Literal["regression", "classification"] = "regression",
    ):
        super().__init__()

        # self.save_args(locals(), kwargs)

        activation: nn.Module = ActivationFunction[activation_type].value()

        element_types = element_types or DEFAULT_ELEMENTS

        # basis expansions for bond lengths, triple interaction bond lengths and angles
        self.bond_expansion = RadialBesselFunction(
            max_n=max_n, cutoff=cutoff, learnable=learn_basis
        )
        self.threebody_bond_expansion = RadialBesselFunction(
            max_n=max_n, cutoff=threebody_cutoff, learnable=learn_basis
        )
        self.angle_expansion = FourierExpansion(
            max_f=max_f, learnable=learn_basis
        )

        # embedding block for atom, bond, angle, and optional state features
        self.include_states = dim_state_feats is not None
        self.state_embedding = (
            nn.Embedding(dim_state_feats, dim_state_embedding)
            if self.include_states
            else None
        )
        self.atom_embedding = nn.Embedding(
            len(element_types), dim_atom_embedding
        )

        # self.atom_embedding = MLP_norm(
        #    1, dim_state_embedding
        # )

        self.bond_embedding = MLP_norm(
            [max_n, dim_bond_embedding],
            activation=activation,
            activate_last=non_linear_bond_embedding,
            bias_last=False,
        )
        self.angle_embedding = MLP_norm(
            [2 * max_f + 1, dim_angle_embedding],
            activation=activation,
            activate_last=non_linear_angle_embedding,
            bias_last=False,
        )

        # shared message bond distance smoothing weights
        self.atom_bond_weights = (
            nn.Linear(max_n, dim_atom_embedding, bias=False)
            if shared_bond_weights in ["bond", "both"]
            else None
        )
        self.bond_bond_weights = (
            nn.Linear(max_n, dim_bond_embedding, bias=False)
            if shared_bond_weights in ["bond", "both"]
            else None
        )
        self.threebody_bond_weights = (
            nn.Linear(max_n, dim_bond_embedding, bias=False)
            if shared_bond_weights in ["three_body_bond", "both"]
            else None
        )

        # operations involving the graph (i.e. atom graph) to update atom and bond features
        self.atom_graph_layers = nn.ModuleList(
            [
                EFFAtomGraphBlock(
                    num_atom_feats=dim_atom_embedding,
                    num_bond_feats=dim_bond_embedding,
                    atom_hidden_dims=atom_conv_hidden_dims,
                    bond_hidden_dims=bond_update_hidden_dims,
                    num_state_feats=dim_state_embedding,
                    activation=activation,
                    normalization=normalization,
                    normalize_hidden=normalize_hidden,
                    dropout=conv_dropout,
                    rbf_order=0,
                )
                for _ in range(num_blocks)
            ]
        )

        # operations involving the line graph (i.e. bond graph) to update bond and angle features
        self.bond_graph_layers = nn.ModuleList(
            [
                EFFBondGraphBlock(
                    num_atom_feats=dim_atom_embedding,
                    num_bond_feats=dim_bond_embedding,
                    num_angle_feats=dim_angle_embedding,
                    bond_hidden_dims=bond_conv_hidden_dims,
                    angle_hidden_dims=angle_update_hidden_dims,
                    activation=activation,
                    normalization=normalization,
                    normalize_hidden=normalize_hidden,
                    bond_dropout=conv_dropout,
                    angle_dropout=conv_dropout,
                    rbf_order=0,
                )
                for _ in range(num_blocks - 1)
            ]
        )

        self.sitewise_readout = (
            nn.Linear(dim_atom_embedding, num_site_targets)
            if num_site_targets > 0
            else lambda x: None
        )
        print("final_mlp_type", final_mlp_type)
        input_dim = (
            dim_atom_embedding
            if readout_field == "node_feat"
            else dim_bond_embedding
        )

        self.final_layer = MLP_norm(
            dims=[input_dim, *final_hidden_dims, num_targets],
            activation=activation,
            activate_last=False,
        )

        self.element_types = element_types
        self.max_n = max_n
        self.max_f = max_f
        self.cutoff = cutoff
        self.cutoff_exponent = cutoff_exponent
        self.three_body_cutoff = threebody_cutoff

        self.n_blocks = num_blocks
        self.readout_operation = pooling_operation
        self.readout_field = readout_field
        self.readout_type = final_mlp_type

        self.task_type = task_type
        self.is_intensive = is_intensive

    def forward(
        self,
        g,
        state_attr: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the model.

        Args:
            g (dgl.DGLGraph): Input g.
            state_attr (torch.Tensor, optional): State features. Defaults to None.
            l_g (dgl.DGLGraph, optional): Line graph. Defaults to None and is computed internally.

        Returns:
            torch.Tensor: Model output.
        """
        g, l_g, lat = g
        st = lat.new_zeros([g.batch_size, 3, 3])
        st.requires_grad_(True)
        lattice = lat @ (torch.eye(3, device=lat.device) + st)
        g.edata["lattice"] = torch.repeat_interleave(
            lattice, g.batch_num_edges(), dim=0
        )
        g.edata["pbc_offshift"] = (
            g.edata["images"].unsqueeze(dim=-1) * g.edata["lattice"]
        ).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1)
            * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        g.ndata["pos"].requires_grad_(True)

        # compute bond vectors and distances and add to g, needs to be computed here to register gradients
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec.to(g.device)
        g.edata["bond_dist"] = bond_dist.to(g.device)
        bond_expansion = self.bond_expansion(bond_dist)
        smooth_cutoff = polynomial_cutoff(
            bond_expansion, self.cutoff, self.cutoff_exponent
        )
        g.edata["bond_expansion"] = smooth_cutoff * bond_expansion

        # create bond graph (line graoh) with necessary node and edge data
        # print("self.readout_field", self.readout_field)
        bond_graph = create_line_graph(
            g, self.three_body_cutoff, directed=True
        )

        bond_graph.ndata["bond_index"] = bond_graph.ndata["edge_ids"]
        threebody_bond_expansion = self.threebody_bond_expansion(
            bond_graph.ndata["bond_dist"]
        )
        smooth_cutoff = polynomial_cutoff(
            threebody_bond_expansion,
            self.three_body_cutoff,
            self.cutoff_exponent,
        )
        bond_graph.ndata["bond_expansion"] = (
            smooth_cutoff * threebody_bond_expansion
        )
        bond_indices = bond_graph.ndata["bond_index"][bond_graph.edges()[0]]
        bond_graph.edata["center_atom_index"] = g.edges()[1][bond_indices]
        bond_graph.apply_edges(compute_theta)
        bond_graph.edata["angle_expansion"] = self.angle_expansion(
            bond_graph.edata["theta"]
        )

        # atom_features = self.atom_embedding(g.ndata["atom_features"])
        atom_features = self.atom_embedding(g.ndata["node_type"])

        bond_features = self.bond_embedding(g.edata["bond_expansion"])
        angle_features = self.angle_embedding(
            bond_graph.edata["angle_expansion"]
        )
        if self.state_embedding is not None and state_attr is not None:
            state_attr = self.state_embedding(state_attr)
        else:
            state_attr = None

        # shared message weights
        atom_bond_weights = (
            self.atom_bond_weights(g.edata["bond_expansion"])
            if self.atom_bond_weights is not None
            else None
        )
        # print("atom_bond_weights", torch.sum(atom_bond_weights))
        bond_bond_weights = (
            self.bond_bond_weights(g.edata["bond_expansion"])
            if self.bond_bond_weights is not None
            else None
        )
        # print("bond_bond_weights", torch.sum(bond_bond_weights))
        threebody_bond_weights = (
            self.threebody_bond_weights(bond_graph.ndata["bond_expansion"])
            if self.threebody_bond_weights is not None
            else None
        )

        # message passing layers
        for i in range(self.n_blocks - 1):
            atom_features, bond_features, state_attr = self.atom_graph_layers[
                i
            ](
                g,
                atom_features,
                bond_features,
                state_attr,
                atom_bond_weights,
                bond_bond_weights,
            )
            bond_features, angle_features = self.bond_graph_layers[i](
                bond_graph,
                atom_features,
                bond_features,
                angle_features,
                threebody_bond_weights,
            )

        atom_features, bond_features, state_attr = self.atom_graph_layers[-1](
            g,
            atom_features,
            bond_features,
            state_attr,
            atom_bond_weights,
            bond_bond_weights,
        )

        g.ndata["atom_feat"] = self.final_layer(atom_features)
        structure_properties = readout_nodes(
            g, "atom_feat", op=self.readout_operation
        )
        # self.add_ewald=True
        # ewald_en = 0
        # if self.add_ewald:
        #    ewald_en = get_atomic_repulsion(g)
        # total_energies = (torch.squeeze(structure_properties)) +ewald_en/g.num_nodes()
        total_energies = torch.squeeze(structure_properties)

        penalty_factor = 500.0  # Penalty weight, tune as needed
        penalty_factor = 1000.0  # Penalty weight, tune as needed
        penalty_threshold = 1.0  # 1 angstrom

        # Calculate penalties for distances less than the threshold
        penalties = torch.where(
            bond_dist < penalty_threshold,
            penalty_factor * (penalty_threshold - bond_dist),
            torch.zeros_like(bond_dist),
        )
        total_penalty = torch.sum(penalties)

        # min_distance=1.0
        # mask = bond_dist < min_distance
        # penalty = torch.zeros_like(bond_dist)
        # epsilon=1.0
        # alpha=12
        # Smooth penalty calculation for close distances
        # penalty[mask] = epsilon * ((min_distance / bond_dist[mask]) ** alpha)

        # Sum up the penalties
        # total_penalty = torch.sum(penalty)
        total_energies += total_penalty
        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)
        grad_vars = [
            g.ndata["pos"],
            st,
        ]  # if self.calc_stresses else [g.ndata["pos"]]
        # print('total_energies',total_energies)
        grads = grad(
            total_energies,
            grad_vars,
            grad_outputs=torch.ones_like(total_energies),
            create_graph=True,
            retain_graph=True,
        )
        forces = -grads[0]
        volume = torch.abs(torch.det(lattice))
        sts = -grads[1]
        scale = 1.0 / volume * -160.21766208
        sts = (
            [i * j for i, j in zip(sts, scale)]
            if sts.dim() == 3
            else [sts * scale]
        )
        stresses = torch.cat(sts)
        result = {}
        result["out"] = total_energies
        result["grad"] = forces
        result["stresses"] = stresses
        return result
