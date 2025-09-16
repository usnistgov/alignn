"""Shared model-building components."""

from typing import Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import dgl
from typing import Tuple


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
    ):
        """Register torch parameters for RBF expansion."""
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
            self.gamma = 1 / (lengthscale**2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )


def compute_pair_vector_and_distance(g: dgl.DGLGraph):
    """Calculate bond vectors and distances using dgl graphs."""
    # print('g.edges()',g.ndata["cart_coords"][g.edges()[1]].shape,g.edata["pbc_offshift"].shape)
    dst_pos = g.ndata["cart_coords"][g.edges()[1]] + g.edata["images"]
    src_pos = g.ndata["cart_coords"][g.edges()[0]]
    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


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


def compute_cartesian_coordinates(g, lattice, dtype=torch.float32):
    """
    Compute Cartesian coords from fractional coords and lattice matrices.

    Args:
        g: DGL graph with 'frac_coords' as node data.
        lattice: Tensor of shape (B, 3, 3), where B is the batch size.
        dtype: Torch dtype to ensure consistent tensor types.

    Returns:
        Tensor of Cartesian coordinates with shape (N, 3).
    """
    # Get fractional coordinates (N, 3) and ensure correct dtype
    frac_coords = g.ndata["frac_coords"].to(dtype)

    # Ensure lattice is 3D with shape (B, 3, 3) and correct dtype
    if lattice.dim() == 2:  # If shape is (3, 3), expand to (1, 3, 3)
        lattice = lattice.unsqueeze(0).to(dtype)
    else:
        lattice = lattice.to(dtype)

    # Generate batch indices to map nodes to their corresponding graph
    batch_indices = torch.repeat_interleave(
        torch.arange(len(lattice), device=frac_coords.device),
        g.batch_num_nodes(),
    )

    # Expand lattice matrices based on batch indices to match node count
    expanded_lattice = lattice[batch_indices]  # Shape: (N, 3, 3)

    # Perform batched matrix multiplication to get Cartesian coordinates
    cart_coords = torch.bmm(
        frac_coords.unsqueeze(1),  # Shape: (N, 1, 3)
        expanded_lattice,  # Shape: (N, 3, 3)
    ).squeeze(
        1
    )  # Shape: (N, 3)

    return cart_coords


def lightweight_line_graph(
    input_graph: dgl.DGLGraph,
    feature_name: str,
    filter_condition: Callable[[torch.Tensor], torch.Tensor],
) -> dgl.DGLGraph:
    """Make the line graphs lightweight with preserved node ordering.
    Handles both batched and unbatched graphs.

    Args:
        input_graph: Input DGL graph (can be batched)
        feature_name: Name of the edge feature to filter on
        filter_condition: Take edge features and returns boolean mask

    Returns:
        New DGL graph with filtered edges preserving original node ordering
    """
    # Check if graph is batched
    is_batched = (
        hasattr(input_graph, "batch_size") and input_graph.batch_size > 1
    )

    if is_batched:
        # Get the batch size and number of nodes per graph
        # batch_size = input_graph.batch_size
        graph_list = dgl.unbatch(input_graph)
        processed_graphs = []

        # Process each graph individually
        for g in graph_list:
            # Get active edges for this graph
            g_active_edges = torch.logical_not(
                filter_condition(g.edata[feature_name])
            )

            # Get filtered edges
            g_src, g_dst = g.edges()
            g_src = g_src[g_active_edges]
            g_dst = g_dst[g_active_edges]

            # Get edge IDs for active edges
            g_edge_ids = g_active_edges.nonzero().squeeze()

            # Create new graph with same number of nodes
            new_g = dgl.graph(
                (g_src, g_dst),
                num_nodes=g.num_nodes(),
                device=input_graph.device,
            )

            # Copy edge IDs
            new_g.edata["edge_ids"] = g_edge_ids

            # Copy node features
            for node_feature, node_value in g.ndata.items():
                new_g.ndata[node_feature] = node_value

            # Copy filtered edge features
            for edge_feature, edge_value in g.edata.items():
                new_g.edata[edge_feature] = edge_value[g_active_edges]

            processed_graphs.append(new_g)

        # Batch the processed graphs back together
        return dgl.batch(processed_graphs)

    else:
        # Handle single graph case (original implementation)
        active_edges = torch.logical_not(
            filter_condition(input_graph.edata[feature_name])
        )

        source_nodes, destination_nodes = input_graph.edges()
        source_nodes, destination_nodes = (
            source_nodes[active_edges],
            destination_nodes[active_edges],
        )

        edge_ids = active_edges.nonzero().squeeze()

        new_graph = dgl.graph(
            (source_nodes, destination_nodes),
            num_nodes=input_graph.num_nodes(),
            device=input_graph.device,
        )

        new_graph.edata["edge_ids"] = edge_ids

        for node_feature, node_value in input_graph.ndata.items():
            new_graph.ndata[node_feature] = node_value

        for edge_feature, edge_value in input_graph.edata.items():
            new_graph.edata[edge_feature] = edge_value[active_edges]

        return new_graph


def lightweight_line_graph1(
    input_graph: dgl.DGLGraph,
    feature_name: str,
    filter_condition: Callable[[torch.Tensor], torch.Tensor],
) -> dgl.DGLGraph:
    """Make the line graphs lightweight with preserved node ordering.

    Args:
        input_graph: Input DGL graph
        feature_name: Name of the edge feature to filter on
        filter_condition: Takes edge features and returns boolean mask

    Returns:
        Filtered edges while preserving original node ordering
    """
    # Get active edges based on filter condition
    active_edges = torch.logical_not(
        filter_condition(input_graph.edata[feature_name])
    )

    # Get filtered edges
    source_nodes, destination_nodes = input_graph.edges()
    source_nodes, destination_nodes = (
        source_nodes[active_edges],
        destination_nodes[active_edges],
    )

    # Get edge IDs for the active edges
    edge_ids = active_edges.nonzero().squeeze()

    # Create new graph with same number of nodes as input graph
    new_graph = dgl.graph(
        (source_nodes, destination_nodes),
        num_nodes=input_graph.num_nodes(),
        device=input_graph.device,
    )

    # Copy edge IDs
    new_graph.edata["edge_ids"] = edge_ids
    # print('input_graph',input_graph)
    # print('new_graph',new_graph)
    # Copy all node features directly (maintaining same number of nodes)
    for node_feature, node_value in input_graph.ndata.items():
        new_graph.ndata[node_feature] = node_value

    # Copy filtered edge features
    for edge_feature, edge_value in input_graph.edata.items():
        new_graph.edata[edge_feature] = edge_value[active_edges]

    return new_graph


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
        # print('xtype',x.dtype)
        return self.layer(x)


def compute_net_torque(
    positions: torch.Tensor, forces: torch.Tensor, n_nodes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the net torque on a system of particles."""
    total_mass = n_nodes.float().sum()  # Total number of particles
    com = torch.sum(positions, dim=0) / total_mass  # Center of mass

    # Compute the relative positions of particles with respect to CoM
    com_repeat = com.repeat(
        positions.size(0), 1
    )  # Repeat CoM for each particle
    com_relative_positions = (
        positions - com_repeat
    )  # Relative position to the CoM

    # Compute individual torques (cross product of r_i and F_i)
    torques = torch.cross(com_relative_positions, forces)  # Shape: (N, 3)

    # Aggregate torques to get the net torque (sum all torques)
    net_torque = torch.sum(torques, dim=0)  # Sum of all individual torques

    return net_torque, com_relative_positions


def remove_net_torque(
    g: dgl.DGLGraph,
    forces: torch.Tensor,
    n_nodes: torch.Tensor,
) -> torch.Tensor:
    """Adjust the predicted forces to eliminate net torque.

    Args:
        g : dgl.DGLGraph
            The graph representing a batch of particles (atoms).
        forces : torch.Tensor of shape (N, 3)
            Predicted forces on atoms.
        n_nodes : torch.Tensor of shape (B,)
            Number of nodes in each graph,
            where B is the number of graphs in the batch.

    Returns:
        adjusted_forces : torch.Tensor of shape (N, 3)
            Adjusted forces with zero net torque
            and net force for each graph.
    """
    # Step 1: Get positions from the graph
    positions = g.ndata["cart_coords"]

    # Compute the net torque and relative positions
    tau_total, r = compute_net_torque(positions, forces, n_nodes)

    # Step 2: Compute scalar s per graph: sum_i ||r_i||^2
    r_squared = torch.sum(r**2, dim=1)  # Shape: (N,)

    # Sum over nodes to aggregate r_squared for each graph
    s = torch.zeros(n_nodes.size(0), device=positions.device)
    start_idx = 0
    for i, num_nodes in enumerate(n_nodes):
        end_idx = start_idx + num_nodes
        s[i] = torch.sum(r_squared[start_idx:end_idx])
        start_idx = end_idx

    # Step 3: Compute matrix S per graph: sum_i outer(r_i, r_i)
    r_unsqueezed = r.unsqueeze(2)  # Shape: (N, 3, 1)
    r_T_unsqueezed = r.unsqueeze(1)  # Shape: (N, 1, 3)
    outer_products = r_unsqueezed @ r_T_unsqueezed  # Shape: (N, 3, 3)

    # Aggregate outer products for each graph
    S = torch.zeros(n_nodes.size(0), 3, 3, device=positions.device)
    start_idx = 0
    for i, num_nodes in enumerate(n_nodes):
        end_idx = start_idx + num_nodes
        S[i] = torch.sum(outer_products[start_idx:end_idx], dim=0)
        start_idx = end_idx

    # Step 4: Compute M = S - sI
    Imat = (
        torch.eye(3, device=positions.device)
        .unsqueeze(0)
        .expand(n_nodes.size(0), -1, -1)
    )  # Identity matrix
    M = S - s.view(-1, 1, 1) * Imat  # Shape: (B, 3, 3)

    # Step 5: Right-hand side vector b per graph
    b = -tau_total  # Shape: (B, 3)

    # Step 6: Solve M * mu = b for mu per graph
    try:
        mu = torch.linalg.solve(
            M, b
        )  # Shape: (B, 3) -- No need for unsqueeze(2)
    except RuntimeError:
        # Handle singular matrix M by using the pseudo-inverse
        M_pinv = torch.linalg.pinv(M)  # Shape: (B, 3, 3)
        mu = torch.bmm(M_pinv, b.unsqueeze(2)).squeeze(2)  # Shape: (B, 3)

    # Step 7: Compute adjustments to forces
    mu_batch = torch.repeat_interleave(mu, n_nodes, dim=0)  # Shape: (N, 3)
    forces_delta = torch.cross(r, mu_batch)  # Shape: (N, 3)

    # Step 8: Adjust forces
    adjusted_forces = forces + forces_delta  # Shape: (N, 3)

    return adjusted_forces
