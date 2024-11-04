"""Shared model-building components."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import dgl


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
