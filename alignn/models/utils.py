"""Shared model-building components."""

from typing import Optional
import numpy as np
import torch

# from torch import nn
from math import pi
import torch.nn as nn

# from scipy.special import spherical_jn
# from scipy.special import sph_harm, lpmv
import math
import dgl


class BesselExpansion(nn.Module):
    """Expand interatomic distances with spherical Bessel functions."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        cutoff: Optional[float] = None,
    ):
        """Register torch parameters for Bessel function expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.cutoff = cutoff if cutoff is not None else vmax

        # Generate frequency parameters for Bessel functions
        # Convert to float32 explicitly
        frequencies = torch.tensor(
            [(n * np.pi) / self.cutoff for n in range(1, bins + 1)],
            dtype=torch.float32,
        )
        self.register_buffer("frequencies", frequencies)

        # Precompute normalization factors
        norm_factors = torch.tensor(
            [np.sqrt(2 / self.cutoff) for _ in range(bins)],
            dtype=torch.float32,
        )
        self.register_buffer("norm_factors", norm_factors)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply Bessel function expansion to interatomic distance tensor."""
        # Ensure input is float32
        distance = distance.to(torch.float32)

        # Compute the zero-order spherical Bessel functions
        x = distance.unsqueeze(-1) * self.frequencies

        # Handle the case where x is close to zero
        mask = x.abs() < 1e-10
        j0 = torch.where(mask, torch.ones_like(x), torch.sin(x) / x)

        # Apply normalization
        bessel_features = j0 * self.norm_factors

        # Apply smooth cutoff function if cutoff is specified
        if self.cutoff < self.vmax:
            envelope = self._smooth_cutoff(distance)
            bessel_features = bessel_features * envelope.unsqueeze(-1)

        return bessel_features

    def _smooth_cutoff(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply smooth cutoff function to ensure continuity at boundary."""
        x = torch.pi * distance / self.cutoff
        cutoffs = 0.5 * (torch.cos(x) + 1.0)
        return torch.where(
            distance <= self.cutoff, cutoffs, torch.zeros_like(distance)
        )


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


class FourierExpansion(nn.Module):
    """Fourier Expansion of a (periodic) scalar feature."""

    def __init__(
        self,
        max_f: int = 5,
        interval: float = pi,
        scale_factor: float = 1.0,
        learnable: bool = False,
    ):
        """Args:
        max_f (int): the maximum frequency of the Fourier expansion.
            Default = 5
        interval (float): interval of the Fourier exp, such that functions
            are orthonormal over [-interval, interval]. Default = pi
        scale_factor (float): pre-factor to scale all values.
            learnable (bool): whether to set the frequencies as learnable
            Default = False.
        """
        super().__init__()
        self.max_f = max_f
        self.interval = interval
        self.scale_factor = scale_factor
        # Initialize frequencies at canonical
        if learnable:
            self.frequencies = torch.nn.Parameter(
                data=torch.arange(0, max_f + 1, dtype=torch.float32),
                requires_grad=True,
            )
        else:
            self.register_buffer(
                "frequencies", torch.arange(0, max_f + 1, dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand x into cos and sin functions."""
        result = x.new_zeros(x.shape[0], 1 + 2 * self.max_f)
        tmp = torch.outer(x, self.frequencies)
        result[:, ::2] = torch.cos(tmp * pi / self.interval)
        result[:, 1::2] = torch.sin(tmp[:, 1:] * pi / self.interval)
        return result / self.interval * self.scale_factor


class SphericalHarmonicsExpansion(nn.Module):
    """Expand angles with spherical harmonics."""

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = math.pi,
        bins: int = 20,
        l_max: int = 3,
    ):
        """Register torch parameters for spherical harmonics expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.l_max = l_max
        self.num_harmonics = (l_max + 1) ** 2
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Apply spherical harmonics expansion to angular tensors."""
        harmonics = []
        phi = torch.zeros_like(theta)
        for l_x in range(self.l_max + 1):
            for m in range(-l_x, l_x + 1):
                y_lm = self._spherical_harmonic(l_x, m, theta, phi)
                harmonics.append(y_lm)
        return torch.stack(harmonics, dim=-1)

    def _legendre_polynomial(
        self, l_x: int, m: int, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the associated Legendre polynomials P_l^m(x).
        :param l: Degree of the polynomial.
        :param m: Order of the polynomial.
        :param x: Input tensor.
        :return: Associated Legendre polynomial evaluated at x.
        """
        pmm = torch.ones_like(x)
        if m > 0:
            somx2 = torch.sqrt((1 - x) * (1 + x))
            fact = 1.0
            for i in range(1, m + 1):
                pmm = -pmm * fact * somx2
                fact += 2.0

        if l_x == m:
            return pmm
        pmmp1 = x * (2 * m + 1) * pmm
        if l_x == m + 1:
            return pmmp1

        pll = torch.zeros_like(x)
        for ll in range(m + 2, l_x + 1):
            pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
            pmm = pmmp1
            pmmp1 = pll

        return pll

    def _spherical_harmonic(
        self, l_x: int, m: int, theta: torch.Tensor, phi: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the real part of the spherical harmonics Y_l^m(theta, phi).
        :param l: Degree of the harmonic.
        :param m: Order of the harmonic.
        :param theta: Polar angle (in radians).
        :param phi: Azimuthal angle (in radians).
        :return: Real part of the spherical harmonic Y_l^m.
        """
        sqrt2 = torch.sqrt(torch.tensor(2.0))
        if m > 0:
            return (
                sqrt2
                * self._k(l_x, m)
                * torch.cos(m * phi)
                * self._legendre_polynomial(l_x, m, torch.cos(theta))
            )
        elif m < 0:
            return (
                sqrt2
                * self._k(l_x, -m)
                * torch.sin(-m * phi)
                * self._legendre_polynomial(l_x, -m, torch.cos(theta))
            )
        else:
            return self._k(l_x, 0) * self._legendre_polynomial(
                l_x, 0, torch.cos(theta)
            )

    def _k(self, l_x: int, m: int) -> float:
        """
        Normalization constant for the spherical harmonics.
        :param l: Degree of the harmonic.
        :param m: Order of the harmonic.
        :return: Normalization constant.
        """
        return math.sqrt(
            (2 * l_x + 1)
            / (4 * math.pi)
            * math.factorial(l_x - m)
            / math.factorial(l_x + m)
        )


def compute_pair_vector_and_distance(g: dgl.DGLGraph):
    """Calculate bond vectors and distances using dgl graphs."""
    dst_pos = g.ndata["coords"][g.edges()[1]] + g.edata["images"]
    src_pos = g.ndata["coords"][g.edges()[0]]
    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


def check_line_graph(
    graph: dgl.DGLGraph, line_graph: dgl.DGLGraph, threebody_cutoff: float
):
    """Ensure that 3body line graph is compatible with a given graph.

    Args:
        graph: atomistic graph
        line_graph: line graph of atomistic graph
        threebody_cutoff: cutoff for three-body interactions
    """
    valid_three_body = graph.edata["d"] <= threebody_cutoff
    if line_graph.num_nodes() == graph.edata["r"][valid_three_body].shape[0]:
        line_graph.ndata["r"] = graph.edata["r"][valid_three_body]
        line_graph.ndata["d"] = graph.edata["d"][valid_three_body]
        line_graph.ndata["images"] = graph.edata["images"][valid_three_body]
    else:
        three_body_id = torch.concatenate(line_graph.edges())
        max_three_body_id = (
            torch.max(three_body_id) + 1 if three_body_id.numel() > 0 else 0
        )
        line_graph.ndata["r"] = graph.edata["r"][:max_three_body_id]
        line_graph.ndata["d"] = graph.edata["d"][:max_three_body_id]
        line_graph.ndata["images"] = graph.edata["images"][:max_three_body_id]

    return line_graph


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
