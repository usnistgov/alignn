"""Shared model-building components."""

from typing import Optional
import numpy as np
import torch
from math import pi
import torch.nn as nn
import math
import dgl
import torch
from typing import Any, Callable, Literal, cast


def get_atomic_repulsion(g, cutoff=5.0):
    """
    Calculate atomic repulsion energy using pairwise Coulomb interactions within a cutoff distance.

    Parameters:
        g (DGLGraph): ALIGNN graph with atom charges (Z) and precomputed bond lengths in g.edata['d'].
        cutoff (float): Cutoff distance for pairwise interactions.

    Returns:
        float: Atomic repulsion energy for the given graph.
    """

    # Atomic charges
    Z = g.ndata["Z"].squeeze()  # Ensure Z is a 1D tensor
    bond_lengths = g.edata[
        "d"
    ]  # Precomputed bond lengths in Cartesian coordinates

    # Atomic indices for each edge
    src, dst = g.edges()

    # Mask for distances below the cutoff
    valid_edges = bond_lengths < cutoff

    # Get charges for each pair
    Zi = Z[src[valid_edges]]
    Zj = Z[dst[valid_edges]]
    rij = bond_lengths[valid_edges]

    # Compute repulsion energy
    repulsion_energy = torch.sum(Zi * Zj / rij)

    return repulsion_energy


class RadialBesselFunction(nn.Module):

    def __init__(
        self,
        max_n: int,
        cutoff: float,
        learnable: bool = False,
        dtype=torch.float32,
    ):
        """
        Args:
            max_n: int, max number of roots (including max_n)
            cutoff: float, cutoff radius
            learnable: bool, whether to learn the location of roots.
        """
        super().__init__()
        self.max_n = max_n
        self.inv_cutoff = 1 / cutoff
        self.norm_const = (2 * self.inv_cutoff) ** 0.5
        if learnable:
            self.frequencies = torch.nn.Parameter(
                data=torch.Tensor(
                    pi * torch.arange(1, self.max_n + 1, dtype=dtype)
                ),
                requires_grad=True,
            )
        else:
            self.register_buffer(
                "frequencies",
                pi * torch.arange(1, self.max_n + 1, dtype=dtype),
            )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r[:, None]  # (nEdges,1)
        d_scaled = r * self.inv_cutoff
        return self.norm_const * torch.sin(self.frequencies * d_scaled) / r


def get_ewald_sum(g, lattice_mat, alpha=0.2, r_cut=10.0, k_cut=5):
    """
    Calculate the Ewald sum energy for the DGL graph using precomputed rij vectors.

    Parameters:
        g (DGLGraph): ALIGNN graph with atom features, fractional coordinates, and precomputed rij vectors.
        alpha (float): Ewald splitting parameter, controls the balance between real and reciprocal space sums.
        r_cut (float): Real-space cutoff distance for pairwise interactions.
        k_cut (int): Reciprocal-space cutoff for Fourier components.

    Returns:
        float: Ewald sum energy for the given graph.
    """

    # Atomic numbers (charges) and fractional coordinates
    Z = g.ndata["Z"]  # Atomic charges (assuming Z is atomic number)
    cart_pos = g.ndata[
        "frac_coords"
    ]  # Fractional coordinates in Cartesian space
    r_ij_vectors = g.edata[
        "r"
    ]  # Precomputed rij vectors in Cartesian coordinates

    # Initialize Ewald sum energy
    ewald_energy = 0.0

    # Real-space sum using precomputed rij vectors
    src, dst = g.edges()  # Get the source and destination nodes for each edge
    for edge_idx in range(len(src)):
        i = src[edge_idx]
        j = dst[edge_idx]

        # Pairwise distance (norm of r_ij)
        r = torch.norm(r_ij_vectors[edge_idx])

        if r < r_cut:
            ewald_energy += Z[i] * Z[j] * torch.erfc(alpha * r) / r

    # Reciprocal-space sum
    # lattice_mat = g.ndata['lattice_mat'][0]  # Assuming lattice matrix is uniform across nodes
    recip_vectors = (
        2 * pi * torch.inverse(lattice_mat).T
    )  # Reciprocal lattice vectors
    for h in range(-k_cut, k_cut + 1):
        for k in range(-k_cut, k_cut + 1):
            for l in range(-k_cut, k_cut + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                k_vec = (
                    h * recip_vectors[:, 0]
                    + k * recip_vectors[:, 1]
                    + l * recip_vectors[:, 2]
                )
                k_sq = torch.dot(k_vec, k_vec)
                structure_factor = torch.sum(
                    Z * torch.exp(1j * torch.matmul(cart_pos, k_vec))
                )
                ewald_energy += (
                    torch.exp(-k_sq / (4 * alpha**2)) / k_sq
                ) * (torch.norm(structure_factor) ** 2)

    # Self-interaction correction
    # print('Z',Z)
    # ewald_energy -= alpha / torch.sqrt(pi) * torch.sum(Z ** 2)
    ewald_energy -= (
        alpha / torch.sqrt(torch.tensor(torch.pi)) * torch.sum(Z**2)
    )
    return ewald_energy.real  # Return the real part of the energy


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
    # print('g.edges()',g.ndata["cart_coords"][g.edges()[1]].shape,g.edata["pbc_offshift"].shape)
    dst_pos = g.ndata["cart_coords"][g.edges()[1]] + g.edata["images"]
    src_pos = g.ndata["cart_coords"][g.edges()[0]]
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


def compute_cartesian_coordinates(g, lattice, dtype=torch.float32):
    """
    Compute Cartesian coordinates from fractional coordinates and lattice matrices.

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


class RBFExpansionSmooth(nn.Module):
    """
    RBF Expansion layer for bond lengths with smooth output variation.
    """

    def __init__(self, num_centers=10, cutoff=5.0, sigma=0.5):
        super(RBFExpansionSmooth, self).__init__()

        # Initialize centers and sigma for Gaussian RBFs
        self.cutoff = cutoff
        self.sigma = sigma
        self.centers = torch.linspace(0, cutoff, num_centers).view(
            1, -1
        )  # Shape (1, num_centers)

    def forward(self, bondlengths):
        """
        Compute the RBF features for a batch of bond lengths.

        Parameters:
        - bondlengths: Tensor of shape (batch_size,) containing bond lengths.

        Returns:
        - RBF expanded features: Tensor of shape (batch_size, num_centers) with smoothly varying RBFs.
        """
        # Reshape bondlengths to (batch_size, 1) for broadcasting
        bondlengths = bondlengths.view(-1, 1)  # Shape (batch_size, 1)

        # Calculate RBF values
        rbf_features = torch.exp(
            -((bondlengths - self.centers.to(bondlengths.device)) ** 2)
            / (2 * self.sigma**2)
        )

        # Apply cutoff
        mask = bondlengths <= self.cutoff
        rbf_features = (
            rbf_features * mask.float()
        )  # Mask to zero out beyond cutoff

        return rbf_features


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


def _create_directed_line_graph(
    graph: dgl.DGLGraph, threebody_cutoff: float
) -> dgl.DGLGraph:
    with torch.no_grad():
        pg = prune_edges_by_features(
            graph,
            feat_name="bond_dist",
            condition=lambda x: torch.gt(x, threebody_cutoff),
        )
        """
        lg=graph.line_graph(shared=True)
        lg.ndata["src_bond_sign"] = torch.ones(
            (lg.number_of_nodes(), 1),
            dtype=lg.ndata["bond_vec"].dtype,
            device=lg.device,
        )
        return lg
        """
        src_indices, dst_indices = pg.edges()
        images = pg.edata["images"]
        all_indices = torch.arange(
            pg.number_of_nodes(), device=graph.device
        ).unsqueeze(dim=0)
        num_bonds_per_atom = torch.count_nonzero(
            src_indices.unsqueeze(dim=1) == all_indices, dim=0
        )
        num_edges_per_bond = (num_bonds_per_atom - 1).repeat_interleave(
            num_bonds_per_atom
        )
        lg_src = torch.empty(
            num_edges_per_bond.sum(), dtype=torch.int64, device=graph.device
        )
        lg_dst = torch.empty(
            num_edges_per_bond.sum(), dtype=torch.int64, device=graph.device
        )
        incoming_edges = src_indices.unsqueeze(1) == dst_indices
        is_self_edge = src_indices == dst_indices
        not_self_edge = ~is_self_edge

        n = 0
        # create line graph edges for bonds that are self edges in atom graph
        if is_self_edge.any():
            edge_inds_s = is_self_edge.nonzero()
            lg_dst_s = edge_inds_s.repeat_interleave(
                num_edges_per_bond[is_self_edge] + 1
            )
            lg_src_s = incoming_edges[is_self_edge].nonzero()[:, 1].squeeze()
            lg_src_s = lg_src_s[lg_src_s != lg_dst_s]
            lg_dst_s = edge_inds_s.repeat_interleave(
                num_edges_per_bond[is_self_edge]
            )
            n = len(lg_dst_s)
            lg_src[:n], lg_dst[:n] = lg_src_s, lg_dst_s

        # create line graph edges for bonds that are not self edges in atom graph
        shared_src = src_indices.unsqueeze(1) == src_indices
        back_tracking = (dst_indices.unsqueeze(1) == src_indices) & torch.all(
            -images.unsqueeze(1) == images, axis=2
        )
        incoming = incoming_edges & (shared_src | ~back_tracking)

        edge_inds_ns = not_self_edge.nonzero().squeeze()
        lg_src_ns = incoming[not_self_edge].nonzero()[:, 1].squeeze()
        lg_dst_ns = edge_inds_ns.repeat_interleave(
            num_edges_per_bond[not_self_edge]
        )
        lg_src[n:], lg_dst[n:] = lg_src_ns, lg_dst_ns
        lg = dgl.graph((lg_src, lg_dst))

        for key in pg.edata:
            lg.ndata[key] = pg.edata[key][: lg.number_of_nodes()]

        # we need to store the sign of bond vector when a bond is a src node in the line
        # graph in order to appropriately calculate angles when self edges are involved
        lg.ndata["src_bond_sign"] = torch.ones(
            (lg.number_of_nodes(), 1),
            dtype=lg.ndata["bond_vec"].dtype,
            device=lg.device,
        )
        # if we flip self edges then we need to correct computed angles by pi - angle
        # lg.ndata["src_bond_sign"][edge_inds_s] = -lg.ndata["src_bond_sign"][edge_ind_s]
        # find the intersection for the rare cases where not all edges end up as nodes in the line graph
        all_ns, counts = torch.cat(
            [
                torch.arange(lg.number_of_nodes(), device=graph.device),
                edge_inds_ns,
            ]
        ).unique(return_counts=True)
        lg_inds_ns = all_ns[torch.where(counts > 1)]
        lg.ndata["src_bond_sign"][lg_inds_ns] = -lg.ndata["src_bond_sign"][
            lg_inds_ns
        ]

    return lg


def prune_edges_by_features(
    graph: dgl.DGLGraph,
    feat_name: str,
    condition: Callable[[torch.Tensor], torch.Tensor],
    keep_ndata: bool = False,
    keep_edata: bool = True,
    *args,
    **kwargs,
) -> dgl.DGLGraph:
    if feat_name not in graph.edata:
        raise ValueError(
            f"Edge field {feat_name} not an edge feature in given graph."
        )

    valid_edges = torch.logical_not(
        condition(graph.edata[feat_name], *args, **kwargs)
    )
    valid_edges1 = torch.ones(
        graph.num_edges(), dtype=torch.bool, device=graph.device
    )
    # print('valid_edges',valid_edges,valid_edges.shape)
    # print('valid_edges1',valid_edges1,valid_edges1.shape)

    src, dst = graph.edges()
    src, dst = src[valid_edges], dst[valid_edges]
    e_ids = valid_edges.nonzero().squeeze()
    new_g = dgl.graph((src, dst), device=graph.device)
    new_g.edata["edge_ids"] = e_ids  # keep track of original edge ids

    if keep_ndata:
        for key, value in graph.ndata.items():
            new_g.ndata[key] = value
    if keep_edata:
        for key, value in graph.edata.items():
            new_g.edata[key] = value[valid_edges]

    return new_g


def compute_theta(
    edges: dgl.udf.EdgeBatch,
    cosine: bool = False,
    directed: bool = True,
    eps=1e-7,
) -> dict[str, torch.Tensor]:
    """User defined dgl function to calculate bond angles from edges in a graph.

    Args:
        edges: DGL graph edges
        cosine: Whether to return the cosine of the angle or the angle itself
        directed: Whether to the line graph was created with create directed line graph.
            In which case bonds (only those that are not self bonds) need to
            have their bond vectors flipped.
        eps: eps value used to clamp cosine values to avoid acos of values > 1.0

    Returns:
        dict[str, torch.Tensor]: Dictionary containing bond angles and distances
    """
    vec1 = (
        edges.src["bond_vec"] * edges.src["src_bond_sign"]
        if directed
        else edges.src["bond_vec"]
    )
    vec2 = edges.dst["bond_vec"]
    key = "cos_theta" if cosine else "theta"
    val = torch.sum(vec1 * vec2, dim=1) / (
        torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1)
    )
    val = val.clamp_(
        min=-1 + eps, max=1 - eps
    )  # stability for floating point numbers > 1.0
    if not cosine:
        val = torch.acos(val)
    return {key: val, "triple_bond_lengths": edges.dst["bond_dist"]}


def create_line_graph(
    g: dgl.DGLGraph, threebody_cutoff: float, directed: bool = False
) -> dgl.DGLGraph:
    """
    Calculate the three body indices from pair atom indices.

    Args:
        g: DGL graph
        threebody_cutoff (float): cutoff for three-body interactions
        directed (bool): Whether to create a directed line graph, or an M3gnet 3body line graph
            Default = False (M3Gnet)

    Returns:
        l_g: DGL graph containing three body information from graph
    """
    graph_with_three_body = prune_edges_by_features(
        g, feat_name="bond_dist", condition=lambda x: x > threebody_cutoff
    )
    if directed:
        # lg = g.line_graph(shared=True)
        # return lg
        lg = _create_directed_line_graph(
            graph_with_three_body, threebody_cutoff
        )
    else:
        lg = _compute_3body(graph_with_three_body)

    return lg


def compute_pair_vector_and_distance(g: dgl.DGLGraph):
    """Calculate bond vectors and distances using dgl graphs.

    Args:
    g: DGL graph

    Returns:
    bond_vec (torch.tensor): bond distance between two atoms
    bond_dist (torch.tensor): vector from src node to dst node
    """
    dst_pos = g.ndata["pos"][g.edges()[1]] + g.edata["images"]
    src_pos = g.ndata["pos"][g.edges()[0]]
    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


def polynomial_cutoff(
    r: torch.Tensor, cutoff: float, exponent: int = 3
) -> torch.Tensor:
    """Envelope polynomial function that ensures a smooth cutoff.

    Ensures first and second derivative vanish at cuttoff. As described in:
        https://arxiv.org/abs/2003.03123

    Args:
        r (torch.Tensor): radius distance tensor
        cutoff (float): cutoff distance.
        exponent (int): minimum exponent of the polynomial. Default is 3.
            The polynomial includes terms of order exponent, exponent + 1, exponent + 2.

    Returns: polynomial cutoff function
    """
    coef1 = -(exponent + 1) * (exponent + 2) / 2
    coef2 = exponent * (exponent + 2)
    coef3 = -exponent * (exponent + 1) / 2
    ratio = r / cutoff
    poly_envelope = (
        1
        + coef1 * ratio**exponent
        + coef2 * ratio ** (exponent + 1)
        + coef3 * ratio ** (exponent + 2)
    )

    return torch.where(r <= cutoff, poly_envelope, 0.0)


if __name__ == "__main__":
    from jarvis.core.atoms import Atoms
    from alignn.graphs import radius_graph_jarvis

    FIXTURES = {
        "lattice_mat": [
            [2.715, 2.715, 0],
            [0, 2.715, 2.715],
            [2.715, 0, 2.715],
        ],
        "coords": [[0, 0, 0], [0.25, 0.25, 0.25]],
        "elements": ["Si", "Si"],
    }
    Si = Atoms(
        lattice_mat=FIXTURES["lattice_mat"],
        coords=FIXTURES["coords"],
        elements=FIXTURES["elements"],
    )
    g, lg = radius_graph_jarvis(
        atoms=s1, cutoff=5, atom_features="atomic_number"
    )
    ewald = get_ewald_sum(g, torch.tensor(Si.lattice_mat))
