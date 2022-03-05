"""Module for DimeNet."""

# https://github.com/xnuohz/DimeNet-dgl
import torch
import torch.nn as nn
import numpy as np
import sympy as sym
from scipy.optimize import brentq
from scipy import special as sp

# from .spherical_basis_layer import SphericalBasisLayer
# from .embedding_block import EmbeddingBlock
# from .output_block import OutputBlock
# from .interaction_block import InteractionBlock

from alignn.utils import BaseSettings
from pydantic.typing import Literal
import dgl
import dgl.function as fn


def Jn(r, n):
    """
    r: int or list
    n: int or list
    len(r) == len(n)
    return value should be the same shape as the input data
    ===
    example:
        r = n = np.array([1, 2, 3, 4])
        res = [0.3, 0.1, 0.1, 0.1]
    ===
    numerical spherical bessel functions of order n
    """
    return np.sqrt(np.pi / (2 * r)) * sp.jv(n + 0.5, r)  # the same shape as n


def Jn_zeros(n, k):
    """
    n: int
    k: int
    res: array of shape [n, k]
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj


def GlorotOrthogonal(tensor, scale=2.0):
    if tensor is not None:
        nn.init.orthogonal_(tensor.data)
        scale /= (tensor.size(-2) + tensor.size(-1)) * tensor.var()
        tensor.data *= scale.sqrt()


def spherical_bessel_formulas(n):
    """
    n: int
    res: array of shape [n,]
    n sympy functions
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols("x")

    f = [sym.sin(x) / x]
    a = sym.sin(x) / x
    for i in range(1, n):
        b = sym.diff(a, x) / x
        f += [sym.simplify(b * (-x) ** i)]
        a = sym.simplify(b)
    return f


def bessel_basis(n, k):
    """
    n: int
    k: int
    res: [n, k]
    n * k sympy functions
    Computes the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    """

    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5 * Jn(zeros[order, i], order + 1) ** 2]
        normalizer_tmp = 1 / np.array(normalizer_tmp) ** 0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols("x")
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [
                sym.simplify(
                    normalizer[order][i]
                    * f[order].subs(x, zeros[order, i] * x)
                )
            ]
        bess_basis += [bess_basis_tmp]
    return bess_basis


def sph_harm_prefactor(l, m):
    """
    l: int
    m: int
    res: float
    Computes the constant pre-factor for the spherical harmonic of degree l and order m
    input:
    l: int, l>=0
    m: int, -l<=m<=l
    """
    return (
        (2 * l + 1)
        * np.math.factorial(l - abs(m))
        / (4 * np.pi * np.math.factorial(l + abs(m)))
    ) ** 0.5


def associated_legendre_polynomials(l, zero_m_only=True):
    """
    l: int
    return: l sympy functions
    Computes sympy formulas of the associated legendre polynomials up to order l (excluded).
    """
    z = sym.symbols("z")
    P_l_m = [[0] * (j + 1) for j in range(l)]

    P_l_m[0][0] = 1

    if l > 0:
        P_l_m[1][0] = z

        for j in range(2, l):
            P_l_m[j][0] = sym.simplify(
                ((2 * j - 1) * z * P_l_m[j - 1][0] - (j - 1) * P_l_m[j - 2][0])
                / j
            )

        if not zero_m_only:
            for i in range(1, l):
                P_l_m[i][i] = sym.simplify((1 - 2 * i) * P_l_m[i - 1][i - 1])
                if i + 1 < l:
                    P_l_m[i + 1][i] = sym.simplify(
                        (2 * i + 1) * z * P_l_m[i][i]
                    )
                for j in range(i + 2, l):
                    P_l_m[j][i] = sym.simplify(
                        (
                            (2 * j - 1) * z * P_l_m[j - 1][i]
                            - (i + j - 1) * P_l_m[j - 2][i]
                        )
                        / (j - i)
                    )

    return P_l_m


def real_sph_harm(l, zero_m_only=True, spherical_coordinates=True):
    """
    return: a sympy function list of length l, for i-th index of the list, it is also a list of length (2 * i + 1)
    Computes formula strings of the real part of the spherical harmonics up to order l (excluded).
    Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.
    """
    if not zero_m_only:
        S_m = [0]
        C_m = [1]
        for i in range(1, l):
            x = sym.symbols("x")
            y = sym.symbols("y")
            S_m += [x * S_m[i - 1] + y * C_m[i - 1]]
            C_m += [x * C_m[i - 1] - y * S_m[i - 1]]

    P_l_m = associated_legendre_polynomials(l, zero_m_only)

    if spherical_coordinates:
        theta = sym.symbols("theta")
        z = sym.symbols("z")

        for i in range(len(P_l_m)):
            for j in range(len(P_l_m[i])):
                if type(P_l_m[i][j]) != int:
                    P_l_m[i][j] = P_l_m[i][j].subs(z, sym.cos(theta))

        if not zero_m_only:
            phi = sym.symbols("phi")
            for i in range(len(S_m)):
                S_m[i] = (
                    S_m[i]
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )
            for i in range(len(C_m)):
                C_m[i] = (
                    C_m[i]
                    .subs(x, sym.sin(theta) * sym.cos(phi))
                    .subs(y, sym.sin(theta) * sym.sin(phi))
                )

    Y_func_l_m = [["0"] * (2 * j + 1) for j in range(l)]

    for i in range(l):
        Y_func_l_m[i][0] = sym.simplify(sph_harm_prefactor(i, 0) * P_l_m[i][0])

    if not zero_m_only:
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][j] = sym.simplify(
                    2 ** 0.5 * sph_harm_prefactor(i, j) * C_m[j] * P_l_m[i][j]
                )
        for i in range(1, l):
            for j in range(1, i + 1):
                Y_func_l_m[i][-j] = sym.simplify(
                    2 ** 0.5 * sph_harm_prefactor(i, -j) * S_m[j] * P_l_m[i][j]
                )

    return Y_func_l_m


def swish(x):
    """
    Swish activation function,
    from Ramachandran, Zopf, Le 2017. "Searching for Activation Functions"
    """
    return x * torch.sigmoid(x)


class Envelope(nn.Module):
    """
    Envelope function that ensures a smooth cutoff
    """

    def __init__(self, exponent):
        super(Envelope, self).__init__()

        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        # Envelope function divided by r
        x_p_0 = x.pow(self.p - 1)
        x_p_1 = x_p_0 * x
        x_p_2 = x_p_1 * x
        env_val = 1 / x + self.a * x_p_0 + self.b * x_p_1 + self.c * x_p_2
        return env_val


class BesselBasisLayer(nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()

        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)
        self.frequencies = nn.Parameter(torch.Tensor(num_radial))
        # self.reset_params()

    def reset_params(self):
        torch.arange(
            1, self.frequencies.numel() + 1, out=self.frequencies
        ).mul_(np.pi)

    def forward(self, g):
        d_scaled = g.edata["d"] / self.cutoff
        # Necessary for proper broadcasting behaviour
        d_scaled = torch.unsqueeze(d_scaled, -1)
        d_cutoff = self.envelope(d_scaled)
        g.edata["rbf"] = d_cutoff * torch.sin(self.frequencies * d_scaled)
        return g


class SphericalBasisLayer(nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()

        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # retrieve formulas
        self.bessel_formulas = bessel_basis(
            num_spherical, num_radial
        )  # x, [num_spherical, num_radial] sympy functions
        self.sph_harm_formulas = real_sph_harm(
            num_spherical
        )  # theta, [num_spherical, ] sympy functions
        self.sph_funcs = []
        self.bessel_funcs = []

        # convert to torch functions
        x = sym.symbols("x")
        theta = sym.symbols("theta")
        modules = {"sin": torch.sin, "cos": torch.cos}
        for i in range(num_spherical):
            if i == 0:
                first_sph = sym.lambdify(
                    [theta], self.sph_harm_formulas[i][0], modules
                )(0)
                self.sph_funcs.append(
                    lambda tensor: torch.zeros_like(tensor) + first_sph
                )
            else:
                self.sph_funcs.append(
                    sym.lambdify(
                        [theta], self.sph_harm_formulas[i][0], modules
                    )
                )
            for j in range(num_radial):
                self.bessel_funcs.append(
                    sym.lambdify([x], self.bessel_formulas[i][j], modules)
                )

    def get_bessel_funcs(self):
        return self.bessel_funcs

    def get_sph_funcs(self):
        return self.sph_funcs


class EmbeddingBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_radial,
        bessel_funcs,
        cutoff,
        envelope_exponent,
        num_atom_types=95,
        activation=None,
    ):
        super(EmbeddingBlock, self).__init__()

        self.bessel_funcs = bessel_funcs
        self.cutoff = cutoff
        self.activation = activation
        self.envelope = Envelope(envelope_exponent)
        self.embedding = nn.Embedding(num_atom_types, emb_size)
        self.dense_rbf = nn.Linear(num_radial, emb_size)
        self.dense = nn.Linear(emb_size * 3, emb_size)
        self.reset_params()

    def reset_params(self):
        nn.init.uniform_(self.embedding.weight, a=-np.sqrt(3), b=np.sqrt(3))
        GlorotOrthogonal(self.dense_rbf.weight)
        GlorotOrthogonal(self.dense.weight)

    def edge_init(self, edges):
        """msg emb init"""
        # m init
        rbf = self.dense_rbf(edges.data["rbf"])
        if self.activation is not None:
            rbf = self.activation(rbf)

        m = torch.cat([edges.src["h"], edges.dst["h"], rbf], dim=-1)
        m = self.dense(m)
        if self.activation is not None:
            m = self.activation(m)

        # rbf_env init
        d_scaled = edges.data["d"] / self.cutoff
        rbf_env = [f(d_scaled) for f in self.bessel_funcs]
        rbf_env = torch.stack(rbf_env, dim=1)

        d_cutoff = self.envelope(d_scaled)
        rbf_env = d_cutoff[:, None] * rbf_env

        return {"m": m, "rbf_env": rbf_env}

    def forward(self, g):
        g.ndata["h"] = self.embedding(g.ndata["Z"])
        g.apply_edges(self.edge_init)
        return g


class OutputBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_radial,
        num_dense,
        num_targets,
        activation=None,
        output_init=nn.init.zeros_,
    ):
        super(OutputBlock, self).__init__()

        self.activation = activation
        self.output_init = output_init
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_layers = nn.ModuleList(
            [nn.Linear(emb_size, emb_size) for _ in range(num_dense)]
        )
        self.dense_final = nn.Linear(emb_size, num_targets, bias=False)
        self.reset_params()

    def reset_params(self):
        GlorotOrthogonal(self.dense_rbf.weight)
        for layer in self.dense_layers:
            GlorotOrthogonal(layer.weight)
        self.output_init(self.dense_final.weight)

    def forward(self, g):
        with g.local_scope():
            g.edata["tmp"] = g.edata["m"] * self.dense_rbf(g.edata["rbf"])
            g.update_all(fn.copy_e("tmp", "x"), fn.sum("x", "t"))

            for layer in self.dense_layers:
                g.ndata["t"] = layer(g.ndata["t"])
                if self.activation is not None:
                    g.ndata["t"] = self.activation(g.ndata["t"])
            g.ndata["t"] = self.dense_final(g.ndata["t"])
            return dgl.readout_nodes(g, "t")


class ResidualLayer(nn.Module):
    def __init__(self, units, activation=None):
        super(ResidualLayer, self).__init__()

        self.activation = activation
        self.dense_1 = nn.Linear(units, units)
        self.dense_2 = nn.Linear(units, units)

        self.reset_params()

    def reset_params(self):
        GlorotOrthogonal(self.dense_1.weight)
        nn.init.zeros_(self.dense_1.bias)
        GlorotOrthogonal(self.dense_2.weight)
        nn.init.zeros_(self.dense_2.bias)

    def forward(self, inputs):
        x = self.dense_1(inputs)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dense_2(x)
        if self.activation is not None:
            x = self.activation(x)
        return inputs + x


class InteractionBlock(nn.Module):
    def __init__(
        self,
        emb_size,
        num_radial,
        num_spherical,
        num_bilinear,
        num_before_skip,
        num_after_skip,
        activation=None,
    ):
        super(InteractionBlock, self).__init__()

        self.activation = activation
        # Transformations of Bessel and spherical basis representations
        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.dense_sbf = nn.Linear(
            num_radial * num_spherical, num_bilinear, bias=False
        )
        # Dense transformations of input messages
        self.dense_ji = nn.Linear(emb_size, emb_size)
        self.dense_kj = nn.Linear(emb_size, emb_size)
        # Bilinear layer
        bilin_initializer = torch.empty(
            (emb_size, num_bilinear, emb_size)
        ).normal_(mean=0, std=2 / emb_size)
        self.W_bilin = nn.Parameter(bilin_initializer)
        # Residual layers before skip connection
        self.layers_before_skip = nn.ModuleList(
            [
                ResidualLayer(emb_size, activation=activation)
                for _ in range(num_before_skip)
            ]
        )
        self.final_before_skip = nn.Linear(emb_size, emb_size)
        # Residual layers after skip connection
        self.layers_after_skip = nn.ModuleList(
            [
                ResidualLayer(emb_size, activation=activation)
                for _ in range(num_after_skip)
            ]
        )

        self.reset_params()

    def reset_params(self):
        GlorotOrthogonal(self.dense_rbf.weight)
        GlorotOrthogonal(self.dense_sbf.weight)
        GlorotOrthogonal(self.dense_ji.weight)
        GlorotOrthogonal(self.dense_kj.weight)
        GlorotOrthogonal(self.final_before_skip.weight)

    def edge_transfer(self, edges):
        # Transform from Bessel basis to dence vector
        rbf = self.dense_rbf(edges.data["rbf"])
        # Initial transformation
        x_ji = self.dense_ji(edges.data["m"])
        x_kj = self.dense_kj(edges.data["m"])
        if self.activation is not None:
            x_ji = self.activation(x_ji)
            x_kj = self.activation(x_kj)

        # w: W * e_RBF \bigodot \sigma(W * m + b)
        return {"x_kj": x_kj * rbf, "x_ji": x_ji}

    def msg_func(self, edges):
        sbf = self.dense_sbf(edges.data["sbf"])
        # Apply bilinear layer to interactions and basis function activation
        # [None, 8] * [128, 8, 128] * [None, 128] -> [None, 128]
        x_kj = torch.einsum(
            "wj,wl,ijl->wi", sbf, edges.src["x_kj"], self.W_bilin
        )
        return {"x_kj": x_kj}

    def forward(self, g, l_g):
        g.apply_edges(self.edge_transfer)

        # nodes correspond to edges and edges correspond to nodes in the original graphs
        # node: d, rbf, o, rbf_env, x_kj, x_ji
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g.update_all(self.msg_func, fn.sum("x_kj", "m_update"))

        for k, v in l_g.ndata.items():
            g.edata[k] = v

        # Transformations before skip connection
        g.edata["m_update"] = g.edata["m_update"] + g.edata["x_ji"]
        for layer in self.layers_before_skip:
            g.edata["m_update"] = layer(g.edata["m_update"])
        g.edata["m_update"] = self.final_before_skip(g.edata["m_update"])
        if self.activation is not None:
            g.edata["m_update"] = self.activation(g.edata["m_update"])

        # Skip connection
        g.edata["m"] = g.edata["m"] + g.edata["m_update"]

        # Transformations after skip connection
        for layer in self.layers_after_skip:
            g.edata["m"] = layer(g.edata["m"])

        return g


class DimeNetConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["dimenet"]
    atom_input_features: int = 1
    output_features: int = 1
    """
    emb_size:int=128,
    num_blocks:int=6,
    num_bilinear:int=8,
    num_spherical:int=7,
    num_radial:int=6,
    cutoff:float=5.0,
    envelope_exponent:int=5,
    num_before_skip:int=1,
    num_after_skip:int=2,
    num_dense_output:int=3,
    num_targets:int=1,
    classification: bool = False
    alignn_layers: int = 4
    gcn_layers: int = 4
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    output_features: int = 1
    """

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class DimeNet(nn.Module):
    """
    DimeNet model.

    Parameters
    ----------
    emb_size
        Embedding size used throughout the model
    num_blocks
        Number of building blocks to be stacked
    num_bilinear
        Third dimension of the bilinear layer tensor
    num_spherical
        Number of spherical harmonics
    num_radial
        Number of radial basis functions
    cutoff
        Cutoff distance for interatomic interactions
    envelope_exponent
        Shape of the smooth cutoff
    num_before_skip
        Number of residual layers in interaction block before skip connection
    num_after_skip
        Number of residual layers in interaction block after skip connection
    num_dense_output
        Number of dense layers for the output blocks
    num_targets
        Number of targets to predict
    activation
        Activation function
    output_init
        Initial function in output block
    """

    def __init__(self, config: DimeNetConfig = DimeNetConfig(name="dimenet")):
        super(DimeNet, self).__init__()
        """
        emb_size=128,
        num_blocks=6,
        num_bilinear=8,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_dense_output=3,
        num_targets=12,
        activation=swish,
        output_init=nn.init.zeros_):
        """
        emb_size = 128
        num_blocks = 6
        num_bilinear = 8
        num_spherical = 7
        num_radial = 6
        cutoff = 5.0
        envelope_exponent = 5
        num_before_skip = 1
        num_after_skip = 2
        num_dense_output = 3
        num_targets = 1
        activation = swish
        output_init = nn.init.zeros_

        self.num_blocks = num_blocks
        self.num_radial = num_radial

        # cosine basis function expansion layer
        self.rbf_layer = BesselBasisLayer(
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
        )

        self.sbf_layer = SphericalBasisLayer(
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
        )

        # embedding block
        self.emb_block = EmbeddingBlock(
            emb_size=emb_size,
            num_radial=num_radial,
            bessel_funcs=self.sbf_layer.get_bessel_funcs(),
            cutoff=cutoff,
            envelope_exponent=envelope_exponent,
            activation=activation,
        )

        # output block
        self.output_blocks = nn.ModuleList(
            {
                OutputBlock(
                    emb_size=emb_size,
                    num_radial=num_radial,
                    num_dense=num_dense_output,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                )
                for _ in range(num_blocks + 1)
            }
        )

        # interaction block
        self.interaction_blocks = nn.ModuleList(
            {
                InteractionBlock(
                    emb_size=emb_size,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_bilinear=num_bilinear,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    activation=activation,
                )
                for _ in range(num_blocks)
            }
        )

    def edge_init(self, edges):
        # Calculate angles k -> j -> i
        R1, R2 = edges.src["o"], edges.dst["o"]
        x = torch.sum(R1 * R2, dim=-1)
        y = torch.cross(R1, R2)
        y = torch.norm(y, dim=-1)
        angle = torch.atan2(y, x)
        # Transform via angles
        cbf = [f(angle) for f in self.sbf_layer.get_sph_funcs()]
        cbf = torch.stack(cbf, dim=1)  # [None, 7]
        cbf = cbf.repeat_interleave(self.num_radial, dim=1)  # [None, 42]
        sbf = edges.src["rbf_env"] * cbf  # [None, 42]
        return {"sbf": sbf}

    def forward(self, g):
        # g,lg=g
        # g,lg=g
        l_g = dgl.line_graph(g, backtracking=False)
        g.edata["o"] = g.edata.pop("r")
        # g.edata['o']=g.edata['r']
        g.edata["d"] = torch.norm(g.edata["o"], dim=1)
        # add rbf features for each edge in one batch graph, [num_radial,]
        # print ('g',g)
        g = self.rbf_layer(g)
        # Embedding block
        g = self.emb_block(g)
        # Output block
        P = self.output_blocks[0](g)  # [batch_size, num_targets]
        # Prepare sbf feature before the following blocks
        for k, v in g.edata.items():
            l_g.ndata[k] = v

        l_g.apply_edges(self.edge_init)
        # Interaction blocks
        for i in range(self.num_blocks):
            g = self.interaction_blocks[i](g, l_g)
            P += self.output_blocks[i + 1](g)

        return P
