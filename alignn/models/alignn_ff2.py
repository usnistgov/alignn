"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""

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
    RBFExpansion,
    BesselExpansion,
    SphericalHarmonicsExpansion,
    FourierExpansion,
    compute_pair_vector_and_distance,
    check_line_graph,
    cutoff_function_based_edges,
)
from alignn.graphs import compute_bond_cosines
from alignn.utils import BaseSettings
from matgl.layers._basis import RadialBesselFunction,FourierExpansion
from matgl.layers import MLP_norm

torch.autograd.set_detect_anomaly(True)


class ALIGNNFF2Config(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn_ff2"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    output_features: int = 1
    grad_multiplier: int = -1
    calculate_gradient: bool = True
    atomwise_output_features: int = 0
    graphwise_weight: float = 1.0
    gradwise_weight: float = 1.0
    stresswise_weight: float = 0.00001
    atomwise_weight: float = 0.0
    classification: bool = False
    force_mult_natoms: bool = True
    use_cutoff_function: bool = True
    inner_cutoff: float = 4  # Ansgtrom
    stress_multiplier: float = 1
    add_reverse_forces: bool = False  # will make True as default soon
    batch_stress: bool = True
    multiply_cutoff: bool = False
    extra_features: int = 0
    exponent: int = 3
    bond_exp_basis: str = "gaussian"  # "bessel"  # or gaussian
    angle_exp_basis: str = "gaussian"  # "bessel"  # or gaussian
    max_n: int = 9
    max_f: int = 4
    learn_basis: bool = True


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



class AtomWise(nn.Module):
    """A class representing an interatomic potential."""

    __version__ = 3

    def __init__(
        self,
        model: nn.Module,
        data_mean: torch.Tensor | float = 0.0,
        data_std: torch.Tensor | float = 1.0,
        #element_refs: np.ndarray | None = None,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
        calc_magmom: bool = False,
        calc_repuls: bool = False,
        zbl_trainable: bool = False,
        debug_mode: bool = False,
    ):
        """Initialize Potential from a model and elemental references.

        Args:
            model: Model for predicting energies.
            data_mean: Mean of target.
            data_std: Std dev of target.
            element_refs: Element reference values for each element.
            calc_forces: Enable force calculations.
            calc_stresses: Enable stress calculations.
            calc_hessian: Enable hessian calculations.
            calc_magmom: Enable site-wise property calculation.
            calc_repuls: Whether the ZBL repulsion is included
            zbl_trainable: Whether zbl repulsion is trainable
            debug_mode: Return gradient of total energy with respect to atomic positions and lattices for checking
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.calc_forces = calc_forces
        self.calc_stresses = calc_stresses
        self.calc_hessian = calc_hessian
        self.calc_magmom = calc_magmom
        #self.element_refs: AtomRef | None
        self.debug_mode = debug_mode
        self.calc_repuls = calc_repuls

        if calc_repuls:
            self.repuls = NuclearRepulsion(self.model.cutoff, trainable=zbl_trainable)

        if element_refs is not None:
            self.element_refs = AtomRef(property_offset=torch.tensor(element_refs, dtype=matgl.float_th))
        else:
            self.element_refs = None
        # for backward compatibility
        if data_mean is None:
            data_mean = 0.0
        self.register_buffer("data_mean", torch.tensor(data_mean, dtype=matgl.float_th))
        self.register_buffer("data_std", torch.tensor(data_std, dtype=matgl.float_th))

    def forward(
        self,
        g: dgl.DGLGraph,
        lat: torch.Tensor,
        lg: dgl.DGLGraph | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            g: DGL graph
            lat: lattice
            state_attr: State attrs
            l_g: Line graph.

        Returns:
            (energies, forces, stresses, hessian) or (energies, forces, stresses, hessian, site-wise properties)
        """
        # st (strain) for stress calculations
        result = {}
        #st = lat.new_zeros([g.batch_size, 3, 3])
        #if self.calc_stresses:
        #    st.requires_grad_(True)
        lattice = lat @ (torch.eye(3, device=lat.device) + st)
        g.edata["lattice"] = torch.repeat_interleave(lattice, g.batch_num_edges(), dim=0)
        g.edata["pbc_offshift"] = (g.edata["images"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        if self.calc_forces:
            g.ndata["pos"].requires_grad_(True)

        total_energies = self.model(g,lg)

        total_energies = self.data_std * total_energies + self.data_mean

        if self.calc_repuls:
            total_energies += self.repuls(self.model.element_types, g)

        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_refs(g))
            total_energies += property_offset

        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)

        grad_vars = [g.ndata["pos"], st] if self.calc_stresses else [g.ndata["pos"]]

        if self.calc_forces:
            grads = grad(
                total_energies,
                grad_vars,
                grad_outputs=torch.ones_like(total_energies),
                create_graph=True,
                retain_graph=True,
            )
            forces = -grads[0]

        if self.calc_hessian:
            r = -grads[0].view(-1)
            s = r.size(0)
            hessian = total_energies.new_zeros((s, s))
            for iatom in range(s):
                tmp = grad([r[iatom]], g.ndata["pos"], retain_graph=iatom < s)[0]
                if tmp is not None:
                    hessian[iatom] = tmp.view(-1)

        if self.calc_stresses:
            volume = (
                torch.abs(torch.det(lattice.float())).half()
                if matgl.float_th == torch.float16
                else torch.abs(torch.det(lattice))
            )
            sts = -grads[1]
            scale = 1.0 / volume * -160.21766208
            sts = [i * j for i, j in zip(sts, scale)] if sts.dim() == 3 else [sts * scale]
            stresses = torch.cat(sts)

        if self.debug_mode:
            return total_energies, grads[0], grads[1]

        if self.calc_magmom:
            return total_energies, forces, stresses, hessian, g.ndata["magmom"]
        result['out']=total_energies
        result['grad']=forces
        result['stresses']=stresses
        result['atomwise_pred']=atomwise_pred

        return result


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
        if self.config.bond_exp_basis == "bessel":
            self.bond_expansion = RadialBesselFunction(max_n=config.max_n, cutoff=config.inner_cutoff, learnable=False) 
            self.edge_embedding = MLP_norm([config.max_n, config.hidden_features],bias_last=False)
            #self.edge_embedding = MLP_norm([config.edge_input_features, config.hidden_features],bias_last=False)
            #self.bond_expansion = RadialBesselFunction(max_n=config.edge_input_features, cutoff=config.inner_cutoff, learnable=True) 
        else:
            self.edge_embedding = nn.Sequential(
                RBFExpansion(
                    vmin=0,
                    vmax=8.0,
                    bins=config.edge_input_features,
                ),
                MLPLayer(
                    config.edge_input_features, config.embedding_features
                ),
                MLPLayer(config.embedding_features, config.hidden_features),
            )
        if self.config.angle_exp_basis == "spherical":
            self.angle_embedding = nn.Sequential(
                SphericalHarmonicsExpansion(),
                MLPLayer(
                    config.triplet_input_features, config.embedding_features
                ),
                MLPLayer(config.embedding_features, config.hidden_features),
            )  # not tested
        elif self.config.angle_exp_basis == "bessel":
            self.angle_expansion = FourierExpansion(max_f=config.max_f,  learnable=False) 
            self.angle_embedding = MLP_norm([2*config.max_f+1, config.hidden_features],bias_last=False)
        elif self.config.angle_exp_basis == "fourier":
            self.angle_embedding = nn.Sequential(
                FourierExpansion(),
                MLPLayer(
                    config.triplet_input_features, config.embedding_features
                ),
                MLPLayer(config.embedding_features, config.hidden_features),
            )  # not tested
        else:
            self.angle_embedding = nn.Sequential(
                RBFExpansion(
                    vmin=-1.0,
                    vmax=1.0,
                    bins=config.triplet_input_features,
                ),
                MLPLayer(
                    config.triplet_input_features, config.embedding_features
                ),
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

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        result = {}
        if len(self.alignn_layers) > 0:
            g, lg = g
            lg = lg.local_var()
        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            features = self.extra_feature_embedding(features)
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)
        check_lg=True
        g.edata["pbc_offshift"] =  (g.edata["images"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
        g.ndata["cart_coords"] = (g.ndata["frac_coords"].unsqueeze(dim=-1) * g.ndata["lattice"][0]).sum(dim=1)
        if self.config.calculate_gradient:
            #g.edata["images"] = (g.edata["images"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
            #torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)).sum(dim=1)
            g.ndata["cart_coords"].requires_grad_(True)
        r, bondlength = compute_pair_vector_and_distance(g)
        bondlength = torch.norm(r, dim=1)
        g.edata["d"] = bondlength
        g.edata["r"] = r
        # bond_expansion = self.bond_expansion(bondlength)
        # z = self.angle_embedding(lg.edata.pop("h"))
        lg = check_line_graph(g, lg, self.config.inner_cutoff)
        lg.apply_edges(compute_bond_cosines)

        # smooth_cutoff = polynomial_cutoff(
        #    bond_expansion, self.config.inner_cutoff, self.config.exponent
        # )
        # bond_expansion *= smooth_cutoff
        # g.edata["bond_expansion"] = (
        #    bond_expansion  # smooth_cutoff * bond_expansion
        # )

        if self.config.bond_exp_basis=='bessel':
         z = self.angle_embedding(self.angle_expansion(lg.edata.pop("h")))
         y = self.edge_embedding(self.bond_expansion(bondlength))
        else:
           y = self.edge_embedding(bondlength)
           z = self.angle_embedding(lg.edata.pop("h"))
         
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
            #print('bondlength',bondlength,bondlength.shape)
            y = self.edge_embedding(bondlength)
        #"""
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

        if self.config.calculate_gradient:
            en_out = out
            #g.edata["images"] = (g.edata["images"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
            #torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)).sum(dim=1)
            #g.ndata["cart_coords"].requires_grad_(True)
            grad_vars = [g.ndata["cart_coords"]]
            grads = grad(g.num_nodes()*en_out,grad_vars,grad_outputs=torch.ones_like(en_out),create_graph=True,retain_graph=True)
            forces_out = -grads[0] 

            # force calculation based on bond displacement vectors
            # autograd gives dE / d{r_{i->j}}
            pair_forces = (
                self.config.grad_multiplier
                * grad(
                    en_out,
                    r,
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
            # print('forces',forces)

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

        if self.classification:
            # out = torch.max(out,dim=1)
            out = self.softmax(out)
        result["out"] = out
        result["grad"] = forces_out
        result["stresses"] = stress
        result["atomwise_pred"] = atomwise_pred
        # print(result)
        return result
