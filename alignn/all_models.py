"""
DGL implementation.

GCNSimple, CGCNNSimple, ALIGNNEdge, ALIGNNSimple, ALIGNNCF.
"""
from typing import Optional
import numpy as np
import torch
from torch import nn
from typing import Tuple
import dgl
import dgl.function as fn

# import numpy as np
# import torch
import torch.nn.functional as F
from dgl.nn import AvgPooling, CFConv, GraphConv
from pydantic.typing import Literal

# from torch import nn

# from jarvisdgl.config import CGCNNConfig

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


class CGCNNConv(nn.Module):
    """Xie and Grossman graph convolution function.

    10.1103/PhysRevLett.120.145301
    """

    def __init__(
        self,
        node_features: int = 64,
        edge_features: int = 32,
        return_messages: bool = False,
    ):
        """Initialize torch modules for CGCNNConv layer."""
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.return_messages = return_messages

        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.linear_src = nn.Linear(node_features, 2 * node_features)
        self.linear_dst = nn.Linear(node_features, 2 * node_features)
        self.linear_edge = nn.Linear(edge_features, 2 * node_features)
        self.bn_message = nn.BatchNorm1d(2 * node_features)

        # final batchnorm
        self.bn = nn.BatchNorm1d(node_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """CGCNN convolution defined in Eq 5.

        10.1103/PhysRevLett.120.14530
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # compute edge messages -- coalesce W_f and W_s from the paper
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))
        g.ndata["h_src"] = self.linear_src(node_feats)
        g.ndata["h_dst"] = self.linear_dst(node_feats)
        g.apply_edges(fn.u_add_v("h_src", "h_dst", "h_nodes"))
        m = g.edata.pop("h_nodes") + self.linear_edge(edge_feats)
        m = self.bn_message(m)

        # split messages into W_f and W_s terms
        # multiply output of atom interaction net and edge attention net
        # i.e. compute the term inside the summation in eq 5
        # σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        h_f, h_s = torch.chunk(m, 2, dim=1)
        m = F.sigmoid(h_f) * F.softplus(h_s)
        g.edata["m"] = m

        # apply the convolution term in eq. 5 (without residual connection)
        # storing the results in edge features `h`
        g.update_all(
            message_func=fn.copy_e("m", "z"),
            reduce_func=fn.sum("z", "h"),
        )

        # final batchnorm
        h = self.bn(g.ndata.pop("h"))

        # residual connection plus nonlinearity
        out = F.softplus(node_feats + h)

        if self.return_messages:
            return out, m

        return out


class ALIGNNSimple(nn.Module):
    """CGCNN dgl implementation."""

    def __init__(self, config):
        """Set up CGCNN modules."""
        super().__init__()
        print(config)
        print("ALIGNNSimple")
        self.rbf = RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_features)
        self.abf = RBFExpansion(
            vmin=-np.pi / 2, vmax=np.pi / 2, bins=config.angle_features
        )
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )

        self.conv_layers = nn.ModuleList(
            [
                CGCNNConv(config.node_features, config.edge_features)
                for _ in range(config.conv_layers)
            ]
        )
        self.conv_layers2 = nn.ModuleList(
            [
                CGCNNConv(config.edge_features, config.angle_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.readout = AvgPooling()

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.Softplus()
        )

        if config.zero_inflated:
            # add latent Bernoulli variable model to zero out
            # predictions in non-negative regression model
            self.zero_inflated = True
            self.fc_nonzero = nn.Linear(config.fc_features, 1)
            self.fc_scale = nn.Linear(config.fc_features, 1)
            # self.fc_shape = nn.Linear(config.fc_features, 1)
            self.fc_scale.bias.data = torch.tensor(
                # np.log(2.1), dtype=torch.float
                2.1,
                dtype=torch.float,
            )
        else:
            self.zero_inflated = False
            self.fc_out = nn.Linear(config.fc_features, config.output_features)

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            if not self.zero_inflated:
                self.fc_out.bias.data = torch.tensor(
                    np.log(avg_gap), dtype=torch.float
                )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self, g: dgl.DGLGraph, mode=Literal["train", "predict"]
    ) -> torch.Tensor:
        """CGCNN function mapping graph to outputs."""
        g, lg = g
        g = g.local_var()
        lg = lg.local_var()

        # fixed edge features: RBF-expanded bondlengths
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        edge_features = self.rbf(bondlength)

        # initial node features: atom feature network...
        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(v)
        angle_features = self.abf(lg.edata["h"])
        # CGCNN-Conv block: update node features
        for conv_layer, conv_layer2 in zip(
            self.conv_layers, self.conv_layers2
        ):
            node_features = conv_layer(g, node_features, edge_features)
            edge_features = conv_layer2(lg, edge_features, angle_features)

        # crystal-level readout
        features = self.readout(g, node_features)
        features = F.softplus(features)
        features = self.fc(features)
        features = F.softplus(features)

        if self.zero_inflated:
            logit_p = self.fc_nonzero(features)
            log_scale = self.fc_scale(features)
            # log_shape = self.fc_shape(features)

            # pred = (torch.sigmoid(logit_p)
            #         * torch.exp(log_scale)
            #         * torch.exp(log_shape))
            # out = torch.where(p < 0.5, torch.zeros_like(out), out)
            return (
                torch.squeeze(logit_p),
                torch.squeeze(log_scale),
                # torch.squeeze(log_shape),
            )

        else:
            out = self.fc_out(features)
            if self.link:
                out = self.link(out)

        return torch.squeeze(out)

        # return bce_loss + torch.tensor(2.0) * g_loss.sum() / indicator.sum()


class CGCNNConvSimple(nn.Module):
    """Xie and Grossman graph convolution function.

    10.1103/PhysRevLett.120.145301
    """

    def __init__(self, node_features: int = 64, edge_features: int = 32):
        """Initialize torch modules for CGCNNConv layer."""
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features

        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        in_feats = 2 * self.node_features + self.edge_features

        # edge interaction model (W_f)
        self.edge_interaction = nn.Sequential(
            nn.Linear(in_feats, self.node_features),
            nn.BatchNorm1d(self.node_features),
            nn.Sigmoid(),
        )

        # edge attention model (W_s)
        self.edge_update = nn.Sequential(
            nn.Linear(in_feats, self.node_features),
            nn.BatchNorm1d(self.node_features),
            nn.Softplus(),
        )

        # final batchnorm
        self.bn = nn.BatchNorm1d(self.node_features)

    def combine_edge_features(self, edges):
        """Edge update for CGCNNConv.

        concatenate source and destination node features with edge features
        then apply the edge update modulated by the edge interaction model
        """
        # form augmented edge features z_ij = [v_i, v_j, u_ij]
        z = torch.cat((edges.src["h"], edges.dst["h"], edges.data["h"]), dim=1)

        # multiply output of atom interaction net and edge attention net
        # i.e. compute the term inside the summation in eq 5
        # σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        return {"z": self.edge_interaction(z) * self.edge_update(z)}

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """CGCNN convolution defined in Eq 5.

        10.1103/PhysRevLett.120.14530
        """
        g = g.local_var()

        g.ndata["h"] = node_feats
        g.edata["h"] = edge_feats

        # apply the convolution term in eq. 5 (without residual connection)
        # storing the results in edge features `h`
        g.update_all(
            message_func=self.combine_edge_features,
            reduce_func=fn.sum("z", "h"),
        )

        # final batchnorm
        h = self.bn(g.ndata.pop("h"))

        # residual connection plus nonlinearity
        return F.softplus(node_feats + h)


class CGCNNSimple(nn.Module):
    """CGCNN dgl implementation."""

    def __init__(self, config):
        """Set up CGCNN modules."""
        super().__init__()
        print(config)
        print("CGCNNOld")

        self.rbf = RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_features)
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )

        self.conv_layers = nn.ModuleList(
            [
                CGCNNConvSimple(config.node_features, config.edge_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.readout = AvgPooling()

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.Softplus()
        )

        self.fc_out = nn.Linear(config.fc_features, config.output_features)
        # self.logscale = config.logscale

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """CGCNN function mapping graph to outputs."""
        g, lg = g
        g = g.local_var()

        # fixed edge features: RBF-expanded bondlengths
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        edge_features = self.rbf(bondlength)

        # initial node features: atom feature network...
        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(v)

        # CGCNN-Conv block: update node features
        for conv_layer in self.conv_layers:
            node_features = conv_layer(g, node_features, edge_features)

        # crystal-level readout
        features = self.readout(g, node_features)
        features = F.softplus(features)
        features = self.fc(features)
        features = F.softplus(features)

        out = self.fc_out(features)

        # if self.logscale:
        #    out = torch.exp(out)

        return torch.squeeze(out)


##################################
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
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

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


class ALIGNNConvEdge(nn.Module):
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
        """Linear, Batchnorm, softplus layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.Softplus(),
        )

    def forward(self, x):
        """Linear, Batchnorm, softplus layer."""
        return self.layer(x)


class ALIGNNEdge(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.atom_embedding = MLPLayer(
            config.node_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
                lengthscale=0.5,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-np.pi / 2,
                vmax=np.pi / 2,
                bins=config.triplet_input_features,
                # vmin=-np.pi / 2,
                # vmax=np.pi / 2,
                # bins=config.triplet_input_features,
                # lengthscale=0.1,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConvEdge(
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

        self.fc = nn.Linear(config.hidden_features, config.output_features)

        # self.link = lambda x: x
        # self.link = torch.exp
        self.link = lambda x: x
        """
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
        """

    def forward(self, g):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        g, lg = g
        g = g.local_var()
        lg = lg.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # angle features (fixed)
        z = self.angle_embedding(lg.edata.pop("h"))

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-relu-pool-classify
        h = self.readout(g, x)
        out = self.fc(h)

        if self.link:
            out = self.link(out)

        return torch.squeeze(out)


class CLGNLayer(nn.Module):
    """Crystal line graph network layer."""

    def __init__(
        self,
        node_in_feats: int,
        node_out_feats: int,
        edge_in_feats: int,
        edge_out_feats: int,
        angle_in_feats: int,
        hidden_feats: int,
    ):
        """Initialize CLGN layer."""
        super().__init__()
        # self.bn = nn.BatchNorm1d(node_out_feats)

        self.project_node = nn.Linear(node_in_feats, hidden_feats)
        self.project_out = nn.Linear(hidden_feats, node_out_feats)
        self.project_edge = nn.Sequential(
            nn.Linear(edge_in_feats, hidden_feats),
            nn.Softplus(),
            nn.Linear(hidden_feats, hidden_feats),
        )

        # self.g_conv = CFConv(
        #     node_in_feats, edge_in_feats, hidden_feats, node_out_feats
        # )

        self.lg_conv = CFConv(
            edge_in_feats + hidden_feats,
            angle_in_feats,
            hidden_feats,
            edge_out_feats,
        )

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for CLGN layer."""
        # x: g.ndata["h"]
        # y: lg.ndata["h"]
        # z: lg.edata["h"]

        # node update
        # like CFConv, but save edge messages to fuse to line graph
        # x = self.g_conv(g, x, y)
        g.srcdata["hv"] = self.project_node(x)
        g.edata["he"] = self.project_edge(y)
        g.apply_edges(fn.u_mul_e("hv", "he", "m"))
        g.update_all(fn.copy_e("m", "m"), fn.sum("m", "hv"))
        x = self.project_out(g.ndata.pop("hv"))
        x = F.softplus(x)

        # edge update: CFConv
        # concatenate edge features and edge messages
        y = torch.cat((y, g.edata.pop("m")), 1)
        y = self.lg_conv(lg, y, z)
        y = F.softplus(y)

        return x, y


class ALIGNNCF(nn.Module):
    """Line graph network."""

    def __init__(self, config):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        print(config)

        self.rbf = RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_features)
        self.angle_bf = RBFExpansion(
            vmin=-1, vmax=1.0, bins=config.angle_features
        )
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )

        self.bn = nn.BatchNorm1d(config.node_features)

        self.conv1 = CLGNLayer(
            config.node_features,
            config.node_features,
            config.edge_features,
            config.edge_features,
            config.angle_features,
            config.hidden_features,
        )
        self.conv2 = CLGNLayer(
            config.node_features,
            config.node_features,
            config.edge_features,
            config.edge_features,
            config.angle_features,
            config.hidden_features,
        )
        self.conv3 = CFConv(
            config.node_features,
            config.edge_features,
            config.hidden_features,
            config.node_features,
        )

        self.bn_final = nn.BatchNorm1d(config.node_features)

        self.readout = AvgPooling()

        self.fc = nn.Linear(config.node_features, config.output_features)

    def forward(self, g):
        """Baseline SimpleGCN : start with `atom_features`."""
        g, lg = g
        g = g.local_var()

        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        edge_features = self.rbf(bondlength)

        # initial node features: atom feature network...
        # conv-bn-relu
        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(v)
        node_features = F.relu(self.bn(node_features))

        angle_features = self.angle_bf(lg.edata.pop("h"))
        x, y = self.conv1(g, lg, node_features, edge_features, angle_features)
        x, y = self.conv2(g, lg, x, y, angle_features)
        x = self.conv3(g, x, y)

        # norm-relu-pool-classify
        h = F.relu(self.bn_final(x))

        h = self.readout(g, h)

        out = self.fc(h)

        return torch.squeeze(out)


class GCNSimple(nn.Module):
    """GraphConv GCN with DenseNet-style connections."""

    def __init__(self, config):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.edge_lengthscale = 1  # config.edge_lengthscale
        self.weight_edges = 1  # config.weight_edges
        config.width = 10
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.width
        )

        self.layer1 = GraphConv(config.width, config.width)
        self.layer2 = GraphConv(config.width, config.output_features)
        self.readout = AvgPooling()

    def forward(self, g):
        """Baseline SimpleGCN : start with `atom_features`."""
        g, lg = g
        g = g.local_var()

        r = torch.norm(g.edata.pop("r"), dim=1)
        edge_weights = torch.exp(-(r ** 2) / self.edge_lengthscale ** 2)

        # initial node features: atom feature network...
        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(v)

        x = F.relu(self.layer1(g, node_features, edge_weight=edge_weights))
        x = self.layer2(g, x, edge_weight=edge_weights)
        x = self.readout(g, x)

        return torch.squeeze(x)


class ZeroInflatedGammaLoss(nn.modules.loss._Loss):
    """Zero inflated Gamma regression loss."""

    def predict(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        """Combine ZIG multi-part outputs to yield real-valued predictions."""
        # logit_p, log_scale, log_shape = inputs
        logit_p, log_scale = inputs
        return (
            torch.sigmoid(logit_p)
            * F.softplus(log_scale)
            # * torch.exp(log_scale)
            # * (1 + torch.exp(log_shape))
        )

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-inflated Gamma loss.

        binary crossentropy loss combined with Gamma negative log likelihood
        """
        # logit_p, log_scale, log_shape = inputs
        logit_p, log_scale = inputs

        bce_loss = F.binary_cross_entropy_with_logits(
            logit_p, target, reduction="sum"
        )

        indicator = target > 0
        # g_loss = F.mse_loss(
        #     log_scale[indicator],
        #     torch.log(target[indicator]), reduction="sum"
        # )
        # g_loss = F.mse_loss(
        #     torch.exp(log_scale[indicator]),
        # target[indicator], reduction="sum"
        # )
        g_loss = F.mse_loss(
            F.softplus(log_scale[indicator]),
            target[indicator],
            reduction="sum",
        )

        return (bce_loss + g_loss) / target.numel()


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
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return torch.exp(
            -self.gamma * (distance.unsqueeze(1) - (self.centers)) ** 2
        )
