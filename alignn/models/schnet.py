# -*- coding:utf-8 -*-
#!/usr/bin/env python
# coding: utf-8
# Author Xinyu Li
# https://github.com/UON-comp-chem/GNNforCatalysis-DGL/tree/main/catgnn/layers
import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus
from dgl.nn.pytorch.glob import SumPooling, AvgPooling
from alignn.utils import BaseSettings
from pydantic.typing import Literal
import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn
from dgl.nn.pytorch.conv.cfconv import CFConv, ShiftedSoftplus
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
from torch.nn import Softplus
from dgl.nn.pytorch.conv.cfconv import CFConv, ShiftedSoftplus
from dgl.utils import expand_as_pair
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import torch.nn as nn


class SchNetConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["schnet"]
    atom_input_features: int = 1
    output_features: int = 1


class AtomEmbedding(nn.Module):
    """
    Convert the atom(node) list to atom embeddings.
    The atom with the same element share the same initial embeddding.
    """

    def __init__(self, dim=128, type_num=100, pre_train=None):
        """
        Randomly init the element embeddings.
        Args:
            dim: the dim of embeddings
            type_num: the largest atomic number of atoms in the dataset
            pre_train: the pre_trained embeddings
        """
        super().__init__()
        self._dim = dim
        self._type_num = type_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pre_train, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(type_num, dim, padding_idx=0)

    def forward(self, g, p_name="node_type"):
        """Input type is dgl graph"""
        nnode_feats = self.embedding(g.ndata.pop(p_name))
        return nnode_feats


class FakeAtomEmbedding(nn.Module):
    """
    Convert the atom(node) list to atom embeddings.
    The atom with the same element share the same initial embeddding.
    """

    def __init__(self, dim=128, type_num=300, pre_train=None):
        """
        Randomly init the element embeddings.
        Args:
            dim: the dim of embeddings
            type_num: the largest atomic number of atoms in the dataset
            pre_train: the pre_trained embeddings
        """
        super().__init__()
        self._dim = dim
        self._type_num = type_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pre_train, padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(type_num, dim, padding_idx=0)

    def forward(self, g, p_name="node_type"):
        """Input type is dgl graph"""
        node_type = g.ndata.pop(p_name) + 100 * g.ndata.pop("ls")
        nnode_feats = self.embedding(node_type)
        return nnode_feats


class GPEmbedding(nn.Module):
    """
    Convert the atom(node) list to group and period embeddings.
    The atom with the same element share the same initial embeddding.
    """

    def __init__(
        self,
        dim=128,
        type_group=18,
        type_period=7,
        dim_ratio_group=2 / 3,
    ):
        """
        Randomly init the element embeddings.
        Args:
            dim: the dim of embeddings
            type_num: the largest atomic number of atoms in the dataset
            pre_train: the pre_trained embeddings
        """
        super().__init__()
        self._dim = dim
        self._type_group = type_group
        self._type_period = type_period
        self._dim_group = int(dim * dim_ratio_group)
        self._dim_period = dim - self._dim_group
        self.gembedding = nn.Embedding(
            type_group, self._dim_group, padding_idx=0
        )
        self.pembedding = nn.Embedding(
            type_period, self._dim_period, padding_idx=0
        )

    def forward(self, g, p_name="node"):
        """Input type is dgl graph"""
        group_list = g.ndata.pop("group")
        period_list = g.ndata.pop("period")
        gembed = self.gembedding(group_list)
        pembed = self.pembedding(period_list)
        g.ndata[p_name] = torch.cat((gembed, pembed), dim=1)
        return g.ndata[p_name]


class GPLSEmbedding(nn.Module):
    """
    Convert the atom(node) list to group, period and label site embeddings.
    The atom with the same element share the same initial embeddding.
    """

    def __init__(
        self,
        dim=128,
        type_group=18,
        type_period=7,
        type_ls=3,
        dim_ratio_group=1 / 2,
        dim_ratio_period=1 / 4,
    ):
        """
        Randomly init the element embeddings.
        Args:
            dim:       the dim of embeddings
            type_num:  othe largest atomic number of atoms in the dataset
            pre_train: the pre_trained embeddings
        """
        super().__init__()
        self._dim = dim
        self._type_group = type_group
        self._type_period = type_period
        self._type_ls = type_ls
        # Set different dimension for group, period and LS should have big difference
        # Thus I suggess keep the LS as 0, 1 information
        self._dim_group = int(dim * dim_ratio_group)
        self._dim_period = int(dim * dim_ratio_period)
        self._dim_ls = dim - self._dim_group - self._dim_period
        self.gembedding = nn.Embedding(
            type_group, self._dim_group, padding_idx=0
        )
        self.pembedding = nn.Embedding(
            type_period, self._dim_period, padding_idx=0
        )
        self.lsembedding = nn.Embedding(type_ls, self._dim_ls, padding_idx=0)

    def forward(self, g, p_name="node"):
        """Input type is dgl graph"""
        group_list = g.ndata.pop("group")
        period_list = g.ndata.pop("period")
        ls_list = g.ndata.pop("ls")
        gembed = self.gembedding(group_list)
        pembed = self.pembedding(period_list)
        lsembed = self.lsembedding(ls_list)
        g.ndata[p_name] = torch.cat((gembed, pembed, lsembed), dim=1)
        return g.ndata[p_name]


def softplus_inverse(x):
    return x + np.log(-np.expm1(-x))


class RBFLayer(nn.Module):
    r"""RBF Layer"""

    def __init__(self, low=0, high=10.0, num_gaussians=128):
        super(RBFLayer, self).__init__()
        offset = torch.linspace(low, high, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)
        self._fan_out = num_gaussians

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

    def reset_parameters(self):
        pass


class PhysRBFLayer(nn.Module):
    r"""RBF layer used in PhysNet"""

    def __init__(self, low=0.0, high=10.0, num_gaussians=64):
        super(PhysRBFLayer, self).__init__()
        self.num_gaussians = num_gaussians
        center = softplus_inverse(
            np.linspace(1.0, np.exp(-high), num_gaussians)
        )
        width = [
            softplus_inverse(
                (0.5 / ((1.0 - np.exp(-high)) / num_gaussians)) ** 2
            )
        ] * num_gaussians
        self.register_buffer("high", torch.tensor(high, dtype=torch.float32))
        self.register_buffer("center", torch.tensor(center))
        self.register_buffer(
            "width",
            torch.tensor(width, dtype=torch.float32),
        )
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, r):
        rbf = torch.exp(
            -self.width
            * (torch.exp(-r.unsqueeze(-1)) - F.softplus(self.center)) ** 2
        )
        return rbf


class SchInteraction(nn.Module):
    """Building block for SchNet.
    SchNet is introduced in `SchNet: A continuous-filter convolutional neural network for
    modeling quantum interactions <https://arxiv.org/abs/1706.08566>`__.
    This layer combines node and edge features in message passing and updates node
    representations.
    Parameters
    ----------
    node_feats : int
        Size for the input and output node features.
    edge_in_feats : int
        Size for the input edge features.
    """

    def __init__(self, edge_in_feats, node_feats):
        super(SchInteraction, self).__init__()

        self.conv = CFConv(node_feats, edge_in_feats, node_feats, node_feats)
        self.project_out = nn.Linear(node_feats, node_feats)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for layer in self.conv.project_edge:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.conv.project_node.reset_parameters()
        self.conv.project_out[0].reset_parameters()
        self.project_out.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feats)
            Input node features, V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features, E for the number of edges.
        Returns
        -------
        float32 tensor of shape (V, node_feats)
            Updated node representations.
        """
        node_feats = self.conv(g, node_feats, edge_feats)
        return self.project_out(node_feats)


class SchNet(nn.Module):
    """
    SchNet Model from:
        Schütt, Kristof, et al.
        SchNet: A continuous-filter convolutional neural network
        for modeling quantum interactions. (NIPS'2017)
    """

    def __init__(self, config: SchNetConfig = SchNetConfig(name="schnet")):
        """
        Args:
            embed: Group and Period embeding to atomic number
                    Embedding
            dim: dimension of features
            output_dim: dimension of prediction
            cutoff: radius cutoff
            num_gaussians: dimension in the RBF function
            n_conv: number of interaction layers
            norm: normalization
        """
        super().__init__()
        embed = "gpls"
        embed = "atom"
        dim = 64
        hidden_dim = (64,)
        num_gaussians = 64
        cutoff = 5.0
        output_dim = 1
        n_conv = 3
        act = ShiftedSoftplus()
        aggregation_mode = "avg"
        norm = False
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.n_conv = n_conv
        self.norm = norm
        self.output_dim = output_dim
        self.aggregation_mode = aggregation_mode

        if act == None:
            self.activation = ShiftedSoftplus()
        else:
            self.activation = act

        assert embed in [
            "gpls",
            "atom",
            "gp",
        ], "Expect mode to be 'gpls' or 'atom' or 'gp', got {}".format(embed)
        if embed == "gpls":
            self.embedding_layer = GPLSEmbedding(dim)
        elif embed == "atom":
            self.embedding_layer = AtomEmbedding(dim)
        elif embed == "gp":
            self.embedding_layer = GPEmbedding(dim)

        self.rbf_layer = RBFLayer(0, cutoff, num_gaussians)
        self.conv_layers = nn.ModuleList(
            [
                SchInteraction(self.rbf_layer._fan_out, dim)
                for i in range(n_conv)
            ]
        )
        self.atom_dense_layer1 = nn.Linear(dim, int(dim / 2))
        self.atom_dense_layer2 = nn.Linear(int(dim / 2), output_dim)
        if self.aggregation_mode == "sum":
            self.readout = SumPooling()
        elif self.aggregation_mode == "avg":
            self.readout = AvgPooling()

    def set_mean_std(self, mean, std):
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, g):
        """g is the DGL.graph"""
        g.ndata["node_type"] = g.ndata["Z"]
        g.edata["distance"] = torch.norm(g.edata["r"], dim=1)

        node_feats = self.embedding_layer(g)
        edge_feats = self.rbf_layer(g.edata["distance"])

        for idx in range(self.n_conv):
            node_feats = self.conv_layers[idx](g, node_feats, edge_feats)

        atom = self.atom_dense_layer1(node_feats)
        atom = self.activation(atom)
        res = self.atom_dense_layer2(atom)

        if self.norm:
            res = res * self.std + self.mean

        res = self.readout(g, res)
        return res
