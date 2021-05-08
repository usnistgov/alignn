"""A baseline graph convolution network dgl implementation."""
from typing import List, Optional

import dgl
import torch
from dgl.nn import AvgPooling, GraphConv
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F

from alignn.utils import BaseSettings


class DenseGCNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.densegcn."""

    name: Literal["densegcn"]
    atom_input_features: int = 1
    edge_lengthscale: float = 4.0
    weight_edges: bool = True
    conv_layers: int = 4
    node_features: int = 32
    growth_rate: int = 32
    output_features: int = 1
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class _DenseLayer(nn.Module):
    """BatchNorm-ReLU-GraphConv Dense layer."""

    def __init__(self, in_features: int, growth_rate: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.conv = GraphConv(in_features, growth_rate)

    def forward(
        self,
        g: dgl.DGLGraph,
        input: List[torch.Tensor],
        edge_weight: Optional[torch.Tensor],
    ):

        prev_features = F.relu(self.bn(torch.cat(input, 1)))
        new_features = self.conv(g, prev_features, edge_weight=edge_weight)

        return new_features


class _DenseBlock(nn.ModuleDict):
    """Block of densely-connected bn-ReLU-conv layers."""

    def __init__(self, n_layers: int, in_features: int, growth_rate: int):
        super().__init__()
        for id_layer in range(n_layers):
            layer = _DenseLayer(
                in_features + id_layer * growth_rate, growth_rate
            )
            self.add_module(f"denselayer{1+id_layer}", layer)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_features: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        features = [node_features]
        for name, layer in self.items():
            new_features = layer(g, features, edge_weight=edge_weight)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseGCN(nn.Module):
    """GraphConv GCN with DenseNet-style connections."""

    def __init__(
        self, config: DenseGCNConfig = DenseGCNConfig(name="densegcn")
    ):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        print(config)
        self.edge_lengthscale = config.edge_lengthscale
        self.weight_edges = config.weight_edges

        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
        )

        self.bn = nn.BatchNorm1d(config.node_features)

        # bn-relu-conv
        self.dense_layers = _DenseBlock(
            config.conv_layers, config.node_features, config.growth_rate
        )

        final_size = (
            config.node_features + config.conv_layers * config.growth_rate
        )

        self.bn_final = nn.BatchNorm1d(final_size)

        self.readout = AvgPooling()

        self.fc = nn.Linear(final_size, config.output_features)

    def forward(self, g):
        """Baseline SimpleGCN : start with `atom_features`."""
        g = g.local_var()

        if self.weight_edges:
            r = torch.norm(g.edata["r"], dim=1)
            edge_weights = torch.exp(-(r ** 2) / self.edge_lengthscale ** 2)
        else:
            edge_weights = None

        # initial node features: atom feature network...
        # conv-bn-relu
        v = g.ndata.pop("atom_features")
        node_features = self.atom_embedding(v)
        node_features = F.relu(self.bn(node_features))

        # bn-relu-conv
        h = self.dense_layers(g, node_features, edge_weight=edge_weights)

        # norm-relu-pool-classify
        h = F.relu(self.bn_final(h))

        h = self.readout(g, h)

        out = self.fc(h)

        return torch.squeeze(out)
