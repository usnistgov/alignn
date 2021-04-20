"""Pydantic model for default configuration and validation."""

import subprocess
from enum import Enum, auto
from typing import Optional, Union

from pydantic import BaseSettings as PydanticBaseSettings
from pydantic import Field, root_validator, validator
from pydantic.typing import Literal

VERSION = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
)


class BaseSettings(PydanticBaseSettings):
    """Add configuration to default Pydantic BaseSettings."""

    class Config:
        """Configure BaseSettings behavior."""

        extra = "forbid"
        use_enum_values = True
        env_prefix = "jv_"


FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438, "cgcnn": 92}


class CGCNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["cgcnn"]
    conv_layers: int = 3
    atom_input_features: int = 1
    edge_features: int = 16
    node_features: int = 64
    fc_layers: int = 1
    fc_features: int = 64
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class CLGNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.clgn."""

    name: Literal["clgn"]
    conv_layers: int = 3
    atom_input_features: int = 1
    edge_features: int = 16
    angle_features: int = 16
    node_features: int = 64
    hidden_features: int = 32
    fc_layers: int = 1
    fc_features: int = 64
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 2
    gcn_layers: int = 1
    node_input_features: int = 1
    edge_input_features: int = 40
    triplet_input_features: int = 16
    embedding_features: int = 64
    hidden_features: int = 64
    fc_layers: int = 1
    fc_features: int = 64
    output_features: int = 1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class ICGCNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.icgcnn."""

    name: Literal["icgcnn"]
    conv_layers: int = 3
    atom_input_features: int = 1
    edge_features: int = 16
    node_features: int = 64
    fc_layers: int = 1
    fc_features: int = 64
    output_features: int = 1

    # if logscale is set, apply `exp` to final outputs
    # to constrain predictions to be positive
    logscale: bool = False
    hurdle: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class SimpleGCNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.gcn."""

    name: Literal["simplegcn"]
    atom_input_features: int = 1
    weight_edges: bool = True
    width: int = 64
    output_features: int = 1

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


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

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    version: str = VERSION

    # dataset configuration
    dataset: Literal["dft_3d", "dft_2d"] = "dft_3d"
    target: Literal[
        "formation_energy_peratom", "optb88vdw_bandgap"
    ] = "formation_energy_peratom"
    atom_features: Literal["basic", "atomic_number", "cfid", "cgcnn"] = "basic"
    neighbor_strategy: Literal["k-nearest", "voronoi"] = "k-nearest"

    # logging configuration

    # training configuration
    random_seed: Optional[int] = None
    n_val: int = 1024
    n_train: int = 1024
    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0
    learning_rate: float = 1e-2
    criterion: Literal["mse", "l1", "poisson", "zig"] = "mse"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal["onecycle", "none"] = "onecycle"

    # model configuration
    model: Union[
        CGCNNConfig,
        ICGCNNConfig,
        SimpleGCNConfig,
        DenseGCNConfig,
        CLGNConfig,
        ALIGNNConfig,
    ] = CGCNNConfig(name="cgcnn")

    @root_validator()
    def set_input_size(cls, values):
        """Automatically configure node feature dimensionality."""
        values["model"].atom_input_features = FEATURESET_SIZE[
            values["atom_features"]
        ]

        return values

    # @property
    # def atom_input_features(self):
    #     """Automatically configure node feature dimensionality."""
    #     return FEATURESET_SIZE[self.atom_features]
