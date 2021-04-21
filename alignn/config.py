"""Pydantic model for default configuration and validation."""

import subprocess
from typing import Optional, Union

from pydantic import Field, root_validator, validator
from pydantic.typing import Literal

from alignn import models
from alignn.utils import BaseSettings

VERSION = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
)


FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438, "cgcnn": 92}


TARGET_ENUM = Literal[
    "formation_energy_peratom",
    "optb88vdw_bandgap",
    "gap pbe",
    "e_form",
    "U0",
]


class TrainingConfig(BaseSettings):
    """Training config defaults and validation."""

    version: str = VERSION

    # dataset configuration
    dataset: Literal["dft_3d", "dft_2d", "megnet", "qm9"] = "dft_3d"
    target: TARGET_ENUM = "formation_energy_peratom"
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
    warmup_steps: int = 2000
    criterion: Literal["mse", "l1", "poisson", "zig"] = "mse"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    scheduler: Literal["onecycle", "none"] = "onecycle"

    # model configuration
    model: Union[
        models.CGCNNConfig,
        models.ICGCNNConfig,
        models.SimpleGCNConfig,
        models.DenseGCNConfig,
        models.ALIGNNConfig,
    ] = models.CGCNNConfig(name="cgcnn")

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
