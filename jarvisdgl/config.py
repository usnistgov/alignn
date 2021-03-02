"""Pydantic model for default configuration and validation."""

from enum import Enum, auto
from typing import Optional

from pydantic import BaseModel


class AutoName(Enum):
    """Auto string enum.

    https://stackoverflow.com/questions/44781681/how-to-compare-a-string-with-a-python-enum/44785241#44785241
    """

    def _generate_next_value_(name, start, count, last_values):
        return name


class DatasetEnum(AutoName):
    """Supported datasets."""

    dft_3d = auto()
    dft_2d = auto()


class TargetEnum(AutoName):
    """Supported targets."""

    formation_energy_peratom = auto()
    optb88vdw_bandgap = auto()


class FeatureEnum(AutoName):
    """Supported atom feature sets."""

    basic = auto()
    atomic_number = auto()
    cfid = auto()


class CriterionEnum(AutoName):
    """Supported optimizer criteria."""

    mse = auto()
    l1 = auto()


FEATURESET_SIZE = {"basic": 11, "atomic_number": 1, "cfid": 438}


class OptimizerEnum(AutoName):
    """Supported optimizers."""

    adamw = auto()
    sgd = auto()


class SchedulerEnum(AutoName):
    """Supported learning rate schedulers."""

    none = auto()
    onecycle = auto()


class TrainingConfig(BaseModel):
    """Training config defaults and validation."""

    # dataset configuration
    dataset: DatasetEnum = DatasetEnum.dft_3d
    target: TargetEnum = TargetEnum.formation_energy_peratom

    # logging configuration

    # training configuration
    random_seed: Optional[int] = None
    n_val: int = 1024
    n_train: int = 1024

    epochs: int = 100
    batch_size: int = 32
    weight_decay: float = 0
    learning_rate: float = 1e-2
    criterion: CriterionEnum = CriterionEnum.mse
    atom_features: FeatureEnum = FeatureEnum.basic
    optimizer: OptimizerEnum = OptimizerEnum.adamw
    scheduler: SchedulerEnum = SchedulerEnum.onecycle

    # model configuration
    conv_layers: int = 3
    edge_features: int = 16
    node_features: int = 64

    node_features: int = 32
    edge_features: int = 32

    fc_layers: int = 1
    fc_features: int = 64
    output_features: int = 1

    # if logscale is set, apply `exp` to final outputs
    # to constrain predictions to be positive
    logscale: bool = False

    @property
    def atom_input_features(self):
        """Automatically configure node feature dimensionality."""
        return FEATURESET_SIZE[self.atom_features.value]
