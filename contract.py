from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class GlobalConfig:
    project_name: Optional[str]
    seed: int
    deterministic: bool


@dataclass
class TrainerConfig:
    epochs: int
    run_name: Optional[str]
    learning_rate: float
    batch_size: int
    patience: int
    save_best_model_on: Optional[str]
    epsilon: float
    device: Optional[str]
    model_save_dir: str
    model_save_name: Optional[str]
    is_wandb: bool
    use_autocast: bool


@dataclass
class TrainReturnContract:
    loss: float
    targets: np.ndarray
    predictions: np.ndarray


@dataclass
class TestReturnContract:
    metric: float
    targets: np.ndarray
    predictions: np.ndarray
    metadata: List[Dict[str, Union[str, int, float]]]
