from __future__ import annotations

# pylint: disable=too-many-public-methods
from typing import Any, Dict, List, NamedTuple, Tuple  # pylint: disable=unused-import   # noqa: F401

import numpy as np
import torch
from typing_extensions import TypeAlias

# Imported for typing with PyLance.
import poutyne  # pylint: disable=unused-import
from poutyne.framework.callbacks.callbacks import Callback

NetworkIOType: TypeAlias = "Dict[str, NetworkIOType] | List[NetworkIOType] | Tuple[NetworkIOType, ...] | torch.Tensor"
MetricReturnType: TypeAlias = "torch.Tensor | float | np.ndarray | Dict[str, torch.Tensor | float | np.ndarray]"


class StepOutput(NamedTuple):
    loss: torch.Tensor | float | None = None
    batch_metrics: List[NetworkIOType] | np.ndarray | None = None
    y_pred: NetworkIOType | None = None
    y_true: NetworkIOType | None = None
    x: NetworkIOType | None = None


class BaseStrategy:
    def set_params(self, params: Dict[str, Any]) -> None:
        self.params = params

    def set_model(self, model: 'poutyne.Model') -> None:
        self.model = model

    def compute_loss(self) -> float:
        pass

    def reset_loss(self) -> None:
        pass

    def get_batch_metric_names(self) -> List[str]:
        return []

    def get_epoch_metric_names(self) -> List[str]:
        return []

    def compute_batch_metrics(self) -> List[MetricReturnType]:
        return []

    def compute_epoch_metrics(self) -> List[MetricReturnType]:
        return []

    def reset_batch_metrics(self) -> None:
        pass

    def reset_epoch_metrics(self) -> None:
        pass

    def train_step(
        self, data: NetworkIOType, *, callback: Callback | None = None, step: int | None = None, **kwargs
    ) -> StepOutput:
        pass

    def on_epoch_begin(self, epoch_number: int, logs: Dict) -> None:
        pass

    def on_epoch_end(self, epoch_number: int, logs: Dict) -> None:
        pass

    def on_train_begin(self, logs: Dict) -> None:
        pass

    def on_train_end(self, logs: Dict) -> None:
        pass

    def test_step(self, data, **kwargs: Any) -> StepOutput:
        pass

    def on_valid_begin(self, logs: Dict) -> None:
        pass

    def on_valid_end(self, logs: Dict) -> None:
        pass

    def on_test_begin(self, logs: Dict) -> None:
        pass

    def on_test_end(self, logs: Dict) -> None:
        pass

    def predict_step(self, data, **kwargs: Any) -> NetworkIOType:
        pass

    def on_predict_begin(self, logs: Dict) -> None:
        pass

    def on_predict_end(self, logs: Dict) -> None:
        pass
