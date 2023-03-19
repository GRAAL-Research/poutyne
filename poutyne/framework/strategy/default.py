"""
Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

# pylint: disable=too-many-public-methods
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import PackedSequence

# Imported for typing with PyLance.
import poutyne  # pylint: disable=unused-import
from poutyne.framework.callbacks.callbacks import Callback
from poutyne.framework.metrics import get_callables_and_names
from poutyne.framework.metrics.decomposable import convert_decomposable_metric_to_object
from poutyne.framework.strategy.base import BaseStrategy, MetricReturnType, NetworkIOType, StepOutput


class DefaultStrategy(BaseStrategy):
    def set_model(self, model: 'poutyne.Model') -> None:
        super().set_model(model)
        self._setup_loss_function()
        self._setup_metrics()

    def _setup_loss_function(self):
        self.loss_function = self.model.loss_function
        if self.loss_function is not None:
            self.loss_function = convert_decomposable_metric_to_object(self.loss_function, 'loss')

    def _setup_metrics(self):
        batch_metrics, batch_metrics_names = get_callables_and_names(self.model.batch_metrics)
        self.batch_metrics = [
            convert_decomposable_metric_to_object(metric, names)
            for metric, names in zip(batch_metrics, batch_metrics_names)
        ]

        epoch_metrics, epoch_metrics_names = get_callables_and_names(self.model.epoch_metrics)
        self.epoch_metrics = [
            convert_decomposable_metric_to_object(metric, names, is_epoch_metric=True)
            for metric, names in zip(epoch_metrics, epoch_metrics_names)
        ]

        self.batch_metrics_names, self.epoch_metrics_names = batch_metrics_names, epoch_metrics_names

    def infer(self, x: NetworkIOType, **kwargs: Any) -> NetworkIOType:  # pylint: disable=unused-argument
        x = x if isinstance(x, (tuple, list)) else (x,)

        # Support PackedSequence since it is a namedtuple.
        x = (x,) if isinstance(x, PackedSequence) else x

        if self.model.other_device is not None:
            pred_y = torch.nn.parallel.data_parallel(
                self.model.network, x, [self.model.device] + self.model.other_device
            )
        else:
            pred_y = self.model.network(*x)
        return pred_y

    def compute_loss(self) -> Optional[float]:
        return float(self.loss_function.compute())

    def reset_loss(self) -> None:
        self.loss_function.reset()

    def get_batch_metric_names(self) -> List[str | Tuple[str, ...]]:
        return self.batch_metrics_names

    def get_epoch_metric_names(self) -> List[str | Tuple[str, ...]]:
        return self.epoch_metrics_names

    def _compute_metrics(self, metrics) -> List[NetworkIOType]:
        return [metric.compute() for metric in metrics]

    def _reset_metrics(self, metrics) -> None:
        for metric in metrics:
            metric.reset()

    def compute_batch_metrics(self) -> List[MetricReturnType]:
        return self._compute_metrics(self.batch_metrics)

    def compute_epoch_metrics(self) -> List[MetricReturnType]:
        return self._compute_metrics(self.epoch_metrics)

    def reset_batch_metrics(self) -> None:
        self._reset_metrics(self.batch_metrics)

    def reset_epoch_metrics(self) -> None:
        self._reset_metrics(self.epoch_metrics)

    def _compute_loss_and_metrics(
        self, data: Tuple[NetworkIOType, NetworkIOType], *, return_loss_tensor: bool = True
    ) -> StepOutput:
        x, y = data
        y_pred = self.infer(x)
        loss = self.loss_function(y_pred, y)
        if not return_loss_tensor:
            loss = float(loss)

        with torch.no_grad():
            batch_metrics = [metric(y_pred, y) for metric in self.batch_metrics]
            for epoch_metric in self.epoch_metrics:
                epoch_metric.update(y_pred, y)
        return StepOutput(loss=loss, batch_metrics=batch_metrics, y_pred=y_pred, y_true=y)

    def optimizer_zero_grad(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        for opt in self.model.optimizers:
            opt.zero_grad()

    def backward(self, loss: torch.Tensor, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        loss.backward()

    def optimizer_step(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        for opt in self.model.optimizers:
            opt.step()

    def train_step(
        self,
        data: NetworkIOType,
        *,
        callback: Optional[Callback] = None,
        step: Optional[int] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> StepOutput:
        output = self._compute_loss_and_metrics(data, return_loss_tensor=True)
        loss_tensor = output.loss

        self.optimizer_zero_grad()
        self.backward(loss_tensor)
        if callback is not None:
            callback.on_backward_end(step)
        self.optimizer_step()

        loss = float(loss_tensor)
        output = output._replace(loss=loss)

        return output

    def on_epoch_begin(self, epoch_number: int, logs: Dict) -> None:
        pass

    def on_epoch_end(self, epoch_number: int, logs: Dict) -> None:
        pass

    def on_train_begin(self, logs: Dict) -> None:
        pass

    def on_train_end(self, logs: Dict) -> None:
        pass

    def test_step(self, data, **kwargs: Any) -> StepOutput:  # pylint: disable=unused-argument
        return self._compute_loss_and_metrics(data, return_loss_tensor=False)

    def on_valid_begin(self, logs: Dict) -> None:
        pass

    def on_valid_end(self, logs: Dict) -> None:
        pass

    def on_test_begin(self, logs: Dict) -> None:
        pass

    def on_test_end(self, logs: Dict) -> None:
        pass

    def predict_step(self, data, **kwargs: Any) -> NetworkIOType:  # pylint: disable=unused-argument
        return self.infer(data)

    def on_predict_begin(self, logs: Dict) -> None:
        pass

    def on_predict_end(self, logs: Dict) -> None:
        pass
