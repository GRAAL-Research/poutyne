from collections import namedtuple
from typing import Any, List, Optional, Dict, Tuple, Union
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence

import poutyne  # pylint: disable=unused-import
from poutyne.utils import get_batch_size
from poutyne.framework.callbacks.callbacks import Callback

NetworkIOType = Union[Dict[str, 'NetworkIOType'], List['NetworkIOType'], Tuple['NetworkIOType', ...], torch.Tensor]
MetricReturnType = Union[torch.Tensor, float, np.ndarray, Dict[str, Union[torch.Tensor, float, np.ndarray]]]


StepOutput = namedtuple('StepOutput', ['loss', 'metrics', 'y_pred', 'y_true', 'x'])


class Strategy:
    def set_params(self, params: Dict[str, Any]) -> None:
        self.params = params

    def set_model(self, model: 'poutyne.Model') -> None:
        self.model = model

    def step_output(
        self,
        loss: Union[float, torch.Tensor],
        *,
        metrics: Optional[List[MetricReturnType]] = None,
        y_pred: Optional[NetworkIOType] = None,
        y_true: Optional[NetworkIOType] = None,
        x: Optional[NetworkIOType] = None,
    ) -> StepOutput:
        return StepOutput(loss, metrics, y_pred, y_true, x)

    def infer(self, x: NetworkIOType, **kwargs: Any) -> NetworkIOType:  # pylint: disable=unused-argument
        if self.model.other_device is not None:
            pred_y = torch.nn.parallel.data_parallel(
                self.model.network, x, [self.model.device] + self.model.other_device
            )
        else:
            pred_y = self.model.network(*x)
        return pred_y

    def _pack_input(self, x: NetworkIOType, y: Optional[NetworkIOType] = None) -> NetworkIOType:
        x = x if isinstance(x, (tuple, list)) else (x,)

        # We return PackedSequence in a tuple since it is a namedtuple, thus an iterator object and
        # would break later when we call self.network(*x) since it will iterate over the PackedSequence named attribute.
        x = (x,) if isinstance(x, PackedSequence) else x

        return (x, y) if y is not None else x

    def _compute_loss_and_metrics(
        self, data: Tuple[NetworkIOType, NetworkIOType], *, return_loss_tensor: bool = True
    ) -> StepOutput:
        x, y = data
        x, y = self._pack_input(x, y)
        y_pred = self.infer(x)
        loss = self.model.loss_function(y_pred, y)
        if not return_loss_tensor:
            loss = float(loss)

        with torch.no_grad():
            batch_metrics = [metric(y_pred, y) for metric in self.model.batch_metrics]
            for epoch_metric in self.model.epoch_metrics:
                epoch_metric.update(y_pred, y)
        return self.step_output(loss, metrics=batch_metrics, y_pred=y_pred, y_true=y)

    def optimizer_zero_grad(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        for opt in self.model.optimizer:
            opt.zero_grad()

    def backward(self, loss: torch.Tensor, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        loss.backward()

    def optimizer_step(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        for opt in self.model.optimizer:
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
        x = self._pack_input(data)
        return self.infer(x)

    def on_predict_begin(self, logs: Dict) -> None:
        pass

    def on_predict_end(self, logs: Dict) -> None:
        pass


class GradientAccumulationStrategy(Strategy):
    def __init__(self, batches_per_step: int) -> None:
        super().__init__()
        if batches_per_step <= 0:
            raise ValueError("`batches_per_step` must be greater than 0.")
        self.batches_per_step = batches_per_step

    def on_epoch_begin(self, epoch_number: int, logs: Dict) -> None:
        self.zero_all_gradients = True
        self.do_optimizer_step = True
        self.examples_in_step = 0
        self.current_step_size = 0

    def on_epoch_end(self, epoch_number: int, logs: Dict) -> None:
        if not self.do_optimizer_step:
            self._adjust_step_size()
            super().optimizer_step()

    def train_step(
        self, data: NetworkIOType, *, callback: Optional[Callback] = None, step: Optional[int] = None, **kwargs
    ) -> StepOutput:
        if step is not None:
            self.zero_all_gradients = (step - 1) % self.batches_per_step == 0
            self.do_optimizer_step = step % self.batches_per_step == 0
            self.current_step_size = get_batch_size(*data)
            self.examples_in_step += self.current_step_size

        output = super().train_step(data, callback=callback, step=step, **kwargs)

        return output

    def optimizer_zero_grad(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        if self.zero_all_gradients:
            super().optimizer_zero_grad()

    def backward(self, loss: torch.Tensor, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        loss = loss * self.current_step_size
        loss.backward()

    def optimizer_step(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        if self.do_optimizer_step:
            self._adjust_step_size()
            super().optimizer_step()

    def _adjust_step_size(self) -> None:
        for param in self.model.network.parameters():
            if param.grad is not None:
                param.grad /= max(self.examples_in_step, 1)
