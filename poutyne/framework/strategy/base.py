# pylint: disable=too-many-public-methods
from typing import Any, List, Optional, Dict, Tuple, Union, NamedTuple
import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence

import poutyne  # pylint: disable=unused-import
from poutyne.utils import get_batch_size
from poutyne.framework.callbacks.callbacks import Callback
from poutyne.framework.metrics import get_callables_and_names
from poutyne.framework.metrics.decomposable import convert_decomposable_metric_to_object

NetworkIOType = Union[Dict[str, 'NetworkIOType'], List['NetworkIOType'], Tuple['NetworkIOType', ...], torch.Tensor]
MetricReturnType = Union[torch.Tensor, float, np.ndarray, Dict[str, Union[torch.Tensor, float, np.ndarray]]]


class StepOutput(NamedTuple):
    loss: Optional[Union[torch.Tensor, float]] = None
    batch_metrics: Optional[Union[List[NetworkIOType], np.ndarray]] = None
    y_pred: Optional[NetworkIOType] = None
    y_true: Optional[NetworkIOType] = None
    x: Optional[NetworkIOType] = None


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
        self, data: NetworkIOType, *, callback: Optional[Callback] = None, step: Optional[int] = None, **kwargs
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

    def compute_loss(self) -> float:
        return float(self.loss_function.compute())

    def reset_loss(self) -> None:
        self.loss_function.reset()

    def get_batch_metric_names(self) -> List[str]:
        return self.batch_metrics_names

    def get_epoch_metric_names(self) -> List[str]:
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


class GradientAccumulationStrategy(DefaultStrategy):
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
