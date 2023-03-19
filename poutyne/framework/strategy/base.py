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
from typing import Any, Dict, List, NamedTuple, Optional, Tuple  # pylint: disable=unused-import   # noqa: F401

import numpy as np
import torch
from typing_extensions import TypeAlias

# Imported for typing with PyLance.
import poutyne  # pylint: disable=unused-import
from poutyne.framework.callbacks.callbacks import Callback

NetworkIOType: TypeAlias = "Dict[str, NetworkIOType] | List[NetworkIOType] | Tuple[NetworkIOType, ...] | torch.Tensor"

MetricValue: TypeAlias = "torch.Tensor | float | np.ndarray"
MetricReturnType: TypeAlias = "MetricValue | Dict[str, MetricValue] | List[MetricValue]"


class StepOutput(NamedTuple):
    """
    Dataclass used to return information in a training or a testing step.
    """

    loss: Optional[torch.Tensor | float] = None
    """
    The loss for the step.
    """

    batch_metrics: Optional[List[MetricReturnType] | np.ndarray] = None
    """
    List of batch metric values for the step. :ref:`callbacks <callbacks>`
    """

    y_pred: Optional[NetworkIOType] = None
    y_true: Optional[NetworkIOType] = None
    x: Optional[NetworkIOType] = None


class BaseStrategy:
    """
    This class defines the interface needed in order to intervene at every step of training, evaluating and doing
    predictions using Poutyne. For implementing complex training strategies, this is the interface that one should
    implement. For simpler training strategies, one should instead look into inheriting the
    :class:`~poutyne.DefaultStrategy` class.

    When **training**, here is the order in which the methods of the strategy are called:

    .. code-block:: python

        strategy.set_model(model)
        strategy.get_batch_metric_names()
        strategy.get_epoch_metric_names()
        strategy.set_params(params)
        strategy.on_train_begin({})
        for epoch in range(1, epochs + 1):
            strategy.on_epoch_begin(epoch, {})

            # Training loop
            for step in range(1, steps + 1):
                strategy.train_step((x, y), callback=callback, step=step)

            strategy.compute_loss()
            strategy.reset_loss()
            strategy.compute_batch_metrics()
            strategy.reset_batch_metrics()
            strategy.compute_epoch_metrics()
            strategy.reset_epoch_metrics()

            # Validation loop
            if has_valid:
                strategy.on_valid_begin({})
                for step in range(valid_steps):
                    strategy.test_step((x, y))
                strategy.compute_loss()
                strategy.reset_loss()
                strategy.compute_batch_metrics()
                strategy.reset_batch_metrics()
                strategy.compute_epoch_metrics()
                strategy.reset_epoch_metrics()
                strategy.on_valid_end({...})

            strategy.on_epoch_end(epoch, {...})

        strategy.on_train_end({})

    When **evaluating**, here is the order in which the methods of the strategy are called:

    .. code-block:: python

        strategy.set_model(self.model)
        strategy.get_batch_metric_names()
        strategy.get_epoch_metric_names()
        strategy.set_params(params)
        strategy.on_test_begin({})
        for _ in range(steps):
            strategy.test_step((x, y))
        strategy.compute_loss()
        strategy.reset_loss()
        strategy.compute_batch_metrics()
        strategy.reset_batch_metrics()
        strategy.compute_epoch_metrics()
        strategy.reset_epoch_metrics()
        strategy.on_test_end(logs)

    Attributes:
        params (Dict[str, Any]): Contains ``'epoch'`` and ``'steps_per_epoch'`` keys which are passed to the
            when training. Contains ``'steps'`` when evaluating. May contain other keys.
        model (poutyne.Model): A reference to the :class:`~poutyne.Model` object using the strategy.
    """

    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Called with a dictionary containing ``'epoch'`` and ``'steps_per_epoch'`` as keys. The dictionnary contains
        ``'steps'`` when evaluating. It may also contain other keys.

        Args:
            params (Dict[str, Any]): A dictionary containing ``'epoch'`` and ``'steps_per_epoch'`` as keys. The
                dictionnary contains ``'steps'`` when evaluating. It may also contain other keys.
        """
        self.params = params

    def set_model(self, model: 'poutyne.Model') -> None:
        """
        Called first with the :class:`~poutyne.Model` object using the strategy.

        Args:
            model (poutyne.Model): A reference to the :class:`~poutyne.Model` object using the strategy.
        """
        self.model = model

    def compute_loss(self) -> Optional[float]:
        """
        Return the cumulated loss over a number of training/test steps.

        Returns:
            Optional[float]: The cumulated loss. Return None if not applicable.
        """
        pass

    def reset_loss(self) -> None:
        """
        Reset the accumulation of the loss over a number of training/test steps.
        """
        pass

    def get_batch_metric_names(self) -> List[str | Tuple[str, ...]]:
        """
        Return the names of each batch metrics in order they are provided in
        :class:`~poutyne.StepOutput` and returned by
        :meth:`~BaseStrategy.compute_batch_metrics()`. A single batch metric can have
        multiple names if it returns a list or NumPy array with the

        Returns:
            List[str | Tuple[str, ...]]: _description_
        """
        return []

    def get_epoch_metric_names(self) -> List[str | Tuple[str, ...]]:
        return []

    def compute_batch_metrics(self) -> List[MetricReturnType]:
        """_summary_

        Returns:
            List[MetricReturnType]: _description_
        """
        return []

    def compute_epoch_metrics(self) -> List[MetricReturnType]:
        return []

    def reset_batch_metrics(self) -> None:
        pass

    def reset_epoch_metrics(self) -> None:
        pass

    def train_step(
        self, data: NetworkIOType, *, callback: Optional[Callback] = None, step: Optional[int] = None, **kwargs: Any
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
