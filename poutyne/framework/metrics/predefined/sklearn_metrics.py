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

from typing import Optional, Union, List, Callable, Dict, Tuple
import numpy as np
import torch
from ..base import Metric


class SKLearnMetrics(Metric):
    """
    Wrap metrics with Scikit-learn-like interface
    (``metric(y_true, y_pred, sample_weight=sample_weight, **kwargs)``).
    The ``SKLearnMetrics`` object has to keep in memory the ground truths and
    predictions so that in can compute the metric at the end.

    Example:
        .. code-block:: python

            from sklearn.metrics import roc_auc_score, average_precision_score
            from poutyne import SKLearnMetrics
            my_epoch_metric = SKLearnMetrics([roc_auc_score, average_precision_score])

    Args:
        funcs (Union[Callable, List[Callable]]): A metric or a list of metrics with a
            scikit-learn-like interface.
        kwargs (Optional[Union[dict, List[dict]]]): Optional dictionary of list of dictionaries
            corresponding to keyword arguments to pass to each corresponding metric.
            (Default value = None)
        names (Optional[Union[str, List[str]]]): Optional string or list of strings corresponding to
            the names given to the metrics. By default, the names are the names of the functions.
    """

    def __init__(
        self,
        funcs: Union[Callable, List[Callable]],
        kwargs: Optional[Union[dict, List[dict]]] = None,
        names: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__()

        self.funcs = funcs if isinstance(funcs, (list, tuple)) else [funcs]
        self.kwargs = self._validate_kwargs(kwargs)
        self.__name__ = self._validate_names(names)

        self.reset()

    def _validate_kwargs(self, kwargs):
        if kwargs is not None:
            kwargs = kwargs if isinstance(kwargs, (list, tuple)) else [kwargs]
            if kwargs is not None and len(self.funcs) != len(kwargs):
                raise ValueError("`kwargs` has to have the same length as `funcs` when provided")
        else:
            kwargs = [{}] * len(self.funcs) if kwargs is None else kwargs
        return kwargs

    def _validate_names(self, names):
        if names is not None:
            names = names if isinstance(names, (list, tuple)) else [names]
            if len(self.funcs) != len(names):
                raise ValueError("`names` has to have the same length as `funcs` when provided")
        else:
            names = [func.__name__ for func in self.funcs]
        return names

    def forward(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        Accumulate the predictions, ground truths and sample weights if any, and compute the metric for the current
        batch.

        Args:
            y_pred (torch.Tensor): A tensor of predictions of the shape expected by
                the metric functions passed to the class.
            y_true (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                Ground truths. A tensor of ground truths of the shape expected by
                the metric functions passed to the class.
                It can also be a tuple with two tensors, the first being the
                ground truths and the second corresponding the ``sample_weight``
                argument passed to the metric functions in Scikit-Learn.
        """
        y_pred, y_true, sample_weight = self._update(y_pred, y_true)
        return self._compute(y_true, y_pred, sample_weight)

    def update(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        Accumulate the predictions, ground truths and sample weights if any.

        Args:
            y_pred (torch.Tensor): A tensor of predictions of the shape expected by
                the metric functions passed to the class.
            y_true (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
                Ground truths. A tensor of ground truths of the shape expected by
                the metric functions passed to the class.
                It can also be a tuple with two tensors, the first being the
                ground truths and the second corresponding the ``sample_weight``
                argument passed to the metric functions in Scikit-Learn.
        """
        self._update(y_pred, y_true)

    def _update(self, y_pred: torch.Tensor, y_true: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        y_pred = y_pred.cpu().numpy()
        self.y_pred_list.append(y_pred)

        sample_weight = None
        if isinstance(y_true, (tuple, list)):
            y_true, sample_weight = y_true

            sample_weight = sample_weight.cpu().numpy()
            self.sample_weight_list.append(sample_weight)

        y_true = y_true.cpu().numpy()
        self.y_true_list.append(y_true)

        return y_pred, y_true, sample_weight

    def compute(self) -> Dict:
        """
        Returns the metrics as a dictionary with the names as keys.
        """
        sample_weight = None
        if len(self.sample_weight_list) != 0:
            sample_weight = np.concatenate(self.sample_weight_list)
        y_pred = np.concatenate(self.y_pred_list)
        y_true = np.concatenate(self.y_true_list)
        return self._compute(y_true, y_pred, sample_weight)

    def _compute(self, y_true, y_pred, sample_weight):
        return {
            name: func(y_true, y_pred, sample_weight=sample_weight, **kwargs)
            for name, func, kwargs in zip(self.__name__, self.funcs, self.kwargs)
        }

    def reset(self) -> None:
        self.y_true_list = []
        self.y_pred_list = []
        self.sample_weight_list = []
