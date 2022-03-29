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

from .utils import get_names_of_metric
from .metrics_registering import get_metric
from .base import Metric
from .decomposable import convert_decomposable_metric_to_object


class IndexedArgumentMetric(Metric):
    def __init__(self, metric, *, index=None, pred_index=None, true_index=None):
        super().__init__()

        if isinstance(metric, str):
            metric = get_metric(metric)
        names, metric = get_names_of_metric(metric)
        metric = convert_decomposable_metric_to_object(metric, names)

        self.__name__ = names
        self.metric = metric

        if index is not None and (pred_index is not None or true_index is not None):
            raise ValueError(
                "'pred_index' and 'true_index' arguments should should not be used with the 'index argument."
            )
        if index is not None:
            self.pred_index = index
            self.true_index = index
        else:
            self.pred_index = pred_index
            self.true_index = true_index

    def _select_indices(self, y_pred, y_true):
        if self.pred_index is not None:
            y_pred = y_pred[self.pred_index]
        if self.true_index is not None:
            y_true = y_true[self.true_index]
        return y_pred, y_true

    def forward(self, y_pred, y_true):
        y_pred, y_true = self._select_indices(y_pred, y_true)
        return self.metric(y_pred, y_true)

    def update(self, y_pred, y_true):
        y_pred, y_true = self._select_indices(y_pred, y_true)
        return self.metric.update(y_pred, y_true)

    def compute(self):
        return self.metric.compute()

    def reset(self) -> None:
        return self.metric.reset()
