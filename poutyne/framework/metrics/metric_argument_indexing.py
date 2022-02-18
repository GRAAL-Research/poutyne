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

# Because nn.Module has the abstract method _forward_unimplemented
# pylint: disable=abstract-method
import torch.nn as nn

from .metrics_registering import get_loss_or_metric, get_epoch_metric
from .utils import get_names_of_metric
from .epoch_metrics.base import EpochMetric


class IndexedArgumentMetric(nn.Module):
    def __init__(self, metric, *, index=None, pred_index=None, true_index=None):
        super().__init__()
        names, metric = get_names_of_metric(metric)
        self.__name__ = names
        self.metric = metric

        if index is not None and (pred_index is not None or true_index is not None):
            raise ValueError(
                "'pred_index' and 'true_index' arguments should should not be used " "with the 'index argument."
            )
        if index is not None:
            self.pred_index = index
            self.true_index = index
        else:
            self.pred_index = pred_index
            self.true_index = true_index

    def forward(self, y_pred, y_true):
        if self.pred_index is not None:
            y_pred = y_pred[self.pred_index]
        if self.true_index is not None:
            y_true = y_true[self.true_index]
        return self.metric(y_pred, y_true)


class IndexedArgumentBatchMetric(IndexedArgumentMetric):
    def __init__(self, metric, **kwargs):
        if isinstance(metric, str):
            metric = get_loss_or_metric(metric)
        super().__init__(metric, **kwargs)


class IndexedArgumentEpochMetric(IndexedArgumentMetric, EpochMetric):
    def __init__(self, metric, **kwargs):
        if isinstance(metric, str):
            metric = get_epoch_metric(metric)
        super().__init__(metric, **kwargs)

    def get_metric(self):
        return self.metric.get_metric()

    def reset(self) -> None:
        return self.metric.reset()
