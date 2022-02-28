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

from typing import Mapping, Iterable
import numpy as np
import torch
import torch.nn as nn

from .base import Metric
from ...utils import get_batch_size


class DecomposableMetric(Metric):
    def __init__(self, func, names):
        super().__init__()
        self.func = func
        self.names = [names] if isinstance(names, str) else names
        self.__name__ = self.names
        self.reset()

    def forward(self, y_pred, y_true):
        return self._update(y_pred, y_true)

    def update(self, y_pred, y_true):
        self._update(y_pred, y_true)

    def _update(self, y_pred, y_true):
        output = self.func(y_pred, y_true)

        # In order to aggregate the batches, it is simpler to put into a NumPy array all relevant values.
        np_output = self._output_to_array(output)
        batch_size = get_batch_size(y_pred, y_true)
        self.output_sums += np_output * batch_size
        self.size += batch_size

        # We return `output` on purpose in order to keep the tensors as is so that a DecomposableMetric object can
        # be used as a loss function.
        return output

    def _output_to_array(self, output):
        if (torch.is_tensor(output) or isinstance(output, np.ndarray)) and len(output.shape) == 0:
            values = [float(output)]
        elif isinstance(output, Mapping):
            values = [float(output[name]) for name in self.names]
        elif isinstance(output, Iterable):
            values = [float(metric) for metric in output]
        else:
            values = [float(output)]
        return np.array(values)

    def compute(self):
        return self.output_sums / self.size

    def reset(self) -> None:
        self.output_sums = np.zeros(len(self.names))
        self.size = 0


def convert_decomposable_metric_to_object(metric, names, is_epoch_metric=False):
    if (
        isinstance(metric, nn.Module)
        and hasattr(metric, 'compute')
        and (not is_epoch_metric or hasattr(metric, 'update'))
        and hasattr(metric, 'reset')
    ):
        return metric
    return DecomposableMetric(metric, names)
