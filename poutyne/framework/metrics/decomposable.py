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

import numbers
from typing import Mapping, Iterable
import warnings
import numpy as np
import torch

from .base import Metric
from ..warning_manager import warning_settings


class DecomposableMetric(Metric):
    def __init__(self, func, names):
        super().__init__()
        self.func = func
        self.names = [names] if isinstance(names, str) else names
        self.__name__ = self.names
        self.reset()

    def update(self, y_pred, y_true):
        output = self.func(y_pred, y_true)

        np_output = self._output_to_array(output)
        batch_size = self._get_batch_size(y_pred, y_true)
        self.output_sums += np_output * batch_size
        self.size += batch_size

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

    def _get_batch_size(self, y_pred, y_true):
        """
        This method infers the batch size of a batch. Here is the inferring algorithm used to compute the
        batch size. ``y_true`` and ``y_pred`` are tested in this order at each step of the inferring algorithm. If one
        step succeed for one of ``y_true`` or ``y_pred``, the algorithm stops.

        - Step 1: if ``y_true`` or ``y_pred`` is a tensor or a Numpy array, then the ``len()`` is returned.
        - Step 2: if ``y_true`` or ``y_pred`` is a list or a tuple, then the ``len()`` of the first element is returned
          if it is a tensor or a Numpy array.
        - Step 3: if ``y_true`` or ``y_pred`` is a dict, then the value for the key ``'batch_size'`` is returned if it
          is of integral type.
        - Step 4: if ``y_true`` or ``y_pred`` is a dict, then the ``len()`` of the first element of ``.values()`` is
          returned if it is a tensor or a Numpy array.

        If inferring the batch size is not possible, the batch size is set to 1 and, thus, the computed
        loss and metrics at the end of each epoch is the mean of the batches' losses and metrics. In which
        case, a warning is also raised. To disable this warning, set

        .. code-block:: python

            from poutyne import warning_settings\n
            warning_settings['batch_size'] = 'ignore'\n\n

        Args:
            y_pred: Predictions for a batch.
            y_true: Target data of a batch.
        """

        for v in [y_true, y_pred]:
            if torch.is_tensor(v):
                return len(v)
        for v in [y_true, y_pred]:
            if isinstance(v, (tuple, list)):
                if torch.is_tensor(v[0]):
                    return len(v[0])
        for v in [y_true, y_pred]:
            if isinstance(v, dict):
                if 'batch_size' in v and isinstance(v['batch_size'], numbers.Integral):
                    return v['batch_size']
        for v in [y_true, y_pred]:
            if isinstance(v, dict):
                first_value = list(v.values())[0]
                if torch.is_tensor(first_value):
                    return len(first_value)

        if warning_settings['batch_size'] == 'warn':
            warnings.warn(
                "Inferring the batch size is not possible. Hence, "
                "the batch size is set to 1 and, thus, the computed "
                "loss and metrics at the end of each epoch is the "
                "mean of the batches' losses and metrics. To disable "
                "this warning, set\n"
                "from poutyne import warning_settings\n"
                "warning_settings['batch_size'] = 'ignore'\n\n"
                #
                #
                "Here is the inferring algorithm used to compute the "
                "batch size. 'y_true' and 'y_pred' are tested in this order at "
                "each step of the inferring algorithm. If one step "
                "succeed for one of 'y_true' or 'y_pred', the algorithm stops.\n\n"
                #
                #
                "Step 1: if 'y_true' or 'y_pred' is a tensor or a Numpy array, "
                "then the 'len()' is returned.\n"
                #
                "Step 2: if 'y_true' or 'y_pred' is a list or a tuple, then the "
                "'len()' of the first element is returned if it is a "
                "tensor or a Numpy array.\n"
                #
                "Step 3: if 'y_true' or 'y_pred' is a dict, then the value for "
                "the key 'batch_size' is returned if it is of integral "
                "type.\n"
                #
                "Step 4: if 'y_true' or 'y_pred' is a dict, then the 'len()' of "
                "the first element of '.values()' is returned if it is a "
                "tensor or a Numpy array.\n"
            )
        return 1
