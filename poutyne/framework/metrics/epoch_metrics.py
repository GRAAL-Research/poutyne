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

import warnings
from abc import ABC, abstractmethod

import torch.nn as nn


class EpochMetric(ABC, nn.Module):
    """
    The abstract class representing a epoch metric which can be accumulated at each batch and calculated at the end
    of the epoch.
    """

    def __init__(self) -> None:
        super().__init__()
        warnings.warn(
            "Using the EpochMetric class is deprecated. Use the Metric class instead and follow its interface."
        )

    @abstractmethod
    def forward(self, y_pred, y_true) -> None:
        """
        To define the behavior of the metric when called.

        Args:
            y_pred: The prediction of the model.
            y_true: Target to evaluate the model.
        """
        pass

    @abstractmethod
    def get_metric(self):
        """
        Compute and return the metric. Should not modify the state of
        the epoch metric.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        The information kept for the computation of the metric is cleaned so
        that a new epoch can be done.
        """
        pass

    # For compatibility with torchmetrics.
    def update(self, y_pred, y_true) -> None:
        return self.forward(y_pred, y_true)

    # For compatibility with torchmetrics.
    def compute(self):
        return self.get_metric()
