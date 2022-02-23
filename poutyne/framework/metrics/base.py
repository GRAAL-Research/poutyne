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

from abc import ABC, abstractmethod

import torch.nn as nn


class Metric(ABC, nn.Module):
    """
    The abstract class representing a metric which can be accumulated at each batch and calculated at the end
    of the epoch.
    """

    def forward(self, y_pred, y_true):
        return self.update(y_pred, y_true)

    def update(self, y_pred, y_true) -> None:
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        The information kept for the computation of the metric is cleaned so
        that a new epoch can be done.
        """
        pass
