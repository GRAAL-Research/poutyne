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

import torch.nn as nn


class Lambda(nn.Module):
    """
    Applies a function to the input tensor.

    Args:
        func (Callable[[~torch.Tensor], ~torch.Tensor]): The function to apply.

    Example:

        .. code-block:: python

            # Alternate version to the ``nn.Flatten`` module.
            my_flatten = Lambda(lambda x: x.flatten(1))

    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *x):
        return self.func(*x)
