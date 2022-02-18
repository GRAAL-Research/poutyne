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

from typing import Iterable, Union

import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from .callbacks import Callback

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


class ClipNorm(Callback):
    """
    Uses PyTorch's :func:`~torch.nn.utils.clip_grad_norm_()`
    method to clip gradient.

    See:
        :func:`torch.nn.utils.clip_grad_norm_()`

    """

    def __init__(self, parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0):
        super().__init__()
        self.parameters = list(parameters) if not torch.is_tensor(parameters) else [parameters]
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_backward_end(self, batch_number: int):
        clip_grad_norm_(self.parameters, self.max_norm, norm_type=self.norm_type)


class ClipValue(Callback):
    """
    Uses PyTorch's :func:`~torch.nn.utils.clip_grad_value_()`
    method to clip gradient.

    See:
        :func:`torch.nn.utils.clip_grad_value_()`

    """

    def __init__(self, parameters: _tensor_or_tensors, clip_value: float):
        super().__init__()
        self.parameters = list(parameters) if not torch.is_tensor(parameters) else [parameters]
        self.clip_value = clip_value

    def on_backward_end(self, batch_number: int):
        clip_grad_value_(self.parameters, self.clip_value)
