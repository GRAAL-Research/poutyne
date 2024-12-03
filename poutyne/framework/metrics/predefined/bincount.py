"""
The source code of this file was copied from the torchmetrics project, and has been modified. All modifications
made from the original source code are under the LGPLv3 license.


Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information on the Poutyne and AllenNLP repository.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.


Copyright The PyTorch Lightning team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional

import torch
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor

_TORCH_GREATER_EQUAL_1_12 = RequirementCache("torch>=1.12.0")

try:
    from torchmetrics.utilities.imports import _XLA_AVAILABLE
except ImportError:
    _XLA_AVAILABLE = False


def _bincount(x: Tensor, minlength: Optional[int] = None) -> Tensor:
    """PyTorch currently does not support``torch.bincount`` for:

        - deterministic mode on GPU.
        - MPS devices

    This implementation fallback to a for-loop counting occurrences in that case.

    Args:
        x: tensor to count
        minlength: minimum length to count

    Returns:
        Number of occurrences for each unique element in x
    """
    if minlength is None:
        minlength = len(torch.unique(x))
    if torch.are_deterministic_algorithms_enabled() or _XLA_AVAILABLE or _TORCH_GREATER_EQUAL_1_12 and x.is_mps:
        output = torch.zeros(minlength, device=x.device, dtype=torch.long)
        for i in range(minlength):
            output[i] = (x == i).sum()
        return output
    return torch.bincount(x, minlength=minlength)
