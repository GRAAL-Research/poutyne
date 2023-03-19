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

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from poutyne.framework.callbacks.callbacks import Callback
from poutyne.framework.strategy.base import NetworkIOType, StepOutput
from poutyne.framework.strategy.default import DefaultStrategy
from poutyne.utils import get_batch_size


class GradientAccumulationStrategy(DefaultStrategy):
    def __init__(self, batches_per_step: int) -> None:
        super().__init__()
        if batches_per_step <= 0:
            raise ValueError("`batches_per_step` must be greater than 0.")
        self.batches_per_step = batches_per_step

    def on_epoch_begin(self, epoch_number: int, logs: Dict) -> None:
        self.zero_all_gradients = True
        self.do_optimizer_step = True
        self.examples_in_step = 0
        self.current_step_size = 0

    def on_epoch_end(self, epoch_number: int, logs: Dict) -> None:
        if not self.do_optimizer_step:
            self._adjust_step_size()
            super().optimizer_step()

    def train_step(
        self, data: NetworkIOType, *, callback: Optional[Callback] = None, step: Optional[int] = None, **kwargs
    ) -> StepOutput:
        if step is not None:
            self.zero_all_gradients = (step - 1) % self.batches_per_step == 0
            self.do_optimizer_step = step % self.batches_per_step == 0
            self.current_step_size = get_batch_size(*data)
            self.examples_in_step += self.current_step_size

        output = super().train_step(data, callback=callback, step=step, **kwargs)

        return output

    def optimizer_zero_grad(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        if self.zero_all_gradients:
            super().optimizer_zero_grad()

    def backward(self, loss: torch.Tensor, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        loss = loss * self.current_step_size
        loss.backward()

    def optimizer_step(self, **kwargs: Any) -> None:  # pylint: disable=unused-argument
        if self.do_optimizer_step:
            self._adjust_step_size()
            super().optimizer_step()

    def _adjust_step_size(self) -> None:
        for param in self.model.network.parameters():
            if param.grad is not None:
                param.grad /= max(self.examples_in_step, 1)
