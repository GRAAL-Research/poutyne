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

import inspect
import sys
from typing import Dict, BinaryIO

import torch.optim.lr_scheduler
from torch.optim import Optimizer

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from .callbacks import Callback


class _PyTorchLRSchedulerWrapper(Callback):
    """
    Default class for the LR scheduling callback. Proposes default comportment for the scheduler
    loading and saving as well as for the epoch end handling.
    """

    def __init__(self, torch_lr_scheduler, *args, **kwargs):
        super().__init__()
        if len(args) > 0 and isinstance(args[0], Optimizer):
            raise ValueError(
                "In the LR scheduler callbacks, the optimizer is "
                "automatically passed to the PyTorch's LR scheduler. "
                "You must remove it from the arguments."
            )
        self.args = args
        self.kwargs = kwargs
        self.scheduler = None
        self.state_to_load = None
        self.torch_lr_scheduler = torch_lr_scheduler

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        self.scheduler.step()

    def on_train_begin(self, logs: Dict):
        self.scheduler = self.torch_lr_scheduler(self.model.optimizer, *self.args, **self.kwargs)

        # Load state if the scheduler was not initialized when the user asked
        # to load its state
        if self.state_to_load is not None:
            self.load_state_dict(self.state_to_load)
            self.state_to_load = None

    def load_state_dict(self, state_dict):
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict)
        else:
            self.state_to_load = state_dict

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state(self, f: BinaryIO):
        self.load_state_dict(torch.load(f, map_location='cpu'))

    def save_state(self, f: BinaryIO):
        torch.save(self.state_dict(), f)


def new_init(torch_lr_scheduler):
    def f(self, *args, **kwargs):
        super(type(self), self).__init__(torch_lr_scheduler, *args, **kwargs)

    return f


for name, module_cls in torch.optim.lr_scheduler.__dict__.items():
    if inspect.isclass(module_cls) and issubclass(module_cls, LRScheduler) and module_cls != LRScheduler:
        _new_cls = type(
            name,
            (_PyTorchLRSchedulerWrapper,),
            {
                '__init__': new_init(module_cls),
                '__doc__': f"""
                            See:
                                :class:`~torch.optim.lr_scheduler.{name}`
                            """,
            },
        )
        setattr(sys.modules[__name__], name, _new_cls)


class ReduceLROnPlateau(_PyTorchLRSchedulerWrapper):
    """
    Args:
        monitor (str): The quantity to monitor. (Default value = 'val_loss')
    See:
        :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`
    """

    def __init__(self, *args, monitor: str = 'val_loss', **kwargs):
        super().__init__(torch_lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, *args, **kwargs)
        self.monitor = monitor

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        self.scheduler.step(logs[self.monitor])
