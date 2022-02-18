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
from typing import Dict

from .callbacks import Callback


class BestModelRestore(Callback):
    """
    Restore the weights of the best model at the end of the training depending on a monitored quantity.

    Args:
        monitor (str): Quantity to monitor. (Default value = 'val_loss')
        mode (str): One of {'min', 'max'}.
            Whether the monitored has to be maximized or minimized. For instance, for `val_accuracy`,
            this should be `max`, and for `val_loss`, this should be `min`, etc.
            (Default value = 'min')
        verbose (bool): Whether to display a message when the model has improved or when restoring
            the best model.
            (Default value = False)
    """

    def __init__(self, *, monitor: str = 'val_loss', mode: str = 'min', verbose: bool = False):
        super().__init__()
        self.monitor = monitor

        if mode not in ['min', 'max']:
            raise ValueError(f"Invalid mode '{mode}'")
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y
            self.current_best = float('Inf')
        elif mode == 'max':
            self.monitor_op = lambda x, y: x > y
            self.current_best = -float('Inf')
        self.best_weights = None
        self.verbose = verbose

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        if self.monitor_op(logs[self.monitor], self.current_best):
            old_best = self.current_best
            self.current_best = logs[self.monitor]

            if self.verbose:
                print(
                    f'Epoch {epoch_number:d}: {self.monitor} improved from {old_best:0.5f} to {self.current_best:0.5f}'
                )
            self.best_weights = self.model.get_weight_copies()

    def on_train_end(self, logs: Dict):
        if self.best_weights is not None:
            if self.verbose:
                print('Restoring best model')
            self.model.set_weights(self.best_weights)
        else:
            warnings.warn('No weights to restore!')
