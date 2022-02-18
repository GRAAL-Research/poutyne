"""
The source code of this file was copied from the Keras project, and has been modified. All modifications
made from the original source code are under the LGPLv3 license.

COPYRIGHT

All contributions by François Chollet:
Copyright (c) 2015, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2017, the respective contributors.
All rights reserved.

Copyright (c) 2022 Poutyne.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information on the Poutyne and Keras repository.

LICENSE

The LGPLv3 License

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.


The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from typing import Dict

import numpy as np

from .callbacks import Callback


class EarlyStopping(Callback):
    """
    The source code of this class is under the MIT License and was copied from the Keras project,
    and has been modified.

    Stop training when a monitored quantity has stopped improving.

    Args:
        monitor (str): Quantity to be monitored.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement.
            (Default value = 0)
        patience (int): Number of epochs with no improvement after which training will be stopped.
            (Default value = 0)
        verbose (bool): Whether to print when early stopping is done.
            (Default value = False)
        mode (str): One of {'min', 'max'}. In `min` mode, training will stop when the quantity
            monitored has stopped decreasing; in `max` mode it will stop when the quantity monitored has
            stopped increasing.
            (Default value = 'min')
    """

    def __init__(
        self,
        *,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 0,
        verbose: bool = False,
        mode: str = 'min',
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['min', 'max']:
            raise ValueError(f"Invalid mode '{mode}'")
        self.mode = mode

        if mode == 'min':
            self.min_delta *= -1
            self.monitor_op = np.less
        elif mode == 'max':
            self.min_delta *= 1
            self.monitor_op = np.greater

    def on_train_begin(self, logs: Dict):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.mode == 'min' else -np.Inf

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        current = logs[self.monitor]
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch_number
                self.model.stop_training = True
                if self.verbose:
                    print(f'Epoch {self.stopped_epoch}: early stopping')
