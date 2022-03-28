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

from typing import Dict, Sequence, Optional, Union

from .callbacks import Callback, CallbackList


class DelayCallback(Callback):
    """
    Delays one or many callbacks for a certain number of epochs or number of batches. If both
    ``epoch_delay`` and ``batch_delay`` are provided, the biggest has precedence.

    Args:
        callbacks (Callback, Sequence[Callback]): A callback or a sequence of callbacks to delay.
        epoch_delay (int, optional): Number of epochs to delay.
        batch_delay (int, optional): Number of batches to delay. The number of batches can span many
            epochs. When the batch delay expires (i.e. there are more than `batch_delay` done), the
            :func:`~poutyne.Callback.on_epoch_begin()` method is called on
            the callback(s) before the :func:`~poutyne.Callback.on_train_batch_begin()` method.
    """

    def __init__(
        self,
        callbacks: Union[Callback, Sequence],
        *,
        epoch_delay: Optional[int] = None,
        batch_delay: Optional[int] = None
    ):
        super().__init__()
        if isinstance(callbacks, Sequence):
            self.callbacks = CallbackList(callbacks)
        else:
            self.callbacks = CallbackList([callbacks])

        self.epoch_delay = epoch_delay if epoch_delay else 0
        self.batch_delay = batch_delay if batch_delay else 0

    def set_params(self, params: Dict):
        self.callbacks.set_params(params)

    def set_model(self, model):
        self.callbacks.set_model(model)

    def on_epoch_begin(self, epoch_number: int, logs: Dict):
        self.current_epoch = epoch_number
        if self.has_delay_passed():
            self.has_on_epoch_begin_been_called = True
            self.callbacks.on_epoch_begin(epoch_number, logs)

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        if self.has_delay_passed():
            self.callbacks.on_epoch_end(epoch_number, logs)

    def on_train_batch_begin(self, batch_number: int, logs: Dict):
        self.batch_counter += 1
        if self.has_delay_passed():
            if not self.has_on_epoch_begin_been_called:
                self.has_on_epoch_begin_been_called = True
                self.callbacks.on_epoch_begin(self.current_epoch, logs)
            self.callbacks.on_train_batch_begin(batch_number, logs)

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        if self.has_delay_passed():
            self.callbacks.on_train_batch_end(batch_number, logs)

    def on_backward_end(self, batch_number: int):
        if self.has_delay_passed():
            self.callbacks.on_backward_end(batch_number)

    def on_train_begin(self, logs: Dict):
        self.current_epoch = 0
        self.batch_counter = 0
        self.has_on_epoch_begin_been_called = False
        self.callbacks.on_train_begin(logs)

    def on_train_end(self, logs: Dict):
        self.callbacks.on_train_end(logs)

    def has_delay_passed(self):
        return self.current_epoch > self.epoch_delay and self.batch_counter > self.batch_delay
