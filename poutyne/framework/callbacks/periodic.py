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

import os
import warnings
from typing import IO, Dict, Optional, Callable
from abc import ABC, abstractmethod

from ._utils import atomic_lambda_save
from .callbacks import Callback


class PeriodicSaveCallback(ABC, Callback):
    """
    Write a file (or checkpoint) after every epoch. ``filename`` can contain named formatting options,
    which will be filled the value of ``epoch`` and keys in ``logs`` (passed in ``on_epoch_end``). For
    example: if ``filename`` is ``weights.{epoch:02d}-{val_loss:.2f}.txt``, then ``save_file()`` will be
    called with a file descriptor for a file with the epoch number and the validation loss in the
    filename.

    By default, the file is written atomically to the specified filename so that the training can
    be killed and restarted later using the same filename for periodic file saving. To do so, a
    temporary file is created with the name of ``filename + '.tmp'`` and is then moved to the final
    destination after the checkpoint is done. The ``temporary_filename`` argument allows to change the
    path of this temporary file.

    Args:
        filename (str): Path to save the model file.
        monitor (str): Quantity to monitor.
            (Default value = 'val_loss')
        verbose (bool): Whether to display a message when saving and restoring a checkpoint.
            (Default value = False)
        save_best_only (bool): If `save_best_only` is true, the latest best model according to the
            quantity monitored will not be overwritten.
            (Default value = False)
        keep_only_last_best (bool): Whether only the last saved best checkpoint is kept. Applies only when
             `save_best_only` is true.
             (Default value = False)
        restore_best (bool): If `restore_best` is true, the model will be reset to the last best checkpoint done.
            This option only works when `save_best_only` is also true. (Default value = False)
        mode (str): One of {'min', 'max'}.
            If `save_best_only` is true, the decision to overwrite the current save file is made based
            on either the maximization or the minimization of the monitored quantity. For
            `val_accuracy`, this should be `max`, for `val_loss` this should be `min`, etc.
            (Default value = 'min')
        period (int): Interval (number of epochs) between checkpoints.
            (Default value = 1)
        temporary_filename (str, optional): Temporary filename for the checkpoint so that the
            last checkpoint can be written atomically. See the ``atomic_write`` argument.
        atomic_write (bool): Whether to write atomically the checkpoint. See the description above
            for details.
            (Default value = True)
        open_mode (str): ``mode`` option passed to :func:`open()`.
            (Default value = 'wb')
    """

    def __init__(
        self,
        filename: str,
        *,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = False,
        keep_only_last_best: bool = False,
        restore_best: bool = False,
        period: int = 1,
        verbose: bool = False,
        temporary_filename: Optional[str] = None,
        atomic_write: bool = True,
        open_mode: str = 'wb',
        read_mode: str = 'rb',
    ) -> None:
        super().__init__()
        self.filename = filename
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.keep_only_last_best = keep_only_last_best
        self.restore_best = restore_best
        self.temporary_filename = temporary_filename
        self.atomic_write = atomic_write
        self.open_mode = open_mode
        self.read_mode = read_mode
        self.best_filename = None

        if self.keep_only_last_best and not self.save_best_only:
            raise ValueError("The 'keep_only_last_best' argument only works when 'save_best_only' is also true.")
        if self.restore_best and not self.save_best_only:
            raise ValueError("The 'restore_best' argument only works when 'save_best_only' is also true.")

        if self.save_best_only:
            if mode not in ['min', 'max']:
                raise ValueError(f"Invalid mode '{mode}'")
            if mode == 'min':
                self.monitor_op = lambda x, y: x < y
                self.current_best = float('Inf')
            elif mode == 'max':
                self.monitor_op = lambda x, y: x > y
                self.current_best = -float('Inf')

        self.period = period

    @abstractmethod
    def save_file(self, fd: IO, epoch_number: int, logs: Dict) -> None:
        """
        Abstract method that is called every time a save needs to be done.

        Args:
            fd (IO): The descriptor of the file in which to write.
            epoch_number (int): The epoch number.
            logs (Dict): Dictionary passed on epoch end.
        """
        pass

    def _save_file(self, filename: str, epoch_number: int, logs: Dict) -> None:
        atomic_lambda_save(
            filename,
            self.save_file,
            (epoch_number, logs),
            temporary_filename=self.temporary_filename,
            open_mode=self.open_mode,
            atomic=self.atomic_write,
        )

    def on_epoch_end(self, epoch_number: int, logs: Dict) -> None:
        filename = self.filename.format_map(logs)

        if self.save_best_only:
            if self.monitor in logs:
                if self.monitor_op(logs[self.monitor], self.current_best):
                    old_best = self.current_best
                    self.current_best = logs[self.monitor]
                    old_best_filename = self.best_filename
                    self.best_filename = filename

                    if self.verbose:
                        print(
                            f'Epoch {epoch_number:d}: {self.monitor} improved from {old_best:0.5f} '
                            f'to {self.current_best:0.5f}, saving file to {self.best_filename}'
                        )
                    self._save_file(self.best_filename, epoch_number, logs)
                    if (
                        self.keep_only_last_best
                        and self.best_filename != old_best_filename
                        and old_best_filename is not None
                    ):
                        os.remove(old_best_filename)
            else:
                raise KeyError(f"The monitored metric name {self.monitor} is not found in computed metrics.")
        elif epoch_number % self.period == 0:
            if self.verbose:
                print(f'Epoch {epoch_number:d}: saving file to {filename}')
            self._save_file(filename, epoch_number, logs)

    @abstractmethod
    def restore(self, fd: IO) -> None:
        """
        Abstract method that is called when a save needs to be restored. This happens at the end of the training when
        ``restore_best`` is true.

        Args:
            fd (IO): The descriptor of the file to read.
        """
        pass

    def on_train_end(self, logs: Dict) -> None:
        if self.restore_best:
            if self.best_filename is not None:
                if self.verbose:
                    print(f'Restoring data from {self.best_filename}')
                # pylint: disable=unspecified-encoding
                open_kwargs = dict(encoding='utf-8') if 'b' not in self.read_mode else {}
                with open(self.best_filename, self.read_mode, **open_kwargs) as fd:
                    self.restore(fd)
            else:
                warnings.warn('No data to restore!')


class PeriodicSaveLambda(PeriodicSaveCallback):
    """
    Call a lambda with a file descriptor after every epoch. See
    :class:`~poutyne.PeriodicSaveCallback` for the arguments' descriptions.

    Args:
        func (Callable[[fd, int, dict], None]): The lambda that will be called with a file descriptor, the
            epoch number and the epoch logs.
        restore (Callable[[fd], None]): The lambda that will be called with a file descriptor to restore
            the state if necessary.

    See:
        :class:`~poutyne.PeriodicSaveCallback`
    """

    def __init__(self, func: Callable, *args, restore: Optional[Callable] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.func = func
        self._restore = restore

    def save_file(self, fd: IO, epoch_number: int, logs: Dict) -> None:
        self.func(fd, epoch_number, logs)

    def restore(self, fd: IO) -> None:
        if self._restore is not None:
            self._restore(fd)
