"""
The source code of this file was copied from the Keras project, and has been modified.

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

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

LICENSE

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
from typing import BinaryIO, Dict, Optional, Callable

from ._utils import atomic_lambda_save
from .callbacks import Callback


class PeriodicSaveCallback(Callback):
    """
    The source code of this class is under the MIT License and was copied from the Keras project,
    and has been modified.

    Write a file after every epoch. `filename` can contain named formatting options, which will be
    filled the value of `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
    `filename` is `weights.{epoch:02d}-{val_loss:.2f}.txt`, then `save_file()` will be called with a
    file descriptor for a file with the epoch number and the validation loss in the filename.

    By default, the file is written atomically to the specified filename so that the training can
    be killed and restarted later using the same filename for periodic file saving. To do so, a
    temporary file is created using the system's `tmp` directory and then is moved to the final
    destination after the checkpoint is made. Sometimes, this move is not possible on some systems.
    To address this problem, it is possible to specify the destination of the temporary file using
    the ``temporary_filename`` argument.

    Args:
        filename (str): Path to save the model file.
        monitor (str): Quantity to monitor.
            (Default value = 'val_loss')
        verbose (bool): Whether to display a message when saving and restoring a checkpoint.
            (Default value = False)
        save_best_only (bool): If `save_best_only` is true, the latest best model according to the
            quantity monitored will not be overwritten.
            (Default value = False)
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

    def __init__(self,
                 filename: str,
                 *,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = False,
                 period: int = 1,
                 verbose: bool = False,
                 temporary_filename: Optional[str] = None,
                 atomic_write: bool = True,
                 open_mode: str = 'wb'):
        super().__init__()
        self.filename = filename
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.temporary_filename = temporary_filename
        self.atomic_write = atomic_write
        self.open_mode = open_mode
        self.best_filename = None

        if self.save_best_only:
            if mode not in ['min', 'max']:
                raise ValueError("Invalid mode '%s'" % mode)
            if mode == 'min':
                self.monitor_op = lambda x, y: x < y
                self.current_best = float('Inf')
            elif mode == 'max':
                self.monitor_op = lambda x, y: x > y
                self.current_best = -float('Inf')

        self.period = period

    def save_file(self, fd: BinaryIO, epoch_number: int, logs: Dict):
        raise NotImplementedError

    def _save_file(self, filename: str, epoch_number: int, logs: Dict):
        atomic_lambda_save(filename,
                           self.save_file, (epoch_number, logs),
                           temporary_filename=self.temporary_filename,
                           open_mode=self.open_mode,
                           atomic=self.atomic_write)

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        filename = self.filename.format_map(logs)

        if self.save_best_only:
            if self.monitor_op(logs[self.monitor], self.current_best):
                old_best = self.current_best
                self.current_best = logs[self.monitor]
                self.best_filename = filename

                if self.verbose:
                    print('Epoch %d: %s improved from %0.5f to %0.5f, saving file to %s' %
                          (epoch_number, self.monitor, old_best, self.current_best, self.best_filename))
                self._save_file(self.best_filename, epoch_number, logs)
        elif epoch_number % self.period == 0:
            if self.verbose:
                print('Epoch %d: saving file to %s' % (epoch_number, filename))
            self._save_file(filename, epoch_number, logs)


class PeriodicSaveLambda(PeriodicSaveCallback):
    """
    Call a lambda with a file descriptor after every epoch. See
    :class:`~poutyne.framework.callbacks.PeriodicSaveCallback` for the arguments' descriptions.

    Args:
        func (Callable[[fd, int, dict], None]): The lambda that will be called with a file descriptor, the
            epoch number and the epoch logs.

    See:
        :class:`~poutyne.framework.callbacks.PeriodicSaveCallback`
    """

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def save_file(self, fd: str, epoch_number: int, logs: Dict):
        self.func(fd, epoch_number, logs)
