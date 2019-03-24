"""
The source code of this file was copied from the Keras project, and has been
modified.

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

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import warnings
import tempfile

from .callbacks import Callback


class PeriodicSaveCallback(Callback):
    """
    The source code of this class is under the MIT License and was copied from
    the Keras project, and has been modified.

    Write a file after every epoch. `filename` can contain named formatting
    options, which will be filled the value of `epoch` and keys in `logs`
    (passed in `on_epoch_end`). For example: if `filename` is
    `weights.{epoch:02d}-{val_loss:.2f}.txt`, then `save_file()` will be called
    with a file descriptor for a file with the epoch number and the validation
    loss in the filename.

    By default, the file are written atomically to the specified filename
    so that the training can be killed and restarted later using the same
    filename for periodic file saving. To do so, a temporary file is created
    using the system's `tmp` directory and then is moved a the final destination
    after the checkpoint is made. Sometimes, this move is not possible on some
    system. To address this problem, it is possible to specify the destination
    of the temporary file using the ``temporary_filename`` argument.

    Args:
        filename (string): Path to save the model file.
        monitor (string): Quantity to monitor. (Default value = 'val_loss')
        verbose (bool): Whether to display a message when saving and restoring
            a checkpoint. (Default value = False)
        save_best_only (bool): If `save_best_only` is true, the latest best
            model according to the quantity monitored will not be overwritten.
            (Default value = False)
        mode (string): One of {min, max}.
            If `save_best_only` is true, the decision to overwrite the current
            save file is made based on either the maximization or the
            minimization of the monitored quantity. For `val_accuracy`, this
            should be `max`, for `val_loss` this should be `min`, etc.
            (Default value = 'min')
        period (int): Interval (number of epochs) between checkpoints.
            (Default value = 1)
        temporary_filename (string, optional): Temporary filename for the
            checkpoint so that the last checkpoint can be written
            atomically. See the ``atomic_write`` argument.
        atomic_write (bool): Whether to right atomically the checkpoint. See
            the description above for details. (Default value = True)
        open_mode (str): ``mode`` option passed to ``open()``.
            (Default value = 'wb')
    """

    def __init__(self, filename, *,
                 monitor='val_loss', mode='min', save_best_only=False, period=1, verbose=False,
                 temporary_filename=None, atomic_write=True, open_mode='wb'):
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

    def save_file(self, fd, epoch, logs):
        raise NotImplementedError

    def _save_file(self, filename, epoch, logs):
        if self.atomic_write:
            fd = None
            if self.temporary_filename is not None:
                fd = open(self.temporary_filename, self.open_mode)
                tmp_filename = self.temporary_filename
            else:
                fd = tempfile.NamedTemporaryFile(delete=False)
                tmp_filename = fd.name

            with fd:
                self.save_file(fd, epoch, logs)

            try:
                os.replace(tmp_filename, filename)
            except OSError as e:
                # This may happen if the temp filesystem is not the same as the final destination's.
                warnings.warn(
                    "Impossible to move the file to its final destination: "
                    "os.replace(%s, %s) -> %s\nYou may want to specify the "
                    "'temporary_filename' argument to %s." % (
                        tmp_filename, filename, e, self.__class__.__name__
                    )
                )
                os.remove(tmp_filename)

                warnings.warn('Saving %s non-atomically instead.' % filename)
                with open(filename, self.open_mode) as fd:
                    self.save_file(fd, epoch, logs)
        else:
            with open(filename, self.open_mode) as fd:
                self.save_file(fd, epoch, logs)


    def on_epoch_end(self, epoch, logs):
        filename = self.filename.format_map(logs)

        if self.save_best_only:
            if self.monitor_op(logs[self.monitor], self.current_best):
                old_best = self.current_best
                self.current_best = logs[self.monitor]
                self.best_filename = filename

                if self.verbose:
                    print('Epoch %d: %s improved from %0.5f to %0.5f, saving file to %s'
                          % (epoch, self.monitor, old_best, self.current_best, self.best_filename))
                self._save_file(self.best_filename, epoch, logs)
        elif epoch % self.period == 0:
            if self.verbose:
                print('Epoch %d: saving file to %s' % (epoch, filename))
            self._save_file(filename, epoch, logs)

class PeriodicSaveLambda(PeriodicSaveCallback):
    """
    Call a lambda with a file descriptor after every epoch. See
    `poutyne.framework.PeriodicSaveCallback` for the arguments' descriptions.

    Args:
        func (fd, int, dict -> None): The lambda that will be called with a
            file descriptor, the epoch number and the epoch logs.

    See:
        poutyne.framework.PeriodicSaveCallback
    """
    def __init__(self, func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.func = func

    def save_file(self, fd, epoch, logs):
        self.func(fd, epoch, logs)
