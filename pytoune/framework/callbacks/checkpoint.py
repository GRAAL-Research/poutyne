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


class ModelCheckpoint(Callback):
    """
    The source code of this class is under the MIT License and was copied from
    the Keras project, and has been modified.

    Save the model after every epoch. `filename` can contain named formatting
    options, which will be filled the value of `epoch` and keys in `logs`
    (passed in `on_epoch_end`). For example: if `filename` is
    `weights.{epoch:02d}-{val_loss:.2f}.ckpt`, then the model checkpoints will
    be saved with the epoch number and the validation loss in the filename.

    Args:
        filename (string): Path to save the model file.
        monitor (string): Quantity to monitor. (Default value = 'val_loss')
        verbose (bool): Whether to display a message when saving a checkpoint.
            (Default value = False)
        save_best_only (bool): If `save_best_only` is true, the latest best
            model according to the quantity monitored will not be overwritten.
            (Default value = False)
        restore_best (bool): If `restore_best` is true, the weights of the
            network will be reset to the last best checkpoint done. This option
            only works when `save_best_only` is also true.
            (Default value = False)
        mode (string): One of {min, max}.
            If `save_best_only` is true, the decision to overwrite the current
            save file is made based on either the maximization or the
            minimization of the monitored quantity. For `val_accuracy`, this
            should be `max`, for `val_loss` this should be `min`, etc.
            (Default value = 'min')
        period (int): Interval (number of epochs) between checkpoints.
            (Default value = 1)
    """

    def __init__(self, filename, monitor='val_loss', verbose=False, save_best_only=False, restore_best=False, mode='min', period=1):
        self.filename = filename
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.restore_best = restore_best
        self.best_filename = None

        if self.save_best_only or self.restore_best:
            if mode not in ['min', 'max']:
                raise ValueError("Invalid mode '%s'" % mode)
            if mode == 'min':
                self.monitor_op = lambda x,y: x < y
                self.current_best = float('Inf')
            elif mode == 'max':
                self.monitor_op = lambda x,y: x > y
                self.current_best = -float('Inf')

        self.period = period

    def _save_weights(self, filename):
        with tempfile.NamedTemporaryFile(delete=False) as fp:
            tmp_filename = fp.name
            self.model.save_weights(fp)
        try:
            os.replace(tmp_filename, filename)
        except OSError as e:
            # This may happen if the temp filesystem is not the same as the final destination's.
            warnings.warn('Impossible to move the checkpoint to its final destination: os.replace(%s, %s) -> %s' % (tmp_filename, filename, e))
            warnings.warn('Saving %s non-atomically.' % filename)
            self.model.save_weights(filename)


    def on_epoch_end(self, epoch, logs):
        filename = self.filename.format_map(logs)

        if self.save_best_only or self.restore_best:
            if self.monitor_op(logs[self.monitor], self.current_best):
                old_best = self.current_best
                self.current_best = logs[self.monitor]
                self.best_filename = filename

                if self.verbose:
                    print('Epoch %d: %s improved from %0.5f to %0.5f, saving model to %s'
                          % (epoch, self.monitor, old_best, self.current_best, self.best_filename))
                self._save_weights(self.best_filename)
                return

        if epoch % self.period == 0 and not self.save_best_only:
            if self.verbose:
                print('Epoch %d: saving model to %s' % (epoch, filename))
            self._save_weights(filename)
            return

    def on_train_end(self, logs):
        if self.restore_best:
            if self.best_filename is not None:
                self.model.load_weights(self.best_filename)
            else:
                warnings.warn('No  weights to restore!')
