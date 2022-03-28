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

from typing import Dict, Sequence


class Callback:
    # pylint: disable=too-many-public-methods
    """
    Attributes:
        params (dict): Contains ``'epoch'`` and ``'steps_per_epoch'`` keys which are passed to the
            when training. Contains ``'steps'`` when evaluating. May contain other keys.
        model (Model): A reference to the :class:`~poutyne.Model` object which is using the callback.
    """

    def __init__(self):
        self.model = None
        self.params = None

    def set_params(self, params: Dict):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch_number: int, logs: Dict):
        """
        Is called before the beginning of each epoch.

        Args:
            epoch_number (int): The epoch number.
            logs (dict): Usually an empty dict.
        """
        pass

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        """
        Is called before the end of each epoch.

        Args:
            epoch_number (int): The epoch number.
            logs (dict): Contains the following keys:

                 * ``'epoch'``: The epoch number.
                 * ``'time'``: The computation time of the epoch.
                 * ``'loss'``: The average loss of the batches.
                 * Values of training metrics: One key for each type of metrics. The metrics are also averaged.
                 * ``'val_loss'``: The average loss of the batches on the validation set.
                 * Values of validation metrics: One key for each type of metrics on the validation set. Each key is
                   prefixed by ``'val_'``. The metrics are also averaged.

        Example::

            logs = {'epoch': 2, 'time': 6.08248, 'loss': 0.40161, 'acc': 89.052, 'fscore_micro': 0.89051,
                    'val_loss': 0.36814, 'val_acc': 89.52, 'val_fscore_micro': 0.89520}
        """
        pass

    def on_train_batch_begin(self, batch_number: int, logs: Dict):
        """
        Is called before the beginning of the training batch.

        Args:
            batch_number (int): The batch number.
            logs (dict): Usually an empty dict.
        """
        pass

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        """
        Is called before the end of the training batch.

        Args:
            batch_number (int): The batch number.
            logs (dict): Contains the following keys:

                 * ``'batch'``: The batch number.
                 * ``'size'``: The size of the batch as inferred by :func:`~Model.get_batch_size()`.
                 * ``'time'``: The computation time of the batch.
                 * ``'loss'``: The loss of the batch.
                 * Values of the batch metrics for the specific batch: One key for each type of metrics.

        Example::

            logs = {'batch': 171, 'size': 32, 'time': 0.00310, 'loss': 1.95204, 'acc': 43.75}
        """
        pass

    def on_valid_batch_begin(self, batch_number: int, logs: Dict):
        """
        Is called before the beginning of the validation batch.

        Args:
            batch_number (int): The batch number.
            logs (dict): Usually an empty dict.
        """
        pass

    def on_valid_batch_end(self, batch_number: int, logs: Dict):
        """
        Is called before the end of the validation batch.

        Args:
            batch_number (int): The batch number.
            logs (dict): Contains the following keys:

                 * ``'batch'``: The batch number.
                 * ``'size'``: The size of the batch as inferred by :func:`~Model.get_batch_size()`.
                 * ``'time'``: The computation time of the batch.
                 * ``val_loss'``: The loss of the batch.
                 * Values of the batch metrics for the specific batch: One key for each type of metrics. Each key is
                   prefixed by ``'val_'``. The metrics are also averaged.

        Example::

            logs = {'batch': 171, 'size': 32, 'time': 0.00310, 'val_loss': 1.95204, 'val_acc': 43.75}
        """
        pass

    def on_test_batch_begin(self, batch_number: int, logs: Dict):
        """
        Is called before the beginning of the testing batch.

        Args:
            batch_number (int): The batch number.
            logs (dict): Usually an empty dict.
        """
        pass

    def on_test_batch_end(self, batch_number: int, logs: Dict):
        """
        Is called before the end of the testing batch.

        Args:
            batch_number (int): The batch number.
            logs (dict): Contains the following keys:

                 * ``'batch'``: The batch number.
                 * ``'size'``: The size of the batch as inferred by :func:`~Model.get_batch_size()`.
                 * ``'time'``: The computation time of the batch.
                 * ``'loss'``: The loss of the batch.
                 * Values of the batch metrics for the specific batch: One key for each type of metrics. Each key is
                   prefixed by ``'test_'``. The metrics are also averaged.

        Example::

            logs = {'batch': 171, 'size': 32, 'time': 0.00310, 'test_loss': 1.95204, 'test_acc': 43.75}
        """
        pass

    def on_predict_batch_begin(self, batch_number: int, logs: Dict):
        """
        Is called before the beginning of the predict batch.

        Args:
            batch_number (int): The batch number.
            logs (dict): Usually an empty dict.
        """
        pass

    def on_predict_batch_end(self, batch_number: int, logs: Dict):
        """
        Is called before the end of the predict batch.

        Args:
            batch_number (int): The batch number.
            logs (dict): Contains the following keys:

                 * ``'batch'``: The batch number.
                 * ``'time'``: The computation time of the batch.

        Example::

            logs = {'batch': 171, 'time': 0.00310}
        """
        pass

    def on_train_begin(self, logs: Dict):
        """
        Is called before the beginning of the training.

        Args:
            logs (dict): Usually an empty dict.
        """
        pass

    def on_train_end(self, logs: Dict):
        """
        Is called before the end of the training.

        Args:
            logs (dict): Usually an empty dict.
        """
        pass

    def on_valid_begin(self, logs: Dict):
        """
        Is called before the beginning of the validation.

        Args:
            logs (dict): Usually an empty dict.
        """
        pass

    def on_valid_end(self, logs: Dict):
        """
        Is called before the end of the validation.

        Args:
            logs (dict): Contains the following keys:

                 * ``'time'``: The total computation time of the test.
                 * ``'val_loss'``: The average loss of the batches on the test set.
                 * Values of testing metrics: One key for each type of metrics. Each key is
                   prefixed by ``'val_'``. The metrics are also averaged.

        Example::

            logs = {'time': 6.08248, 'val_loss': 0.40161, 'val_acc': 89.052, 'val_fscore_micro': 0.89051}
        """
        pass

    def on_test_begin(self, logs: Dict):
        """
        Is called before the beginning of the testing.

        Args:
            logs (dict): Usually an empty dict.
        """
        pass

    def on_test_end(self, logs: Dict):
        """
        Is called before the end of the testing.

        Args:
            logs (dict): Contains the following keys:

                 * ``'time'``: The total computation time of the test.
                 * ``'test_loss'``: The average loss of the batches on the test set.
                 * Values of testing metrics: One key for each type of metrics. Each key is
                   prefixed by ``'test_'``. The metrics are also averaged.

        Example::

            logs = {'time': 6.08248, 'test_loss': 0.40161, 'test_acc': 89.052, 'test_fscore_micro': 0.89051}
        """
        pass

    def on_predict_begin(self, logs: Dict):
        """
        Is called before the beginning of the predict.

        Args:
            logs (dict): Usually an empty dict.
        """
        pass

    def on_predict_end(self, logs: Dict):
        """
        Is called before the end of the predict.

        Args:
            logs (dict): Contains the following keys:

                 * ``'time'``: The total computation time of the predict.

        Example::

            logs = {'time': 6.08248}
        """
        pass

    def on_backward_end(self, batch_number: int):
        """
        Is called after the backpropagation but before the optimization step.

        Args:
            batch_number (int): The batch number.
        """
        pass


class CallbackList:
    # pylint: disable=too-many-public-methods
    def __init__(self, callbacks: Sequence[Callback]):
        callbacks = callbacks or []
        self.callbacks = list(callbacks)

    def append(self, callback: Callback):
        self.callbacks.append(callback)

    def set_params(self, params: Dict):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch_number, logs)

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch_number, logs)

    def on_train_batch_begin(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch_number, logs)

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_batch_end(batch_number, logs)

    def on_valid_batch_begin(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_valid_batch_begin(batch_number, logs)

    def on_valid_batch_end(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_valid_batch_end(batch_number, logs)

    def on_test_batch_begin(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_batch_begin(batch_number, logs)

    def on_test_batch_end(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_batch_end(batch_number, logs)

    def on_predict_batch_begin(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_batch_begin(batch_number, logs)

    def on_predict_batch_end(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_batch_end(batch_number, logs)

    def on_train_begin(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_valid_begin(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_valid_begin(logs)

    def on_valid_end(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_valid_end(logs)

    def on_test_begin(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_predict_begin(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    def on_predict_end(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def on_backward_end(self, batch_number: int):
        for callback in self.callbacks:
            callback.on_backward_end(batch_number)

    def __iter__(self):
        return iter(self.callbacks)
