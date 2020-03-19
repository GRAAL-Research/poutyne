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
import warnings
from typing import Dict, List


class Callback:
    """
    Attributes:
        params (dict): Contains 'epoch' and 'steps_per_epoch' keys which are passed to the
            :func:`Model.fit() <poutyne.framework.Model.fit>` function. It may contain other keys.
        model (Model): A reference to the :class:`~poutyne.framework.Model` object which is using the callback.
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

                 * 'epoch': The epoch number.
                 * 'loss': The average loss of the batches.
                 * 'time': The computation time of the epoch.
                 * Other metrics: One key for each type of metrics. The metrics are also averaged.
                 * val_loss': The average loss of the batches on the validation set.
                 * Other metrics: One key for each type of metrics on the validation set. The metrics are also averaged.

        Example::

            logs = {'epoch': 6, 'time': 3.141519837, 'loss': 4.34462, 'accuracy': 0.766,
                    'val_loss': 5.2352, 'val_accuracy': 0.682}
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
            logs (dict): Usually an empty dict.
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
            logs (dict): Usually an empty dict.
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
            logs (dict): Usually an empty dict.
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
    def __init__(self, callbacks: List[Callback]):
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
            if hasattr(callback, 'on_batch_begin'):
                warnings.warn(
                    'on_batch_begin method for callback has been deprecated as of version 0.7. '
                    'Use on_batch_train_begin instead.',
                    Warning,
                    stacklevel=2)
                callback.on_batch_begin(batch_number, logs)
            else:
                callback.on_train_batch_begin(batch_number, logs)

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_end'):
                warnings.warn(
                    'on_batch_end method for callback has been deprecated as of version 0.7. '
                    'Use on_batch_train_end instead.',
                    Warning,
                    stacklevel=2)
                callback.on_batch_end(batch_number, logs)
            else:
                callback.on_train_batch_end(batch_number, logs)

    def on_test_batch_begin(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_batch_begin(batch_number, logs)

    def on_test_batch_end(self, batch_number: int, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_batch_end(batch_number, logs)

    def on_train_begin(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_test_begin(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs: Dict):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_backward_end(self, batch_number: int):
        for callback in self.callbacks:
            callback.on_backward_end(batch_number)

    def __iter__(self):
        return iter(self.callbacks)
