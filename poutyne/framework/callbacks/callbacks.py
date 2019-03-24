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


class CallbackList(object):
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_backward_end(self, batch):
        for callback in self.callbacks:
            callback.on_backward_end(batch)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    """
    Attributes:
        params (dict): Contains a key 'epoch' and a key 'steps_per_epoch'
            which are passed to the `fit` function in `Model`. It may
            contain other keys.
        model (Model): a reference to the `Model` object which is using the
            callback.
    """
    def __init__(self):
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs):
        """
        Is called before the begining of each epoch.

        Args:
            epoch (int): The epoch number.
            logs (dict): Usually an empty dict.
        """
        pass

    def on_epoch_end(self, epoch, logs):
        """
        Is called before the end of each epoch.

        Args:
            epoch (int): The epoch number.
            logs (dict): Contains the following keys:

                 * 'epoch': The epoch number.
                 * 'loss': The average loss of the batches.
                 * 'time': The computation time of the epoch.
                 * Other metrics: One key for each type of metrics. The metrics
                   are also averaged.
                 * val_loss': The average loss of the batches on the validation
                   set.
                 * Other metrics: One key for each type of metrics on the
                   validation set. The metrics are also averaged.

        Example::

            logs = {'epoch': 6, 'time': 3.141519837, 'loss': 4.34462, 'accuracy': 0.766,
                    'val_loss': 5.2352, 'val_accuracy': 0.682}
        """
        pass

    def on_batch_begin(self, batch, logs):
        """
        Is called before the begining of each batch.

        Args:
            batch (int): The batch number.
            logs (dict): Usually an empty dict.
        """
        pass

    def on_batch_end(self, batch, logs):
        """
        Is called before the end of each batch.

        Args:
            batch (int): The batch number.
            logs (dict): Contains the following keys:

                 * 'batch': The batch number.
                 * 'loss': The loss of the batch.
                 * 'time': The computation time of the batch.
                 * Other metrics: One key for each type of metrics.

        Example::

            logs = {'batch': 6, 'time': 0.10012837, 'loss': 4.34462, 'accuracy': 0.766}
        """
        pass

    def on_backward_end(self, batch):
        """
        Is called after the backpropagation but before the optimization step.

        Args:
            batch (int): The batch number.
        """
        pass

    def on_train_begin(self, logs):
        """
        Is called before the begining of the training.

        Args:
            logs (dict): Usually an empty dict.
        """
        pass

    def on_train_end(self, logs):
        """
        Is called before the end of the training.

        Args:
            logs (dict): Usually an empty dict.
        """
        pass
