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

# pylint: disable=too-many-lines,too-many-public-methods
import contextlib
import timeit
from collections import defaultdict
from typing import Iterable, Mapping, List, Union, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import DataLoader

from poutyne import torch_to_numpy, numpy_to_torch, torch_to
from .callbacks import CallbackList, ProgressionCallback, Callback
from .iterators import EpochIterator, _get_step_iterator, StepIterator
from .metrics import get_metric
from .metrics import get_callables_and_names, rename_doubles, flatten_metric_names
from .metrics.decomposable import convert_decomposable_metric_to_object
from .optimizers import get_optimizer
from ..utils import TensorDataset, _concat, get_batch_size


class Model:
    """
    The Model class encapsulates a PyTorch network, a PyTorch optimizer, a loss function and
    metric functions. It allows the user to train a neural network without hand-coding the
    epoch/step logic.

    Args:
        network (torch.nn.Module): A PyTorch network.
        optimizer (Union[torch.optim.Optimizer, str, dict]): If torch.optim.Optimier, an initialized PyTorch.
            If str, should be the name of the optimizer in Pytorch (i.e. 'Adam' for torch.optim.Adam).
            If dict, should contain a key ``'optim'`` with the value be the name of the optimizer; other
            entries are passed to the optimizer as keyword arguments.
            (Default value = None)
        loss_function(Union[Callable, str]) It can be any PyTorch loss layer or custom loss function. It
            can also be a string with the same name as a PyTorch loss function (either the functional or
            object name). The loss function must have the signature ``loss_function(input, target)`` where
            ``input`` is the prediction of the network and ``target`` is the ground truth.
            (Default value = None)
        batch_metrics (list): List of functions with the same signature as a loss function or objects with the same
            signature as either :class:`~poutyne.Metric` or :class:`torchmetrics.Metric <torchmetrics.Metric>`. It can
            also be a string with the same name as a PyTorch loss function (either the functional or object name).
            Some metrics, such as  'accuracy' (or just 'acc'), are also available as strings. See :ref:`metrics` and
            the `TorchMetrics documentation <https://torchmetrics.readthedocs.io/en/latest/references/modules.html>`__
            for available metrics.

            Batch metric are computed on computed for each batch.
            (Default value = None)

            .. warning:: When using this argument, the metrics are computed for each batch. This can significantly slow
                down the compuations depending on the metrics used. This mostly happens on non-decomposable metrics
                such as :class:`torchmetrics.AUROC <torchmetrics.AUROC>` where an ordering of the elements is necessary
                to compute the metric. In such case, we advise to use them as epoch metrics instead.
        epoch_metrics (list): List of functions with the same signature as a loss function or objects with the same
            signature as either :class:`~poutyne.Metric` or :class:`torchmetrics.Metric <torchmetrics.Metric>`. It can
            also be a string with the same name as a PyTorch loss function (either the functional or object name).
            Some metrics, such as  'accuracy' (or just 'acc'), are also available as strings. See :ref:`metrics` and
            the `TorchMetrics documentation <https://torchmetrics.readthedocs.io/en/latest/references/modules.html>`__
            for available metrics.

            Epoch metrics are computed only at the end of the epoch.
            (Default value = None)
        device (Union[torch.torch.device, List[torch.torch.device]]): The device to which the network is
            sent or the list of device to which the network is sent. See :func:`~Model.to()` for details.

    Note:
        The name of each batch and epoch metric can be change by passing a tuple ``(name, metric)`` instead
        of simply the metric function or object, where ``name`` is the alternative name of the metric.
        Batch and epoch metrics can return multiple metrics (e.g. an epoch metric could return an F1-score
        with the associated precision and recall). See :ref:`multiple metrics at once` for more details.


    Attributes:
        network (torch.nn.Module): The associated PyTorch network.
        optimizer (torch.optim.Optimizer): The associated PyTorch optimizer.
        loss_function: The associated loss function.
        batch_metrics (list): The associated metric functions for every batch.
        epoch_metrics (list): The associated metric functions for every epoch.

    Examples:
        Using Numpy arrays (or tensors) dataset::

            from poutyne import Model
            import torch
            import numpy as np
            import torchmetrics

            num_features = 20
            num_classes = 5

            # Our training dataset with 800 samples.
            num_train_samples = 800
            train_x = np.random.randn(num_train_samples, num_features).astype('float32')
            train_y = np.random.randint(num_classes, size=num_train_samples).astype('int64')

            # Our validation dataset with 200 samples.
            num_valid_samples = 200
            valid_x = np.random.randn(num_valid_samples, num_features).astype('float32')
            valid_y = np.random.randint(num_classes, size=num_valid_samples).astype('int64')

            pytorch_network = torch.nn.Linear(num_features, num_classes) # Our network

            # We create and optimize our model
            model = Model(pytorch_network, 'sgd', 'cross_entropy',
                          batch_metrics=['accuracy'],
                          epoch_metrics=[torchmetrics.AUROC(num_classes=num_classes, task="multiclass")])
            model.fit(train_x, train_y,
                      validation_data=(valid_x, valid_y),
                      epochs=5,
                      batch_size=32)

        .. code-block:: none

            Epoch: 1/5 Train steps: 25 Val steps: 7 0.51s loss: 1.757784 acc: 20.750000 auroc: 0.494891
            val_loss: 1.756639 val_acc: 18.500000 val_auroc: 0.499404
            Epoch: 2/5 Train steps: 25 Val steps: 7 0.03s loss: 1.749623 acc: 20.375000 auroc: 0.496878
            val_loss: 1.748795 val_acc: 19.000000 val_auroc: 0.499723
            Epoch: 3/5 Train steps: 25 Val steps: 7 0.03s loss: 1.742070 acc: 20.250000 auroc: 0.499461
            val_loss: 1.741379 val_acc: 19.000000 val_auroc: 0.498577
            ...

        Using PyTorch DataLoader::

           import torch
           from torch.utils.data import DataLoader, TensorDataset
           from poutyne import Model
           import torchmetrics

           num_features = 20
           num_classes = 5

           # Our training dataset with 800 samples.
           num_train_samples = 800
           train_x = torch.rand(num_train_samples, num_features)
           train_y = torch.randint(num_classes, (num_train_samples,), dtype=torch.long)
           train_dataset = TensorDataset(train_x, train_y)
           train_generator = DataLoader(train_dataset, batch_size=32)

           # Our validation dataset with 200 samples.
           num_valid_samples = 200
           valid_x = torch.rand(num_valid_samples, num_features)
           valid_y = torch.randint(num_classes, (num_valid_samples,), dtype=torch.long)
           valid_dataset = TensorDataset(valid_x, valid_y)
           valid_generator = DataLoader(valid_dataset, batch_size=32)

           pytorch_network = torch.nn.Linear(num_features, num_classes)

           model = Model(pytorch_network, 'sgd', 'cross_entropy',
                         batch_metrics=['accuracy'],
                         epoch_metrics=[torchmetrics.AUROC(num_classes=num_classes, task="multiclass")])
           model.fit_generator(train_generator,
                               valid_generator,
                               epochs=5)

        .. code-block:: none

            Epoch: 1/5 Train steps: 25 Val steps: 7 0.07s loss: 1.614473 acc: 20.500000 auroc: 0.516850
            val_loss: 1.617141 val_acc: 21.500000 val_auroc: 0.522068
            Epoch: 2/5 Train steps: 25 Val steps: 7 0.03s loss: 1.614454 acc: 20.125000 auroc: 0.517618
            val_loss: 1.615585 val_acc: 22.000000 val_auroc: 0.521051
            Epoch: 3/5 Train steps: 25 Val steps: 7 0.03s loss: 1.613709 acc: 20.125000 auroc: 0.518307
            val_loss: 1.614440 val_acc: 22.000000 val_auroc: 0.520762
            ...

    """

    def __init__(
        self,
        network,
        optimizer,
        loss_function,
        *,
        batch_metrics=None,
        epoch_metrics=None,
        device=None,
    ):
        if not isinstance(network, nn.Module):
            raise ValueError(f"network should be of type derived from nn.Module, received {type(network)}.")

        if optimizer is not None and not isinstance(optimizer, (optim.Optimizer, str, dict)):
            raise ValueError(f"optimizer should be of type derived from optim.Optimizer, received {type(optimizer)}.")

        batch_metrics = [] if batch_metrics is None else batch_metrics
        epoch_metrics = [] if epoch_metrics is None else epoch_metrics

        self.network = network
        self.optimizer = get_optimizer(optimizer, self.network)

        self.loss_function = loss_function
        if self.loss_function is not None:
            self.loss_function = get_metric(loss_function)
            if isinstance(self.loss_function, tuple):
                self.loss_function = self.loss_function[1]
            self.loss_function = convert_decomposable_metric_to_object(self.loss_function, 'loss')

        self._check_network_optimizer_parameters_match()
        self._set_metrics_attributes(batch_metrics, epoch_metrics)

        self.device = None
        self.other_device = None

        if device is not None:
            self.to(device)

    def _check_network_optimizer_parameters_match(self):
        if self.optimizer is not None:
            param_set = set(self.network.parameters())
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param not in param_set:
                        raise ValueError(
                            "All parameters in the optimizer should be part of the network. "
                            "This is so to insure that weights checkpointing and the likes "
                            "actually consider all parameters."
                        )

    def _set_metrics_attributes(self, batch_metrics, epoch_metrics):
        batch_metrics = list(map(get_metric, batch_metrics))
        batch_metrics, batch_metrics_names = get_callables_and_names(batch_metrics)
        self.batch_metrics = [
            convert_decomposable_metric_to_object(metric, names)
            for metric, names in zip(batch_metrics, batch_metrics_names)
        ]

        epoch_metrics = list(map(get_metric, epoch_metrics))
        epoch_metrics, epoch_metrics_names = get_callables_and_names(epoch_metrics)
        self.epoch_metrics = [
            convert_decomposable_metric_to_object(metric, names, is_epoch_metric=True)
            for metric, names in zip(epoch_metrics, epoch_metrics_names)
        ]

        self.original_batch_metrics_names, self.original_epoch_metrics_names = (
            batch_metrics_names,
            epoch_metrics_names,
        )
        batch_metrics_names, epoch_metrics_names = rename_doubles(batch_metrics_names, epoch_metrics_names)

        self.unflatten_batch_metrics_names = batch_metrics_names
        self.unflatten_epoch_metrics_names = epoch_metrics_names

        self.batch_metrics_names = flatten_metric_names(batch_metrics_names)
        self.epoch_metrics_names = flatten_metric_names(epoch_metrics_names)
        self.metrics_names = self.batch_metrics_names + self.epoch_metrics_names

    @contextlib.contextmanager
    def _set_training_mode(self, training):
        old_training = self.network.training
        self.network.train(training)
        with torch.set_grad_enabled(training):
            yield
        self.network.train(old_training)

    def fit(
        self,
        x,
        y,
        validation_data=None,
        *,
        batch_size=32,
        epochs=1000,
        steps_per_epoch=None,
        validation_steps=None,
        batches_per_step=1,
        initial_epoch=1,
        verbose=True,
        progress_options: Union[dict, None] = None,
        callbacks=None,
        dataloader_kwargs=None,
    ):
        # pylint: disable=line-too-long,too-many-locals
        """
        Trains the network on a dataset. This method creates generators and calls
        the :func:`~Model.fit_generator()` method.

        Args:
            x (Union[~torch.Tensor, ~numpy.ndarray] or Union[tuple, list] of Union[~torch.Tensor, ~numpy.ndarray]):
                Training dataset. Union[Tensor, ndarray] if the model has a single input.
                Union[tuple, list] of Union[Tensor, ndarray] if the model has multiple inputs.
            y (Union[~torch.Tensor, ~numpy.ndarray] or Union[tuple, list] of Union[~torch.Tensor, ~numpy.ndarray]):
                Target. Union[Tensor, ndarray] if the model has a single output.
                Union[tuple, list] of Union[Tensor, ndarray] if the model has multiple outputs.
            validation_data (Tuple[``x_val``, ``y_val``]):
                Same format as ``x`` and ``y`` previously described. Validation dataset on which to
                evaluate the loss and any model metrics at the end of each epoch. The model will not be
                trained on this data.
                (Default value = None)
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)
            epochs (int): Number of times the entire training dataset is seen.
                (Default value = 1000)
            steps_per_epoch (int, optional): Number of batch used during one epoch. Obviously, using
                this argument may cause one epoch not to see the entire training dataset or see it
                multiple times.
                (Defaults the number of steps needed to see the entire training dataset)
            validation_steps (int, optional): Same as for ``steps_per_epoch`` but for the validation
                dataset.
                (Defaults to the number of steps needed to see the entire validation dataset)
            batches_per_step (int): Number of batches on which to compute the running loss before
                backpropagating it through the network. Note that the total loss used for backpropagation is
                the mean of the `batches_per_step` batch losses.
                (Default value = 1)
            initial_epoch (int, optional): Epoch at which to start training
                (useful for resuming a previous training run).
                (Default value = 1)
            verbose (bool): Whether to display the progress of the training.
                (Default value = True)
            progress_options (dict, optional): Keyword arguments to pass to the default progression callback used
                in Poutyne (See :class:`~poutyne.ProgressionCallback` for the available arguments).
                (Default value = None)
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called
                during training.
                (Default value = None)
            dataloader_kwargs (dict, optional): Keyword arguments to pass to the PyTorch dataloaders created
                internally. By default, ``shuffle=True`` is passed for the training dataloader but this can be
                overridden by using this argument.

        Returns:
            List of dict containing the history of each epoch.

        Example:
            .. code-block:: python

                model = Model(pytorch_network, optimizer, loss_function)
                history = model.fit(train_x, train_y,
                                    validation_data=(valid_x, valid_y)
                                    epochs=num_epochs,
                                    batch_size=batch_size,
                                    verbose=False)
                print(*history, sep="\\n")

            .. code-block:: python

                {'epoch': 1, 'loss': 1.7198852968215943, 'time': 0.019999928001197986, 'acc': 19.375, 'val_loss': 1.6674459838867188, 'val_acc': 22.0}
                {'epoch': 2, 'loss': 1.7054892110824584, 'time': 0.015421080999658443, 'acc': 19.75, 'val_loss': 1.660806336402893, 'val_acc': 22.0}
                {'epoch': 3, 'loss': 1.6923445892333984, 'time': 0.01363091799794347, 'acc': 19.625, 'val_loss': 1.6550078630447387, 'val_acc': 22.5}
                ...

        """
        train_dataset = self._dataset_from_data((x, y))
        valid_dataset = None
        if validation_data is not None:
            valid_dataset = self._dataset_from_data(validation_data)

        return self.fit_dataset(
            train_dataset,
            valid_dataset=valid_dataset,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            batches_per_step=batches_per_step,
            initial_epoch=initial_epoch,
            verbose=verbose,
            progress_options=progress_options,
            callbacks=callbacks,
            dataloader_kwargs=dataloader_kwargs,
        )

    def _dataset_from_data(self, args):
        args = numpy_to_torch(args)
        return TensorDataset(*args) if len(args) > 1 else args[0]

    def fit_dataset(
        self,
        train_dataset,
        valid_dataset=None,
        *,
        batch_size=32,
        epochs=1000,
        steps_per_epoch=None,
        validation_steps=None,
        batches_per_step=1,
        initial_epoch=1,
        verbose=True,
        progress_options=None,
        callbacks=None,
        num_workers=0,
        collate_fn=None,
        dataloader_kwargs=None,
    ):
        # pylint: disable=line-too-long,too-many-locals
        """
        Trains the network on a dataset. This method creates dataloaders and calls the
        :func:`~Model.fit_generator()` method.

        Args:
            train_dataset (~torch.utils.data.Dataset): Training dataset.
            valid_dataset (~torch.utils.data.Dataset): Validation dataset.
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)
            epochs (int): Number of times the entire training dataset is seen.
                (Default value = 1000)
            steps_per_epoch (int, optional): Number of batch used during one epoch. Obviously, using
                this argument may cause one epoch not to see the entire training dataset or see it
                multiple times.
                (Defaults the number of steps needed to see the entire training dataset)
            validation_steps (int, optional): Same as for ``steps_per_epoch`` but for the validation
                dataset.
                (Defaults to the number of steps needed to see the entire validation dataset)
            batches_per_step (int): Number of batches on which to compute the running loss before
                backpropagating it through the network. Note that the total loss used for backpropagation is
                the mean of the `batches_per_step` batch losses.
                (Default value = 1)
            initial_epoch (int, optional): Epoch at which to start training
                (useful for resuming a previous training run).
                (Default value = 1)
            verbose (bool): Whether to display the progress of the training.
                (Default value = True)
            progress_options (dict, optional): Keyword arguments to pass to the default progression callback used
                in Poutyne (See :class:`~poutyne.ProgressionCallback` for the available arguments).
                (Default value = None)
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called
                during training.
                (Default value = None)
            dataloader_kwargs (dict, optional): Keyword arguments to pass to the PyTorch dataloaders created
                internally. By default, ``shuffle=True`` is passed for the training dataloader but this can be
                overridden by using this argument.
            num_workers (int, optional): how many subprocesses to use for data loading.
                ``0`` means that the data will be loaded in the main process.
                (Default value = 0)
            collate_fn (Callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
                Used when using batched loading from a map-style dataset.

        Returns:
            List of dict containing the history of each epoch.

        See :class:`~torch.utils.data.DataLoader` for details on ``batch_size``, ``num_workers`` and ``collate_fn``.

        Example:
            .. code-block:: python

                model = Model(pytorch_network, optimizer, loss_function)
                history = model.fit(train_dataset,
                                    valid_dataset,
                                    epochs=num_epochs,
                                    batch_size=batch_size,
                                    verbose=False)
                print(*history, sep="\\n")

            .. code-block:: python

                {'epoch': 1, 'loss': 1.7198852968215943, 'time': 0.019999928001197986, 'acc': 19.375, 'val_loss': 1.6674459838867188, 'val_acc': 22.0}
                {'epoch': 2, 'loss': 1.7054892110824584, 'time': 0.015421080999658443, 'acc': 19.75, 'val_loss': 1.660806336402893, 'val_acc': 22.0}
                {'epoch': 3, 'loss': 1.6923445892333984, 'time': 0.01363091799794347, 'acc': 19.625, 'val_loss': 1.6550078630447387, 'val_acc': 22.5}
                ...

        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            **dataloader_kwargs,
        }

        train_generator = DataLoader(train_dataset, **{'shuffle': True, **dataloader_kwargs})
        valid_generator = None
        if valid_dataset is not None:
            valid_generator = DataLoader(valid_dataset, **dataloader_kwargs)

        return self.fit_generator(
            train_generator,
            valid_generator=valid_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            batches_per_step=batches_per_step,
            initial_epoch=initial_epoch,
            verbose=verbose,
            progress_options=progress_options,
            callbacks=callbacks,
        )

    def fit_generator(
        self,
        train_generator,
        valid_generator=None,
        *,
        epochs=1000,
        steps_per_epoch=None,
        validation_steps=None,
        batches_per_step=1,
        initial_epoch=1,
        verbose=True,
        progress_options: Union[dict, None] = None,
        callbacks=None,
    ):
        # pylint: disable=line-too-long
        """
        Trains the network on a dataset using a generator.

        Args:
            train_generator: Generator-like object for the training dataset. The generator must
                yield a batch in the form of a tuple (x, y) where ``x`` is the input and ``y`` is the
                target. The batch size is inferred from ``x`` and ``y``. See :func:`get_batch_size()` for
                details on the inferring algorithm. The loss and the metrics are averaged using this
                batch size. If the batch size cannot be inferred then a warning is raised and the
                "batch size" defaults to 1.

                If the generator does not have a method ``__len__()``, either the ``steps_per_epoch``
                argument must be provided, or the iterator returned raises a StopIteration exception at
                the end of the training dataset. PyTorch DataLoaders object do provide a ``__len__()``
                method.

                Before each epoch, the method ``__iter__()`` on the generator is called and the method
                ``__next__()`` is called for each step on resulting object returned by ``__iter__()``.
                Notice that a call to ``__iter__()`` on a generator made using the python keyword
                ``yield`` returns the generator itself.
            valid_generator (optional): Generator-like object for the validation dataset. This generator
                is optional. The generator is used the same way as the  generator ``train_generator``. If
                the generator does not have a method ``__len__()``, either the ``validation_steps`` or the
                ``steps_per_epoch`` argument must be provided or the iterator returned raises a StopIteration
                exception at the end of the validation dataset.
                (Default value = None)
            epochs (int): Number of times the entire training dataset is seen.
                (Default value = 1000)
            steps_per_epoch (int, optional): Number of batch used during one epoch. Obviously, using this
                argument may cause one epoch not to see the entire training dataset or see it multiple times.
                See argument ``train_generator`` and ``valid_generator`` for more details of how
                ``steps_per_epoch`` is used.
            validation_steps (int, optional): Same as for ``steps_per_epoch`` but for the validation dataset.
                See argument ``valid_generator`` for more details of how ``validation_steps`` is used.
            batches_per_step (int): Number of batches on which to compute the running loss before
                backpropagating it through the network. Note that the total loss used for backpropagation is
                the mean of the `batches_per_step` batch losses.
                (Default value = 1)
            initial_epoch (int, optional): Epoch at which to start training (useful for resuming a previous
                training run).
                (Default value = 1)
            verbose (bool): Whether to display the progress of the training.
                (Default value = True)
            progress_options (dict, optional): Keyword arguments to pass to the default progression callback used
                in Poutyne (See :class:`~poutyne.ProgressionCallback` for the available arguments).
                (Default value = None, meaning default color setting and progress bar)
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called during
                training. (Default value = None)

        Returns:
            List of dict containing the history of each epoch.

        Example:
            .. code-block:: python

                model = Model(pytorch_network, optimizer, loss_function)
                history = model.fit_generator(train_generator,
                                              valid_generator,
                                              epochs=num_epochs,
                                              verbose=False)
                print(*history, sep="\\n")

            .. code-block:: python

                {'epoch': 1, 'loss': 1.7198852968215943, 'time': 0.019999928001197986, 'acc': 19.375, 'val_loss': 1.6674459838867188, 'val_acc': 22.0}
                {'epoch': 2, 'loss': 1.7054892110824584, 'time': 0.015421080999658443, 'acc': 19.75, 'val_loss': 1.660806336402893, 'val_acc': 22.0}
                {'epoch': 3, 'loss': 1.6923445892333984, 'time': 0.01363091799794347, 'acc': 19.625, 'val_loss': 1.6550078630447387, 'val_acc': 22.5}
                ...

        """
        if self.optimizer is None:
            raise ValueError("Impossible to fit when optimizer is None.")

        self._transfer_optimizer_state_to_right_device()

        callbacks = [] if callbacks is None else callbacks

        if verbose:
            progress_options = {} if progress_options is None else progress_options
            callbacks = [ProgressionCallback(**progress_options)] + callbacks
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)

        self.stop_training = False

        epoch_iterator = EpochIterator(
            self,
            train_generator,
            valid_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            initial_epoch=initial_epoch,
            callback=callback_list,
            batch_metrics_names=self.batch_metrics_names,
            epoch_metrics_names=self.epoch_metrics_names,
        )

        if batches_per_step > 1:
            self._fit_generator_n_batches_per_step(epoch_iterator, callback_list, batches_per_step)
        else:
            self._fit_generator_one_batch_per_step(epoch_iterator, callback_list)

        return epoch_iterator.epoch_logs

    def _fit_generator_n_batches_per_step(self, epoch_iterator, callback_list, batches_per_step):
        for train_step_iterator, valid_step_iterator in epoch_iterator:
            examples_in_step = 0

            with self._set_training_mode(True):
                for step, (x, y) in train_step_iterator:
                    step.size = get_batch_size(x, y)

                    examples_in_step += step.size

                    (step.loss, step.batch_metrics, did_backprop, _,) = self._fit_batch_n_batches_per_step(
                        x, y, batches_per_step, examples_in_step, callback=callback_list, step=step
                    )

                    if did_backprop:
                        examples_in_step = 0

            if not did_backprop:
                # Did not step after last batch
                self._adjust_step_size(examples_in_step)
                self.optimizer.step()

            train_step_iterator.loss = self._get_loss()
            train_step_iterator.batch_metrics = self._get_batch_metrics()
            train_step_iterator.epoch_metrics = self._get_epoch_metrics()

            self._run_validation(valid_step_iterator, callback_list)

    def _fit_batch_n_batches_per_step(
        self,
        x,
        y,
        batches_per_step,
        examples_in_step,
        *,
        callback=Callback(),
        step=None,
        return_pred=False,
        convert_to_numpy=True,
    ):
        # pylint: disable=too-many-locals
        zero_all_gradients = (step.number - 1) % batches_per_step == 0
        do_backprop = step.number % batches_per_step == 0

        if zero_all_gradients:
            self.optimizer.zero_grad()

        loss_tensor, batch_metrics, pred_y = self._compute_loss_and_metrics(
            x, y, return_loss_tensor=True, return_pred=return_pred, convert_to_numpy=convert_to_numpy
        )

        adjusted_loss_tensor = loss_tensor * step.size
        adjusted_loss_tensor.backward()

        callback.on_backward_end(step)

        if do_backprop:
            self._adjust_step_size(examples_in_step)
            self.optimizer.step()

        loss = float(loss_tensor)
        return loss, batch_metrics, do_backprop, pred_y

    def _fit_generator_one_batch_per_step(self, epoch_iterator, callback_list):
        for train_step_iterator, valid_step_iterator in epoch_iterator:
            with self._set_training_mode(True):
                for step, (x, y) in train_step_iterator:
                    step.loss, step.batch_metrics, _ = self._fit_batch(x, y, callback=callback_list, step=step.number)
                    step.size = get_batch_size(x, y)

            train_step_iterator.loss = self._get_loss()
            train_step_iterator.batch_metrics = self._get_batch_metrics()
            train_step_iterator.epoch_metrics = self._get_epoch_metrics()

            self._run_validation(valid_step_iterator, callback_list)

    def _fit_batch(self, x, y, *, callback=Callback(), step=None, return_pred=False, convert_to_numpy=True):
        self.optimizer.zero_grad()

        loss_tensor, batch_metrics, pred_y = self._compute_loss_and_metrics(
            x, y, return_loss_tensor=True, return_pred=return_pred, convert_to_numpy=convert_to_numpy
        )

        loss_tensor.backward()
        callback.on_backward_end(step)
        self.optimizer.step()

        loss = float(loss_tensor)
        return loss, batch_metrics, pred_y

    def _run_validation(self, valid_step_iterator, callback_list):
        if valid_step_iterator is not None:
            valid_begin_time = timeit.default_timer()

            callback_list.on_valid_begin({})
            self._validate(valid_step_iterator)

            valid_step_iterator.loss = self._get_loss()
            valid_step_iterator.batch_metrics = self._get_batch_metrics()
            valid_step_iterator.epoch_metrics = self._get_epoch_metrics()

            valid_total_time = timeit.default_timer() - valid_begin_time

            valid_metrics_log = {'time': valid_total_time}
            valid_metrics_log.update(valid_step_iterator.metrics_logs)

            callback_list.on_valid_end(valid_metrics_log)

    def _adjust_step_size(self, examples_in_step):
        for param in self.network.parameters():
            if param.grad is not None:
                param.grad /= examples_in_step

    def _process_input(self, *args):
        args = numpy_to_torch(args)
        if self.device is not None:
            args = torch_to(args, self.device)
        return args[0] if len(args) == 1 else args

    def preprocess_input(self, x, y=None):
        if y is not None:
            x, y = self._process_input(x, y)
        else:
            x = self._process_input(x)

        x = x if isinstance(x, (tuple, list)) else (x,)

        # We return PackedSequence in a tuple since it is a namedtuple, thus an iterator object and
        # would break later when we call self.network(*x) since it will iterate over the PackedSequence named attribute.
        x = (x,) if isinstance(x, PackedSequence) else x

        return (x, y) if y is not None else x

    def train_on_batch(self, x, y, *, return_pred=False, return_dict_format=False, convert_to_numpy=True):
        """
        Trains the network for the batch ``(x, y)`` and computes the loss and the metrics, and
        optionally returns the predictions.

        Args:
            x: Input data as a batch.
            y: Target data as a batch.
            return_pred (bool, optional): Whether to return the predictions.
                (Default value = False)
            return_dict_format (bool, optional): Whether to return the loss and metrics in a dict format or not.
                (Default value = False)
            convert_to_numpy (bool, optional): Whether to convert the predictions into Numpy Arrays when ``return_pred``
                is true. (Default value = True)

        Returns:
            Float ``loss`` if no metrics were specified and ``return_pred`` is false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is the predictions with tensors converted into Numpy
            arrays.

            If ``return_dict_format`` is True, then ``loss, metrics`` are replaced by a
            dictionary.
        """
        if self.optimizer is None:
            raise ValueError("Impossible to fit when optimizer is None.")

        with self._set_training_mode(True):
            self._transfer_optimizer_state_to_right_device()
            loss, batch_metrics, pred_y = self._fit_batch(
                x, y, return_pred=return_pred, convert_to_numpy=convert_to_numpy
            )

        if return_dict_format:
            logs = dict(loss=loss)
            logs.update(zip(self.batch_metrics_names, batch_metrics))

            return self._format_truth_pred_return((logs,), pred_y, return_pred)

        return self._format_loss_metrics_return(loss, batch_metrics, pred_y, return_pred)

    def _format_loss_metrics_return(self, loss, metrics, pred_y, return_pred, true_y=None, return_ground_truth=False):
        # pylint: disable=too-many-arguments
        ret = (loss,)

        ret += tuple(metrics.tolist()) if len(metrics) <= 1 else (metrics,)

        return self._format_truth_pred_return(ret, pred_y, return_pred, true_y, return_ground_truth)

    def _format_truth_pred_return(self, init, pred_y, return_pred, true_y=None, return_ground_truth=False):
        # pylint: disable=too-many-arguments
        if return_pred:
            init += (pred_y,)

        if return_ground_truth:
            init += (true_y,)

        return init[0] if len(init) == 1 else init

    def predict(
        self,
        x,
        *,
        batch_size=32,
        convert_to_numpy=True,
        verbose=True,
        progress_options: Union[dict, None] = None,
        callbacks=None,
        dataloader_kwargs=None,
    ) -> Any:
        """
        Returns the predictions of the network given a dataset ``x``, where the tensors are
        converted into Numpy arrays.

        Args:
            x (Union[~torch.Tensor, ~numpy.ndarray] or Union[tuple, list] of Union[~torch.Tensor, ~numpy.ndarray]):
                Input to the model. Union[Tensor, ndarray] if the model has a single input.
                Union[tuple, list] of Union[Tensor, ndarray] if the model has multiple inputs.
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)
            concatenate_returns (bool, optional): Whether to concatenate the predictions when returning them.
                (Default value = True)
            verbose (bool): Whether to display the progress of the evaluation.
                (Default value = True)
            progress_options (dict, optional): Keyword arguments to pass to the default progression callback used
                in Poutyne (See :class:`~poutyne.ProgressionCallback` for the available arguments).
                (Default value = None, meaning default color setting and progress bar)
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called during
                testing. (Default value = None)
            dataloader_kwargs (dict, optional): Keyword arguments to pass to the PyTorch dataloaders created
                internally.

        Returns:
            Return the predictions in the format outputted by the model.
        """
        x = x if isinstance(x, (tuple, list)) else (x,)
        dataset = self._dataset_from_data(x)
        return self.predict_dataset(
            dataset,
            batch_size=batch_size,
            concatenate_returns=True,
            convert_to_numpy=convert_to_numpy,
            dataloader_kwargs=dataloader_kwargs,
            verbose=verbose,
            progress_options=progress_options,
            callbacks=callbacks,
        )

    def predict_dataset(
        self,
        dataset,
        *,
        batch_size=32,
        steps=None,
        has_ground_truth=False,
        return_ground_truth=False,
        concatenate_returns=True,
        convert_to_numpy=True,
        num_workers=0,
        collate_fn=None,
        verbose=True,
        progress_options: Union[dict, None] = None,
        callbacks=None,
        dataloader_kwargs=None,
    ) -> Any:
        """
        Returns the predictions of the network given a dataset ``x``, where the tensors are
        converted into Numpy arrays.

        Args:
            dataset (~torch.utils.data.Dataset): Dataset. Must not return ``y``, just ``x``, unless
                `has_ground_truth` is true.
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)
            steps (int, optional): Number of iterations done on ``generator``.
                (Defaults the number of steps needed to see the entire dataset)
            has_ground_truth (bool, optional): Whether the generator yields the target ``y``.  Automatically
                set to true if `return_ground_truth` is true. (Default value = False)
            return_ground_truth (bool, optional): Whether to return the ground truths. If true, automatically
                set `has_ground_truth` to true. (Default value = False)
            concatenate_returns (bool, optional): Whether to concatenate the predictions
                or the ground truths when returning them. See :func:`predict_generator()`
                for details. (Default value = True)
            concatenate_returns (bool, optional): Whether to concatenate the predictions
                or the ground truths when returning them. (Default value = True)
            num_workers (int, optional): how many subprocesses to use for data loading.
                ``0`` means that the data will be loaded in the main process.
                (Default value = 0)
            collate_fn (Callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
                Used when using batched loading from a map-style dataset.
            verbose (bool): Whether to display the progress of the evaluation.
                (Default value = True)
            progress_options (dict, optional): Keyword arguments to pass to the default progression callback used
                in Poutyne (See :class:`~poutyne.ProgressionCallback` for the available arguments).
                (Default value = None, meaning default color setting and progress bar)
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called during
                testing. (Default value = None)
            dataloader_kwargs (dict, optional): Keyword arguments to pass to the PyTorch dataloaders created
                internally.

        Returns:
            Depends on the value of ``concatenate_returns``. By default, (``concatenate_returns`` is true),
            the data structures (tensor, tuple, list, dict) returned as predictions for the batches are
            merged together. In the merge, the tensors are converted into Numpy arrays and are then
            concatenated together. If ``concatenate_returns`` is false, then a list of the predictions
            for the batches is returned with tensors converted into Numpy arrays.

        See:
            :class:`~torch.utils.data.DataLoader` for details on ``batch_size``, ``num_workers`` and ``collate_fn``.
        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            **dataloader_kwargs,
        }

        generator = DataLoader(dataset, **dataloader_kwargs)
        return self.predict_generator(
            generator,
            steps=steps,
            has_ground_truth=has_ground_truth,
            return_ground_truth=return_ground_truth,
            concatenate_returns=concatenate_returns,
            convert_to_numpy=convert_to_numpy,
            verbose=verbose,
            progress_options=progress_options,
            callbacks=callbacks,
        )

    def predict_generator(
        self,
        generator,
        *,
        steps=None,
        has_ground_truth=False,
        return_ground_truth=False,
        concatenate_returns=True,
        convert_to_numpy=True,
        verbose=True,
        progress_options: Union[dict, None] = None,
        callbacks=None,
    ) -> Any:
        """
        Returns the predictions of the network given batches of samples ``x``, where the tensors are
        converted into Numpy arrays.

        Args:
            generator: Generator-like object for the dataset. The generator must yield a batch of
                samples. See the :func:`fit_generator()` method for details on the types of generators
                supported. This should only yield input data ``x`` and NOT the target ``y``, unless
                `has_ground_truth` is true.
            steps (int, optional): Number of iterations done on ``generator``.
                (Defaults the number of steps needed to see the entire dataset)
            has_ground_truth (bool, optional): Whether the generator yields the target ``y``.  Automatically
                set to true if `return_ground_truth` is true. (Default value = False)
            return_ground_truth (bool, optional): Whether to return the ground truths. If true, automatically
                set `has_ground_truth` to true. (Default value = False)
            concatenate_returns (bool, optional): Whether to concatenate the predictions
                or the ground truths when returning them. (Default value = True)
            convert_to_numpy (bool, optional): Whether to convert the predictions or ground truths into Numpy Arrays.
                (Default value = True)
            verbose (bool): Whether to display the progress of the evaluation.
                (Default value = True)
            progress_options (dict, optional): Keyword arguments to pass to the default progression callback used
                in Poutyne (See :class:`~poutyne.ProgressionCallback` for the available arguments).
                (Default value = None, meaning default color setting and progress bar)
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called during
                testing. (Default value = None)

        Returns:
            Depends on the value of ``concatenate_returns``. By default, (``concatenate_returns`` is true),
            the data structures (tensor, tuple, list, dict) returned as predictions for the batches are
            merged together. In the merge, the tensors are converted into Numpy arrays and are then
            concatenated together. If ``concatenate_returns`` is false, then a list of the predictions
            for the batches is returned with tensors converted into Numpy arrays.
        """
        # pylint: disable=too-many-locals
        has_ground_truth = has_ground_truth or return_ground_truth

        if steps is None and hasattr(generator, '__len__'):
            steps = len(generator)
        pred_y = []
        if return_ground_truth:
            true_y = []

        callbacks = [] if callbacks is None else callbacks

        if verbose:
            progress_options = {} if progress_options is None else progress_options
            callbacks = [ProgressionCallback(**progress_options)] + callbacks
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)
        callback_list.set_params({'steps': steps})

        predict_begin_time = timeit.default_timer()
        with self._set_training_mode(False):
            callback_list.on_predict_begin({})
            time_since_last_batch = timeit.default_timer()
            for step, batch in _get_step_iterator(steps, generator):
                callback_list.on_predict_batch_begin(step, {})

                if has_ground_truth:
                    x, y = self.preprocess_input(*batch)
                else:
                    x = self.preprocess_input(batch)

                batch_pred = self.network(*x)
                pred_y.append(torch_to_numpy(batch_pred) if convert_to_numpy else batch_pred)
                if return_ground_truth:
                    true_y.append(torch_to_numpy(y) if convert_to_numpy else y)

                batch_end_time = timeit.default_timer()
                batch_total_time = batch_end_time - time_since_last_batch
                time_since_last_batch = batch_end_time

                callback_list.on_predict_batch_end(step, {'batch': step, 'time': batch_total_time})

        if concatenate_returns:
            pred_y = _concat(pred_y)
            if return_ground_truth:
                true_y = _concat(true_y)

        callback_list.on_predict_end({'time': timeit.default_timer() - predict_begin_time})

        if return_ground_truth:
            return pred_y, true_y
        return pred_y

    def predict_on_batch(self, x, *, convert_to_numpy=True) -> Any:
        """
        Returns the predictions of the network given a batch ``x``, where the tensors are converted
        into Numpy arrays.

        Args:
            x: Input data as a batch.
            convert_to_numpy (bool, optional): Whether to convert the predictions into Numpy Arrays.
                (Default value = True)

        Returns:
            Return the predictions in the format outputted by the model.
        """
        with self._set_training_mode(False):
            x = self.preprocess_input(x)
            y_pred = self.network(*x)
            return torch_to_numpy(y_pred) if convert_to_numpy else y_pred

    def evaluate(
        self,
        x,
        y,
        *,
        batch_size=32,
        return_pred=False,
        return_dict_format=False,
        convert_to_numpy=True,
        callbacks=None,
        verbose=True,
        progress_options: Union[dict, None] = None,
        dataloader_kwargs=None,
    ) -> Tuple:
        """
        Computes the loss and the metrics of the network on batches of samples and optionally
        returns the predictions.

        Args:
            x (Union[~torch.Tensor, ~numpy.ndarray] or Union[tuple, list] of Union[~torch.Tensor, ~numpy.ndarray]):
                Input to the model. Union[Tensor, ndarray] if the model has a single input.
                Union[tuple, list] of Union[Tensor, ndarray] if the model has multiple inputs.
            y (Union[~torch.Tensor, ~numpy.ndarray] or Union[tuple, list] of Union[~torch.Tensor, ~numpy.ndarray]):
                Target, corresponding ground truth.
                Union[Tensor, ndarray] if the model has a single output.
                Union[tuple, list] of Union[Tensor, ndarray] if the model has multiple outputs.
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)
            return_pred (bool, optional): Whether to return the predictions.
                (Default value = False)
            return_dict_format (bool, optional): Whether to return the loss and metrics in a dict format or not.
                (Default value = False)
            convert_to_numpy (bool, optional): Whether to convert the predictions into Numpy Arrays when ``return_pred``
                is true. (Default value = True)
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called during
                testing. (Default value = None)
            verbose (bool): Whether to display the progress of the evaluation.
                (Default value = True)
            progress_options (dict, optional): Keyword arguments to pass to the default progression callback used
                in Poutyne (See :class:`~poutyne.ProgressionCallback` for the available arguments).
                (Default value = None, meaning default color setting and progress bar)
            dataloader_kwargs (dict, optional): Keyword arguments to pass to the PyTorch dataloaders created
                internally.

        Returns:
            Tuple ``(loss, metrics, pred_y)`` where specific elements are omitted if not
            applicable. If only loss is applicable, then it is returned as a float.

            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of batch metrics plus the number of epoch metrics if ``n > 1``. If
            ``n == 1``, then ``metrics`` is a float. If ``n == 0``, the ``metrics`` is
            omitted. The first elements of ``metrics`` are the batch metrics and are
            followed by the epoch metrics. See the :func:`~Model.fit_generator()` method
            for examples with batch metrics and epoch metrics.

            If ``return_pred`` is True, ``pred_y`` is the list of the predictions
            of each batch with tensors converted into Numpy arrays. It is otherwise omitted.

            If ``return_dict_format`` is True, then ``loss, metrics`` are replaced by a
            dictionary as passed to :func:`~poutyne.Callback.on_test_end()`.

        """
        dataset = self._dataset_from_data((x, y))
        return self.evaluate_dataset(
            dataset,
            batch_size=batch_size,
            return_pred=return_pred,
            return_dict_format=return_dict_format,
            concatenate_returns=True,
            convert_to_numpy=convert_to_numpy,
            callbacks=callbacks,
            verbose=verbose,
            progress_options=progress_options,
            dataloader_kwargs=dataloader_kwargs,
        )

    def evaluate_dataset(
        self,
        dataset,
        *,
        batch_size=32,
        steps=None,
        return_pred=False,
        return_ground_truth=False,
        return_dict_format=False,
        concatenate_returns=True,
        convert_to_numpy=True,
        callbacks=None,
        num_workers=0,
        collate_fn=None,
        dataloader_kwargs=None,
        verbose=True,
        progress_options: Union[dict, None] = None,
    ) -> Tuple:
        # pylint: disable=too-many-locals
        """
        Computes the loss and the metrics of the network on batches of samples and optionally
        returns the predictions.

        Args:
            dataset (~torch.utils.data.Dataset): Dataset.
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)
            steps (int, optional): Number of batches used for evaluation.
                (Defaults the number of steps needed to see the entire dataset)
            return_pred (bool, optional): Whether to return the predictions.
                (Default value = False)
            return_ground_truth (bool, optional): Whether to return the ground truths.
                (Default value = False)
            return_dict_format (bool, optional): Whether to return the loss and metrics in a dict format or not.
                (Default value = False)
            concatenate_returns (bool, optional): Whether to concatenate the predictions
                or the ground truths when returning them. (Default value = True)
            convert_to_numpy (bool, optional): Whether to convert the predictions or ground truths into Numpy Arrays
                when ``return_pred`` or ``return_ground_truth`` are true. (Default value = True)
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called during
                testing. (Default value = None)
            num_workers (int, optional): how many subprocesses to use for data loading.
                ``0`` means that the data will be loaded in the main process.
                (Default value = 0)
            collate_fn (Callable, optional): merges a list of samples to form a mini-batch of Tensor(s).
                Used when using batched loading from a map-style dataset.
            dataloader_kwargs (dict, optional): Keyword arguments to pass to the PyTorch dataloaders created
                internally.
            verbose (bool): Whether to display the progress of the evaluation.
                (Default value = True)
            progress_options (dict, optional): Keyword arguments to pass to the default progression callback used
                in Poutyne (See :class:`~poutyne.ProgressionCallback` for the available arguments).
                (Default value = None, meaning default color setting and progress bar)

        Returns:
            Tuple ``(loss, metrics, pred_y)`` where specific elements are omitted if not
            applicable. If only loss is applicable, then it is returned as a float.

            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of batch metrics plus the number of epoch metrics if ``n > 1``. If
            ``n == 1``, then ``metrics`` is a float. If ``n == 0``, the ``metrics`` is
            omitted. The first elements of ``metrics`` are the batch metrics and are
            followed by the epoch metrics. See the :func:`~Model.fit_generator()` method
            for examples with batch metrics and epoch metrics.

            If ``return_pred`` is True, ``pred_y`` is the list of the predictions
            of each batch with tensors converted into Numpy arrays. It is otherwise omitted.

            If ``return_dict_format`` is True, then ``loss, metrics`` are replaced by a
            dictionary as passed to :func:`~poutyne.Callback.on_test_end()`.

        See:
            :class:`~torch.utils.data.DataLoader` for details on ``batch_size``, ``num_workers`` and ``collate_fn``.
        """
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        dataloader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            **dataloader_kwargs,
        }

        generator = DataLoader(dataset, **dataloader_kwargs)
        return self.evaluate_generator(
            generator,
            steps=steps,
            return_pred=return_pred,
            return_ground_truth=return_ground_truth,
            return_dict_format=return_dict_format,
            concatenate_returns=concatenate_returns,
            convert_to_numpy=convert_to_numpy,
            callbacks=callbacks,
            verbose=verbose,
            progress_options=progress_options,
        )

    def evaluate_generator(
        self,
        generator,
        *,
        steps=None,
        return_pred=False,
        return_ground_truth=False,
        return_dict_format=False,
        concatenate_returns=True,
        convert_to_numpy=True,
        verbose=True,
        progress_options: Union[dict, None] = None,
        callbacks=None,
    ) -> Tuple:
        # pylint: disable=too-many-locals
        """
        Computes the loss and the metrics of the network on batches of samples and optionally returns
        the predictions.

        Args:
            generator: Generator-like object for the dataset. See the :func:`~Model.fit_generator()` method for
                details on the types of generators supported.
            steps (int, optional): Number of iterations done on ``generator``.
                (Defaults the number of steps needed to see the entire dataset)
            return_pred (bool, optional): Whether to return the predictions.
                (Default value = False)
            return_ground_truth (bool, optional): Whether to return the ground truths.
                (Default value = False)
            return_dict_format (bool, optional): Whether to return the loss and metrics in a dict format or not.
                (Default value = False)
            convert_to_numpy (bool, optional): Whether to convert the predictions or ground truths into Numpy Arrays
                when ``return_pred`` or ``return_ground_truth`` are true. (Default value = True)
            concatenate_returns (bool, optional): Whether to concatenate the predictions
                or the ground truths when returning them. (Default value = True)
            verbose (bool): Whether to display the progress of the evaluation.
                (Default value = True)
            progress_options (dict, optional): Keyword arguments to pass to the default progression callback used
                in Poutyne (See :class:`~poutyne.ProgressionCallback` for the available arguments).
                (Default value = None, meaning default color setting and progress bar)
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called during
                testing. (Default value = None)

        Returns:
            Tuple ``(loss, metrics, pred_y, true_y)`` where specific elements are
            omitted if not applicable. If only loss is applicable, then it is returned
            as a float.

            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of batch metrics plus the number of epoch metrics if ``n > 1``. If
            ``n == 1``, then ``metrics`` is a float. If ``n == 0``, the ``metrics`` is
            omitted. The first elements of ``metrics`` are the batch metrics and are
            followed by the epoch metrics.

            If ``return_pred`` is True, ``pred_y`` is the predictions returned as in
            the :func:`predict_generator()` method. It is otherwise ommited.

            If ``return_ground_truth`` is True, ``true_y`` is the ground truths returned
            as in the :func:`predict_generator()` method. It is otherwise omitted.

            If ``return_dict_format`` is True, then ``loss, metrics`` are replaced by a
            dictionary as passed to :func:`~poutyne.Callback.on_test_end()`.

        Example:
            With no metrics:

            .. code-block:: python

                model = Model(pytorch_network, optimizer, loss_function,
                              batch_metrics=None)
                loss = model.evaluate_generator(test_generator)

            With only one batch metric:

            .. code-block:: python

                model = Model(pytorch_network, optimizer, loss_function,
                              batch_metrics=[my_metric_fn])
                loss, my_metric = model.evaluate_generator(test_generator)

            With several batch metrics:

            .. code-block:: python

                model = Model(pytorch_network, optimizer, loss_function,
                              batch_metrics=[my_metric1_fn, my_metric2_fn])
                loss, (my_metric1, my_metric2) = model.evaluate_generator(test_generator)

            With one batch metric and one epoch metric:

            .. code-block:: python

                model = Model(pytorch_network, optimizer, loss_function,
                              batch_metrics=[my_metric_fn], epoch_metrics=[MyMetricClass()])
                loss, (my_batch_metric, my__epoch_metric) = model.evaluate_generator(test_generator)

            With batch metrics and ``return_pred`` flag:

            .. code-block:: python

                model = Model(pytorch_network, optimizer, loss_function,
                              batch_metrics=[my_metric1_fn, my_metric2_fn])
                loss, (my_metric1, my_metric2), pred_y = model.evaluate_generator(
                    test_generator, return_pred=True
                )

            With batch metrics, ``return_pred`` and ``return_ground_truth`` flags:

            .. code-block:: python

                model = Model(pytorch_network, optimizer, loss_function,
                              batch_metrics=[my_metric1_fn, my_metric2_fn])
                loss, (my_metric1, my_metric2), pred_y, true_y = model.evaluate_generator(
                    test_generator, return_pred=True, return_ground_truth=True
                )

            With ``return_dict_format``:

            .. code-block:: python

                model = Model(pytorch_network, optimizer, loss_function,
                              batch_metrics=[my_metric_fn])
                logs = model.evaluate_generator(test_generator, return_dict_format=True)
        """
        callbacks = [] if callbacks is None else callbacks

        if verbose:
            progress_options = {} if progress_options is None else progress_options
            callbacks = [ProgressionCallback(**progress_options)] + callbacks

        if steps is None and hasattr(generator, '__len__'):
            steps = len(generator)

        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)

        callback_list.set_params({'steps': steps})
        callback_list.on_test_begin({})

        step_iterator = StepIterator(
            generator, steps, self.batch_metrics_names, self.epoch_metrics_names, callback_list, mode="test"
        )

        test_begin_time = timeit.default_timer()
        pred_y, true_y = self._validate(
            step_iterator,
            return_pred=return_pred,
            return_ground_truth=return_ground_truth,
            convert_to_numpy=convert_to_numpy,
        )

        step_iterator.loss = self._get_loss()
        step_iterator.batch_metrics = self._get_batch_metrics()
        step_iterator.epoch_metrics = self._get_epoch_metrics()
        test_total_time = timeit.default_timer() - test_begin_time

        if return_pred and concatenate_returns:
            pred_y = _concat(pred_y)
        if return_ground_truth and concatenate_returns:
            true_y = _concat(true_y)

        test_metrics_log = {'time': test_total_time}
        test_metrics_log.update(step_iterator.metrics_logs)

        callback_list.on_test_end(test_metrics_log)

        if return_dict_format:
            return self._format_truth_pred_return((test_metrics_log,), pred_y, return_pred, true_y, return_ground_truth)

        metrics = np.concatenate((step_iterator.batch_metrics, step_iterator.epoch_metrics))
        return self._format_loss_metrics_return(
            step_iterator.loss, metrics, pred_y, return_pred, true_y, return_ground_truth
        )

    def evaluate_on_batch(self, x, y, *, return_pred=False, return_dict_format=False, convert_to_numpy=True) -> Tuple:
        """
        Computes the loss and the metrics of the network on a single batch of samples and optionally
        returns the predictions.

        Args:
            x: Input data as a batch.
            y: Target data as a batch.
            return_pred (bool, optional): Whether to return the predictions for ``batch``.
                (Default value = False)
            return_dict_format (bool, optional): Whether to return the loss and metrics in a dict format or not.
                (Default value = False)
            convert_to_numpy (bool, optional): Whether to convert the predictions into Numpy Arrays when ``return_pred``
                is true. (Default value = True)

        Returns:
            Tuple ``(loss, metrics, pred_y)`` where specific elements are omitted if not
            applicable. If only loss is applicable, then it is returned as a float.

            `metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            If ``return_pred`` is True, ``pred_y`` is the list of the predictions
            of each batch with tensors converted into Numpy arrays. It is otherwise omitted.

            If ``return_dict_format`` is True, then ``loss, metrics`` are replaced by a
            dictionary.
        """
        with self._set_training_mode(False):
            loss, batch_metrics, pred_y = self._compute_loss_and_metrics(
                x, y, return_pred=return_pred, convert_to_numpy=convert_to_numpy
            )

        if return_dict_format:
            logs = dict(loss=loss)
            logs.update(zip(self.batch_metrics_names, batch_metrics))

            return self._format_truth_pred_return((logs,), pred_y, return_pred)

        return self._format_loss_metrics_return(loss, batch_metrics, pred_y, return_pred)

    def _validate(self, step_iterator, return_pred=False, return_ground_truth=False, convert_to_numpy=True):
        pred_list = None
        true_list = None
        if return_pred:
            pred_list = []
        if return_ground_truth:
            true_list = []

        with self._set_training_mode(False):
            for step, (x, y) in step_iterator:
                step.loss, step.batch_metrics, pred_y = self._compute_loss_and_metrics(
                    x, y, return_pred=return_pred, convert_to_numpy=convert_to_numpy
                )
                if return_pred:
                    pred_list.append(pred_y)
                if return_ground_truth:
                    true_list.append(torch_to_numpy(y) if convert_to_numpy else y)

                step.size = get_batch_size(x, y)

        return pred_list, true_list

    def _compute_loss_and_metrics(self, x, y, *, return_loss_tensor=False, return_pred=False, convert_to_numpy=True):
        x, y = self.preprocess_input(x, y)
        if self.other_device is not None:
            pred_y = torch.nn.parallel.data_parallel(self.network, x, [self.device] + self.other_device)
        else:
            pred_y = self.network(*x)
        loss = self.loss_function(pred_y, y)
        if not return_loss_tensor:
            loss = float(loss)
        with torch.no_grad():
            batch_metrics = self._compute_batch_metrics(pred_y, y)
            for epoch_metric in self.epoch_metrics:
                epoch_metric.update(pred_y, y)

        if return_pred:
            pred_y = torch_to_numpy(pred_y) if convert_to_numpy else pred_y
        else:
            pred_y = None

        return loss, batch_metrics, pred_y

    def _compute_batch_metrics(self, pred_y, y):
        batch_metrics = [metric(pred_y, y) for metric in self.batch_metrics]
        return self._compute_metric_array(batch_metrics, self.original_batch_metrics_names)

    def _get_loss(self):
        loss = self.loss_function.compute().item()
        self.loss_function.reset()
        return loss

    def _get_batch_metrics(self):
        metrics = [batch_metric.compute() for batch_metric in self.batch_metrics]
        for batch_metric in self.batch_metrics:
            batch_metric.reset()
        return self._compute_metric_array(metrics, self.original_batch_metrics_names)

    def _get_epoch_metrics(self):
        metrics = [epoch_metric.compute() for epoch_metric in self.epoch_metrics]
        for epoch_metric in self.epoch_metrics:
            epoch_metric.reset()
        return self._compute_metric_array(metrics, self.original_epoch_metrics_names)

    def _compute_metric_array(self, metrics_list, names_list):
        def _get_metric(names, metrics):
            names = [names] if isinstance(names, str) else names
            values = None
            if (torch.is_tensor(metrics) or isinstance(metrics, np.ndarray)) and len(metrics.shape) == 0:
                values = [float(metrics)]
            elif isinstance(metrics, Mapping):
                values = [float(metrics[name]) for name in names]
            elif isinstance(metrics, Iterable):
                values = [float(metric) for metric in metrics]
            else:
                values = [float(metrics)]
            return values

        return np.array(
            [metric for names, metrics in zip(names_list, metrics_list) for metric in _get_metric(names, metrics)]
        )

    def load_weights(self, f, strict=True):
        """
        Loads the weights saved using the :func:`torch.save()` method or the :func:`save_weights()` method
        of this class. Contrary to :func:`torch.load()`, the weights are not transferred to the device
        from which they were saved from. In other words, the PyTorch module will stay on the same
        device it already is on.

        Args:
            f: File-like object (has to implement fileno that returns a file descriptor) or string
                containing a file name.


        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        return self.set_weights(torch.load(f, map_location='cpu'), strict=strict)

    def save_weights(self, f):
        """
        Saves the weights of the current network.

        Args:
            f: File-like object (has to implement fileno that returns a file descriptor) or string
                containing a file name.
        """
        torch.save(self.network.state_dict(), f)

    def load_optimizer_state(self, f):
        """
        Loads the optimizer state saved using the :func:`torch.save()` method or the
        :func:`save_optimizer_state()` method of this class.

        Args:
            f: File-like object (has to implement fileno that returns a file descriptor) or string
                containing a file name.
        """
        self.optimizer.load_state_dict(torch.load(f, map_location='cpu'))

    def save_optimizer_state(self, f):
        """
        Saves the state of the current optimizer.

        Args:
            f: File-like object (has to implement fileno that returns a file descriptor) or string
                containing a file name.
        """
        torch.save(self.optimizer.state_dict(), f)

    def _transfer_optimizer_state_to_right_device(self):
        if self.optimizer is None:
            return

        # Since the optimizer state is loaded on CPU, it will crash when the optimizer will receive
        # gradient for parameters not on CPU. Thus, for each parameter, we transfer its state in the
        # optimizer on the same device as the parameter itself just before starting the
        # optimization.
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state:
                    for n, v in self.optimizer.state[p].items():
                        if (
                            ('capturable' not in group or group["capturable"] or n != 'step')
                            and torch.is_tensor(v)
                            and p.device != v.device
                        ):
                            v.data = v.data.to(p.device)

    def _get_named_optimizer_attrs(self):
        param_to_name = {param: name for name, param in self.network.named_parameters()}

        param_name_groups = []
        for group in self.optimizer.param_groups:
            param_name_groups.append([param_to_name[param] for param in group['params']])

        named_state = {param_to_name[param]: state for param, state in self.optimizer.state.items()}

        return param_name_groups, named_state

    def _set_named_optimizer_attrs(self, param_name_groups, named_state):
        name_to_param = dict(self.network.named_parameters())

        for param_name_group, optim_group in zip(param_name_groups, self.optimizer.param_groups):
            optim_group['params'] = [
                name_to_param[param_name] if optim_param is not name_to_param[param_name] else optim_param
                for param_name, optim_param in zip(param_name_group, optim_group['params'])
            ]

        self.optimizer.state = defaultdict(dict, {name_to_param[name]: state for name, state in named_state.items()})

    @contextlib.contextmanager
    def _update_optim_device(self):
        if self.optimizer is None:
            yield
            return

        param_name_groups, named_state = self._get_named_optimizer_attrs()
        try:
            yield
        finally:
            self._set_named_optimizer_attrs(param_name_groups, named_state)
            self._transfer_optimizer_state_to_right_device()

    def get_weights(self):
        """
        Returns a dictionary containing the parameters of the network. The tensors are just
        references to the parameters. To get copies of the weights, see the :func:`get_weight_copies()`
        method.
        """
        return self.network.state_dict()

    def get_weight_copies(self):
        """
        Returns a dictionary containing copies of the parameters of the network.
        """
        weights = self.get_weights()
        for k in weights.keys():
            weights[k] = weights[k].cpu().clone()
        return weights

    def set_weights(self, weights, strict=True):
        """
        Modifies the weights of the network with the given weights.

        Args:
            weights (dict): Weights returned by either :func:`get_weights()` or :func:`get_weight_copies()`.

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
        """
        return self.network.load_state_dict(weights, strict=strict)

    def cuda(self, *args, **kwargs):
        """
        Transfers the network on the GPU. The arguments are passed to the :meth:`torch.nn.Module.cuda()` method.
        Notice that the device is saved so that the batches can send to the right device before passing it to
        the network.

        Note:
            PyTorch optimizers assume that the parameters have been transferred to the right device
            before their creations. Furthermore, future versions of PyTorch will no longer modify
            the parameters of a PyTorch module in-place when transferring them to another device.
            See this `issue <https://github.com/pytorch/pytorch/issues/7844>`_ and this
            `pull request <https://github.com/pytorch/pytorch/pull/21613>`_ for details.

            Since Poutyne supposes that the optimizer has been initialized before the Poutyne Model,
            necessarily the parameters are not guaranteed to be in sync with those contained in the
            optimizer once the PyTorch module is transferred to another device. Thus, this method
            takes care of this inconsistency by updating the parameters inside the optimizer.

        Returns:
            `self`.
        """
        with self._update_optim_device():
            self.network.cuda(*args, **kwargs)

        # Assuming the PyTorch module has at least one parameter.
        self.device = next(self.network.parameters()).device
        self.other_device = None

        self._transfer_loss_and_metrics_modules_to_right_device()

        return self

    def cpu(self, *args, **kwargs):
        """
        Transfers the network on the CPU. The arguments are passed to the :meth:`torch.nn.Module.cpu()`
        method. Notice that the device is saved so that the batches can send to the right device
        before passing it to the network.

        Note:
            PyTorch optimizers assume that the parameters have been transferred to the right device
            before their creations. Furthermore, future versions of PyTorch will no longer modify
            the parameters of a PyTorch module in-place when transferring them to another device.
            See this `issue <https://github.com/pytorch/pytorch/issues/7844>`_ and this
            `pull request <https://github.com/pytorch/pytorch/pull/21613>`_ for details.

            Since Poutyne supposes that the optimizer has been initialized before the Poutyne Model,
            necessarily the parameters are not guaranteed to be in sync with those contained in the
            optimizer once the PyTorch module is transferred to another device. Thus, this method
            takes care of this inconsistency by updating the parameters inside the optimizer.

        Returns:
            `self`.
        """
        with self._update_optim_device():
            self.network.cpu(*args, **kwargs)

        # Assuming the PyTorch module has at least one parameter.
        self.device = next(self.network.parameters()).device
        self.other_device = None

        self._transfer_loss_and_metrics_modules_to_right_device()

        return self

    def to(self, device):
        """
        Transfer the network on the specified device. The device is saved so that the batches can
        send to the right device before passing it to the network. One could also use multi GPUs by
        using either a list of devices or "all" to take all the available devices. In both cases,
        the training loop will use the `~torch.nn.parallel.data_parallel()` function for single
        node multi GPUs parallel process and the main device is the first device.

        Note:
            PyTorch optimizers assume that the parameters have been transferred to the right device
            before their creations. Furthermore, future versions of PyTorch will no longer modify
            the parameters of a PyTorch module in-place when transferring them to another device.
            See this `issue <https://github.com/pytorch/pytorch/issues/7844>`_ and this
            `pull request <https://github.com/pytorch/pytorch/pull/21613>`_ for details.

            Since Poutyne supposes that the optimizer has been initialized before the Poutyne Model,
            necessarily the parameters are not guaranteed to be in sync with those contained in the
            optimizer once the PyTorch module is transferred to another device. Thus, this method
            takes care of this inconsistency by updating the parameters inside the optimizer.

        Args:
            device (Union[torch.torch.device, List[torch.torch.device]]): The device to which the network is sent or
            the list of device to which the network is sent.

        Returns:
            `self`.
        """
        self.other_device = None
        if isinstance(device, List) or device == "all":
            if device == "all":
                device = [f"cuda:{device}" for device in range(torch.cuda.device_count())]
            self.device = device[0]
            if len(device) > 1:  # case where we use all when having only one GPU or using a list of one device
                self.other_device = device[1:]
        else:
            self.device = device

        with self._update_optim_device():
            self.network.to(self.device)
        self._transfer_loss_and_metrics_modules_to_right_device()
        return self

    def _transfer_loss_and_metrics_modules_to_right_device(self):
        if isinstance(self.loss_function, torch.nn.Module):
            self.loss_function.to(self.device)

        for metric in self.batch_metrics + self.epoch_metrics:
            if isinstance(metric, torch.nn.Module):
                metric.to(self.device)

        return self
