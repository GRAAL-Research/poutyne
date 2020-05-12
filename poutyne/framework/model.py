# pylint: disable=too-many-lines
import contextlib
import warnings
from collections import defaultdict
from typing import Iterable, Mapping
import numbers

import numpy as np
import torch
from torch.utils.data import DataLoader

from poutyne import torch_to_numpy, numpy_to_torch, torch_to
from poutyne.framework.metrics import get_epoch_metric
from poutyne.utils import TensorDataset
from .callbacks import CallbackList, ProgressionCallback, Callback
from .iterators import EpochIterator, _get_step_iterator, StepIterator
from .metrics import get_loss_or_metric, get_callables_and_names, rename_doubles, flatten_metric_names
from .optimizers import get_optimizer
from .warning_manager import warning_settings
from ..utils import _concat


class Model:
    """
    The Model class encapsulates a PyTorch network, a PyTorch optimizer, a loss function and
    metric functions. It allows the user to train a neural network without hand-coding the
    epoch/step logic.

    Args:
        network (torch.nn.Module): A PyTorch network.
        optimizer (Union[torch.optim.Optimizer, str]): If torch.optim.Optimier, an initialized PyTorch.
            If str, should be the optimizer's name in Pytorch (i.e. 'Adam' for torch.optim.Adam).
            (Default value = 'sgd')
        loss_function(Union[Callable, str]) It can be any PyTorch loss layer or custom loss function. It
            can also be a string with the same name as a PyTorch loss function (either the functional or
            object name). The loss function must have the signature ``loss_function(input, target)`` where
            ``input`` is the prediction of the network and ``target`` is the ground truth.
            (Default value = None)
        batch_metrics (list): List of functions with the same signature as the loss function. Each metric
            can be any PyTorch loss function. It can also be a string with the same name as a PyTorch
            loss function (either the functional or object name). 'accuracy' (or just 'acc') is also a
            valid metric. Each metric function is called on each batch of the optimization and on the
            validation batches at the end of the epoch.
            (Default value = None)
        epoch_metrics (list): List of functions with the same signature as
            :class:`~poutyne.framework.metrics.epoch_metrics.EpochMetric`
            (Default value = None)

    Note:
        The name of each batch and epoch metric can be change by passing a tuple ``(name, metric)`` instead
        of simply the metric function or object, where ``name`` is the alternative name of the metric.

        Batch and epoch metrics can return multiple metrics (e.g. an epoch metric could return an F1-score
        with the associated precision and recall). The metrics can returned via an iterable (tuple, list,
        Numpy arrays, tensors, etc.) or via a mapping (e.g. a dict). However, in this case, the names of
        the different metric has to be passed in some way. There are two ways to do so. The easiest one
        is to pass the metric as a tuple ``(names, metric)`` where ``names`` is a tuple containing a name for
        each metric returned. Another way is to override the attribute ``__name__`` of the function or object
        so that it returns a tuple containing a name for all metrics returned. Note that, when the metric
        returns a mapping, the names of the different metrics must be keys in the mapping.

        Example:

        .. code-block:: python

            # Example with custom batch metrics
            my_custom_metric = lambda input, target: 42.
            my_custom_metric2 = lambda input, target: torch.tensor([42., 43.])
            my_custom_metric3 = lambda input, target: {'a': 42., 'b': 43.}
            batch_metrics = [('custom_name', my_custom_metric),
                             (('metric_1', 'metric_2'), my_custom_metric2),
                             (('a', 'b'), my_custom_metric3)]

    Attributes:
        network (torch.nn.Module): The associated PyTorch network.
        optimizer (torch.optim.Optimizer): The associated PyTorch optimizer.
        loss_function: The associated loss function.
        batch_metrics (list): The associated metric functions for every batch.
        epoch_metrics (list): The associated metric functions for every epoch.

    Example:
        Using Numpy arrays (or tensors) dataset::

            from poutyne.framework import Model
            import torch
            import numpy as np

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
            model = Model(pytorch_network, 'sgd', 'cross_entropy', batch_metrics=['accuracy'])
            model.fit(train_x, train_y,
                      validation_data=(valid_x, valid_y),
                      epochs=5,
                      batch_size=32)

        .. code-block:: none

            Epoch 1/5 0.02s Step 25/25: loss: 1.719885, acc: 19.375000, val_loss: 1.667446, val_acc: 22.000000
            Epoch 2/5 0.02s Step 25/25: loss: 1.705489, acc: 19.750000, val_loss: 1.660806, val_acc: 22.000000
            Epoch 3/5 0.01s Step 25/25: loss: 1.692345, acc: 19.625000, val_loss: 1.655008, val_acc: 22.500000
            ...

        Using PyTorch DataLoader::

           import torch
           from torch.utils.data import DataLoader, TensorDataset
           from poutyne.framework import Model

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

           pytorch_network = torch.nn.Linear(num_features, num_train_samples)

           model = Model(pytorch_network, 'sgd', 'cross_entropy', batch_metrics=['accuracy'])
           model.fit_generator(train_generator,
                               valid_generator,
                               epochs=5)

        .. code-block:: none

            Epoch 1/5 0.05s Step 25/25: loss: 6.752676, acc: 0.000000, val_loss: 6.575071, val_acc: 0.000000
            Epoch 2/5 0.03s Step 25/25: loss: 6.454859, acc: 0.125000, val_loss: 6.279577, val_acc: 0.000000
            Epoch 3/5 0.03s Step 25/25: loss: 6.158523, acc: 2.125000, val_loss: 5.985811, val_acc: 9.500000
            ...

    """

    def __init__(self, network, optimizer, loss_function, *, batch_metrics=None, epoch_metrics=None):
        batch_metrics = [] if batch_metrics is None else batch_metrics
        epoch_metrics = [] if epoch_metrics is None else epoch_metrics

        self.network = network
        self.optimizer = get_optimizer(optimizer, self.network)
        self.loss_function = get_loss_or_metric(loss_function)

        self._set_metrics_attributes(batch_metrics, epoch_metrics)

        self.device = None

    def _set_metrics_attributes(self, batch_metrics, epoch_metrics):
        batch_metrics = list(map(get_loss_or_metric, batch_metrics))
        self.batch_metrics, batch_metrics_names = get_callables_and_names(batch_metrics)

        epoch_metrics = list(map(get_epoch_metric, epoch_metrics))
        self.epoch_metrics, epoch_metrics_names = get_callables_and_names(epoch_metrics)

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

    def fit(self,
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
            callbacks=None):
        # pylint: disable=line-too-long
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
            callbacks (List[~poutyne.framework.callbacks.Callback]): List of callbacks that will be called
                during training.
                (Default value = None)

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
        train_generator = self._dataloader_from_data((x, y), batch_size=batch_size)
        valid_generator = None
        if validation_data is not None:
            valid_generator = self._dataloader_from_data(validation_data, batch_size=batch_size)

        return self.fit_generator(train_generator,
                                  valid_generator=valid_generator,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  batches_per_step=batches_per_step,
                                  initial_epoch=initial_epoch,
                                  verbose=verbose,
                                  callbacks=callbacks)

    def _dataloader_from_data(self, args, batch_size):
        args = numpy_to_torch(args)
        dataset = TensorDataset(*args) if len(args) > 1 else args[0]
        generator = DataLoader(dataset, batch_size)
        return generator

    def fit_generator(self,
                      train_generator,
                      valid_generator=None,
                      *,
                      epochs=1000,
                      steps_per_epoch=None,
                      validation_steps=None,
                      batches_per_step=1,
                      initial_epoch=1,
                      verbose=True,
                      callbacks=None):
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
            callbacks (List[~poutyne.framework.callbacks.Callback]): List of callbacks that will be called during
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
            callbacks = [ProgressionCallback()] + callbacks
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)

        self.stop_training = False

        epoch_iterator = EpochIterator(train_generator,
                                       valid_generator,
                                       epochs=epochs,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_steps=validation_steps,
                                       initial_epoch=initial_epoch,
                                       callback=callback_list,
                                       batch_metrics_names=self.batch_metrics_names,
                                       epoch_metrics_names=self.epoch_metrics_names)

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
                    step.size = self.get_batch_size(x, y)

                    examples_in_step += step.size

                    step.loss, step.metrics, did_backprop, _ = self._fit_batch_n_batches_per_step(
                        x, y, batches_per_step, examples_in_step, callback=callback_list, step=step)

                    if did_backprop:
                        examples_in_step = 0

            if not did_backprop:
                # Did not step after last batch
                self._adjust_step_size(examples_in_step)
                self.optimizer.step()

            train_step_iterator.epoch_metrics = self._get_epoch_metrics()

            if valid_step_iterator is not None:
                self._validate(valid_step_iterator)
                valid_step_iterator.epoch_metrics = self._get_epoch_metrics()

            epoch_iterator.stop_training = self.stop_training

    def _fit_batch_n_batches_per_step(self,
                                      x,
                                      y,
                                      batches_per_step,
                                      examples_in_step,
                                      *,
                                      callback=Callback(),
                                      step=None,
                                      return_pred=False):
        # pylint: disable=too-many-locals
        zero_all_gradients = ((step.number - 1) % batches_per_step == 0)
        do_backprop = (step.number % batches_per_step == 0)

        if zero_all_gradients:
            self.optimizer.zero_grad()

        loss_tensor, metrics, pred_y = self._compute_loss_and_metrics(x,
                                                                      y,
                                                                      return_loss_tensor=True,
                                                                      return_pred=return_pred)

        adjusted_loss_tensor = loss_tensor * step.size
        adjusted_loss_tensor.backward()

        callback.on_backward_end(step)

        if do_backprop:
            self._adjust_step_size(examples_in_step)
            self.optimizer.step()

        loss = float(loss_tensor)
        return loss, metrics, do_backprop, pred_y

    def _fit_generator_one_batch_per_step(self, epoch_iterator, callback_list):
        for train_step_iterator, valid_step_iterator in epoch_iterator:
            with self._set_training_mode(True):
                for step, (x, y) in train_step_iterator:
                    step.loss, step.metrics, _ = self._fit_batch(x, y, callback=callback_list, step=step.number)
                    step.size = self.get_batch_size(x, y)

            train_step_iterator.epoch_metrics = self._get_epoch_metrics()

            if valid_step_iterator is not None:
                self._validate(valid_step_iterator)
                valid_step_iterator.epoch_metrics = self._get_epoch_metrics()

            epoch_iterator.stop_training = self.stop_training

    def _fit_batch(self, x, y, *, callback=Callback(), step=None, return_pred=False):
        self.optimizer.zero_grad()

        loss_tensor, metrics, pred_y = self._compute_loss_and_metrics(x,
                                                                      y,
                                                                      return_loss_tensor=True,
                                                                      return_pred=return_pred)

        loss_tensor.backward()
        callback.on_backward_end(step)
        self.optimizer.step()

        loss = float(loss_tensor)
        return loss, metrics, pred_y

    def _adjust_step_size(self, examples_in_step):
        for param in self.network.parameters():
            if param.grad is not None:
                param.grad /= examples_in_step

    def _process_input(self, *args):
        args = numpy_to_torch(args)
        if self.device is not None:
            args = torch_to(args, self.device)
        return args[0] if len(args) == 1 else args

    def train_on_batch(self, x, y, return_pred=False):
        """
        Trains the network for the batch ``(x, y)`` and computes the loss and the metrics, and
        optionally returns the predictions.

        Args:
            x: Input data as a batch.
            y: Target data as a batch.
            return_pred (bool, optional): Whether to return the predictions.
                (Default value = False)

        Returns:
            Float ``loss`` if no metrics were specified and ``return_pred`` is false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is the predictions with tensors converted into Numpy
            arrays.
        """
        if self.optimizer is None:
            raise ValueError("Impossible to fit when optimizer is None.")

        with self._set_training_mode(True):
            self._transfer_optimizer_state_to_right_device()
            loss, metrics, pred_y = self._fit_batch(x, y, return_pred=return_pred)
        return self._format_return(loss, metrics, pred_y, return_pred)

    def _format_return(self, loss, metrics, pred_y, return_pred, true_y=None, return_ground_truth=False):
        # pylint: disable=too-many-arguments
        ret = (loss, )

        ret += tuple(metrics.tolist()) if len(metrics) <= 1 else (metrics, )

        if return_pred:
            ret += (pred_y, )

        if return_ground_truth:
            ret += (true_y, )

        return ret[0] if len(ret) == 1 else ret

    def predict(self, x, batch_size=32):
        """
        Returns the predictions of the network given a dataset ``x``, where the tensors are
        converted into Numpy arrays.

        Args:
            x (Union[~torch.Tensor, ~numpy.ndarray] or Union[tuple, list] of Union[~torch.Tensor, ~numpy.ndarray]):
                Input to the model. Union[Tensor, ndarray] if the model has a single input.
                Union[tuple, list] of Union[Tensor, ndarray] if the model has multiple inputs.
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)

        Returns:
            Numpy arrays of the predictions.
        """
        x = x if isinstance(x, (tuple, list)) else (x, )
        generator = self._dataloader_from_data(x, batch_size=batch_size)
        return self.predict_generator(generator, concatenate_returns=True)

    def predict_generator(self, generator, *, steps=None, concatenate_returns=None):
        """
        Returns the predictions of the network given batches of samples ``x``, where the tensors are
        converted into Numpy arrays.

        generator: Generator-like object for the dataset. The generator must yield a batch of
            samples. See the :func:`fit_generator()` method for details on the types of generators
            supported. This should only yield input data ``x`` and not the target ``y``.
        steps (int, optional): Number of iterations done on ``generator``.
            (Defaults the number of steps needed to see the entire dataset)
        concatenate_returns (bool, optional): Whether to concatenate the predictions
            or the ground truths when returning them. Currently defaults to False but
            will default to True in the next version. A warning is raised if not set in
            the current version but the warning will be removed in the version. Disabling
            the warning as instructed in it switches to the new behavior when
            ``concatenate_returns`` is not set.

        Returns:
            List of the predictions of each batch with tensors converted into Numpy arrays.
        """
        if concatenate_returns is None and warning_settings['concatenate_returns'] == 'warn':
            warnings.warn("In the next version of Poutyne, the argument 'concatenate_returns' "
                          "of 'predict_generator' will default to True. To avoid this warning, "
                          "set 'concatenate_returns' to an appropriate boolean value in the "
                          "keyword arguments or get the new behavior by disabling this warning with\n"
                          "from poutyne.framework import warning_settings\n"
                          "warning_settings['concatenate_returns'] = 'ignore'\n"
                          "This warning will be removed in the next version.")
            concatenate_returns = False
        elif concatenate_returns is None:
            concatenate_returns = True

        if steps is None and hasattr(generator, '__len__'):
            steps = len(generator)
        pred_y = []
        with self._set_training_mode(False):
            for _, x in _get_step_iterator(steps, generator):
                x = self._process_input(x)
                x = x if isinstance(x, (tuple, list)) else (x, )
                pred_y.append(torch_to_numpy(self.network(*x)))
        if concatenate_returns:
            return _concat(pred_y)
        return pred_y

    def predict_on_batch(self, x):
        """
        Returns the predictions of the network given a batch ``x``, where the tensors are converted
        into Numpy arrays.

        Args:
            x: Input data as a batch.
        Returns:
            The predictions with tensors converted into Numpy arrays.
        """
        with self._set_training_mode(False):
            x = self._process_input(x)
            x = x if isinstance(x, (tuple, list)) else (x, )
            return torch_to_numpy(self.network(*x))

    def evaluate(self, x, y, *, batch_size=32, return_pred=False, callbacks=None):
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
            callbacks (List[~poutyne.framework.callbacks.Callback]): List of callbacks that will be called during
                testing. (Default value = None)

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

        """
        callbacks = [] if callbacks is None else callbacks

        # Copy callback list.
        callbacks = list(callbacks)

        generator = self._dataloader_from_data((x, y), batch_size=batch_size)
        return self.evaluate_generator(generator,
                                       steps=len(generator),
                                       return_pred=return_pred,
                                       concatenate_returns=True,
                                       callbacks=callbacks)

    def evaluate_generator(self,
                           generator,
                           *,
                           steps=None,
                           return_pred=False,
                           return_ground_truth=False,
                           concatenate_returns=None,
                           callbacks=None):
        # pylint: disable=too-many-locals
        """
        Computes the loss and the metrics of the network on batches of samples and optionaly returns
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
            concatenate_returns (bool, optional): Whether to concatenate the predictions
                or the ground truths when returning them. Currently defaults to False but
                will default to True in the next version. A warning is raised if not set in
                the current version but the warning will be removed in the version. Disabling
                the warning as instructed in it switches to the new behavior when
                ``concatenate_returns`` is not set.
            callbacks (List[~poutyne.framework.callbacks.Callback]): List of callbacks that will be called during
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

            If ``return_pred`` is True, ``pred_y`` is the list of the predictions
            of each batch with tensors converted into Numpy arrays. It is otherwise ommited.

            If ``return_ground_truth`` is True, ``true_y`` is the list of the ground truths
            of each batch with tensors converted into Numpy arrays. It is otherwise ommited.
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
                              batch_metrics=[my_metric_fn], epoch_metrics=[MyEpochMetricClass()])
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
        """
        if (return_pred or return_ground_truth) \
                and concatenate_returns is None and warning_settings['concatenate_returns'] == 'warn':
            warnings.warn("In the next version of Poutyne, the argument 'concatenate_returns' "
                          "of 'evaluate_generator' will default to True. To avoid this warning, "
                          "set 'concatenate_returns' to an appropriate boolean value in the "
                          "keyword arguments or get the new behavior by disabling this warning with\n"
                          "from poutyne.framework import warning_settings\n"
                          "warning_settings['concatenate_returns'] = 'ignore'\n"
                          "This warning will be removed in the next version.")
            concatenate_returns = False
        elif concatenate_returns is None:
            concatenate_returns = True

        callbacks = [] if callbacks is None else callbacks

        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)

        callback_list.on_test_begin({})

        if steps is None:
            steps = len(generator)
        step_iterator = StepIterator(generator, steps, self.batch_metrics_names, callback_list, mode="test")
        loss, batch_metrics, pred_y, true_y = self._validate(step_iterator,
                                                             return_pred=return_pred,
                                                             return_ground_truth=return_ground_truth)
        epoch_metrics = self._get_epoch_metrics()
        metrics = np.concatenate((batch_metrics, epoch_metrics))

        if return_pred and concatenate_returns:
            pred_y = _concat(pred_y)
        if return_ground_truth and concatenate_returns:
            true_y = _concat(true_y)

        res = self._format_return(loss, metrics, pred_y, return_pred, true_y, return_ground_truth)

        callback_list.on_test_end(res)

        return res

    def evaluate_on_batch(self, x, y, *, return_pred=False):
        """
        Computes the loss and the metrics of the network on a single batch of samples and optionally
        returns the predictions.

        Args:
            x: Input data as a batch.
            y: Target data as a batch.
            return_pred (bool, optional): Whether to return the predictions for ``batch``.
                (Default value = False)

        Returns:
            Tuple ``(loss, metrics, pred_y)`` where specific elements are omitted if not
            applicable. If only loss is applicable, then it is returned as a float.

            `metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            If ``return_pred`` is True, ``pred_y`` is the list of the predictions
            of each batch with tensors converted into Numpy arrays. It is otherwise ommited.
        """
        with self._set_training_mode(False):
            loss, metrics, pred_y = self._compute_loss_and_metrics(x, y, return_pred=return_pred)
        return self._format_return(loss, metrics, pred_y, return_pred)

    def _validate(self, step_iterator, return_pred=False, return_ground_truth=False):
        pred_list = None
        true_list = None
        if return_pred:
            pred_list = []
        if return_ground_truth:
            true_list = []

        with self._set_training_mode(False):
            for step, (x, y) in step_iterator:
                step.loss, step.metrics, pred_y = self._compute_loss_and_metrics(x, y, return_pred=return_pred)
                if return_pred:
                    pred_list.append(pred_y)
                if return_ground_truth:
                    true_list.append(torch_to_numpy(y))

                step.size = self.get_batch_size(x, y)

        return step_iterator.loss, step_iterator.metrics, pred_list, true_list

    def _compute_loss_and_metrics(self, x, y, return_loss_tensor=False, return_pred=False):
        x, y = self._process_input(x, y)
        x = x if isinstance(x, (list, tuple)) else (x, )
        pred_y = self.network(*x)
        loss = self.loss_function(pred_y, y)
        if not return_loss_tensor:
            loss = float(loss)
        with torch.no_grad():
            metrics = self._compute_batch_metrics(pred_y, y)
            for epoch_metric in self.epoch_metrics:
                epoch_metric(pred_y, y)

        pred_y = torch_to_numpy(pred_y) if return_pred else None
        return loss, metrics, pred_y

    def _compute_batch_metrics(self, pred_y, y):
        metrics = [metric(pred_y, y) for metric in self.batch_metrics]
        return self._compute_metric_array(metrics, self.unflatten_batch_metrics_names)

    def _get_epoch_metrics(self):
        metrics = [epoch_metric.get_metric() for epoch_metric in self.epoch_metrics]
        for epoch_metric in self.epoch_metrics:
            epoch_metric.reset()
        return self._compute_metric_array(metrics, self.unflatten_epoch_metrics_names)

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
            [metric for names, metrics in zip(names_list, metrics_list) for metric in _get_metric(names, metrics)])

    def get_batch_size(self, x, y):
        """
        This method infers the batch size of a batch. Here is the inferring algorithm used to compute the
        batch size. ``x`` and ``y`` are tested in this order at each step of the inferring algorithm. If one
        step succeed for one of ``x`` or ``y``, the algorithm stops.

        - Step 1: if ``x`` or ``y`` is a tensor or a Numpy array, then the ``len()`` is returned.
        - Step 2: if ``x`` or ``y`` is a list or a tuple, then the ``len()`` of the first element is returned if it
          is a tensor or a Numpy array.
        - Step 3: if ``x`` or ``y`` is a dict, then the value for the key ``'batch_size'`` is returned if it is of
          integral type.
        - Step 4: if ``x`` or ``y`` is a dict, then the ``len()`` of the first element of ``.values()`` is returned
          if it is a tensor or a Numpy array.

        If inferring the batch size is not possible, the batch size is set to 1 and, thus, the computed
        loss and metrics at the end of each epoch is the mean of the batches' losses and metrics. In which
        case, a warning is also raised. To disable this warning, set

        .. code-block:: python

            from poutyne.framework import warning_settings\n
            warning_settings['batch_size'] = 'ignore'\n\n

        Args:
            x: Input data as a batch.
            y: Target data as a batch.
        """

        def is_torch_or_numpy(v):
            return torch.is_tensor(v) or isinstance(v, np.ndarray)

        for v in [x, y]:
            if is_torch_or_numpy(v):
                return len(v)
        for v in [x, y]:
            if isinstance(v, (tuple, list)):
                if is_torch_or_numpy(v[0]):
                    return len(v[0])
        for v in [x, y]:
            if isinstance(v, dict):
                if 'batch_size' in v and isinstance(v['batch_size'], numbers.Integral):
                    return v['batch_size']
        for v in [x, y]:
            if isinstance(v, dict):
                first_value = list(v.values())[0]
                if is_torch_or_numpy(first_value):
                    return len(first_value)

        if warning_settings['batch_size'] == 'warn':
            warnings.warn("Inferring the batch size is not possible. Hence, "
                          "the batch size is set to 1 and, thus, the computed "
                          "loss and metrics at the end of each epoch is the "
                          "mean of the batches' losses and metrics. To disable "
                          "this warning, set\n"
                          "from poutyne.framework import warning_settings\n"
                          "warning_settings['batch_size'] = 'ignore'\n\n"
                          "Here is the inferring algorithm used to compute the "
                          "batch size. 'x' and 'y' are tested in this order at "
                          "each step of the inferring algorithm. If one step "
                          "succeed for one of 'x' or 'y', the algorithm stops.\n\n"
                          "Step 1: if 'x' or 'y' is a tensor or a Numpy array, "
                          "then the 'len()' is returned.\n"
                          "Step 2: if 'x' or 'y' is a list or a tuple, then the "
                          "'len()' of the first element is returned if it is a "
                          "tensor or a Numpy array.\n"
                          "Step 3: if 'x' or 'y' is a dict, then the value for "
                          "the key 'batch_size' is returned if it is of integral "
                          "type.\n"
                          "Step 4: if 'x' or 'y' is a dict, then the 'len()' of "
                          "the first element of '.values()' is returned if it is a "
                          "tensor or a Numpy array.\n")
        return 1

    def load_weights(self, f):
        """
        Loads the weights saved using the :func:`torch.save()` method or the :func:`save_weights()` method
        of this class. Contrary to :func:`torch.load()`, the weights are not transfered to the device
        from which they were saved from. In other words, the PyTorch module will stay on the same
        device it already is on.

        Args:
            f: File-like object (has to implement fileno that returns a file descriptor) or string
                containing a file name.
        """
        self.set_weights(torch.load(f, map_location='cpu'))

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
                    for _, v in self.optimizer.state[p].items():
                        if torch.is_tensor(v) and p.device != v.device:
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

        self.optimizer.state = defaultdict(dict, {name_to_param[name]: state for name, state in named_state})

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

    def set_weights(self, weights):
        """
        Modifies the weights of the network with the given weights.

        Args:
            weights (dict): Weights returned by either :func:`get_weights()` or :func:`get_weight_copies()`.
        """
        self.network.load_state_dict(weights)

    def cuda(self, *args, **kwargs):
        """
        Tranfers the network on the GPU. The arguments are passed to the :meth:`torch.nn.Module.cuda()` method.
        Notice that the device is saved so that the batches can send to the right device before passing it to
        the network.

        Note:
            PyTorch optimizers assume that the parameters have been transfered to the right device
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

        self._transfer_loss_and_metrics_modules_to_right_device()

        return self

    def cpu(self, *args, **kwargs):
        """
        Tranfers the network on the CPU. The arguments are passed to the :meth:`torch.nn.Module.cpu()`
        method. Notice that the device is saved so that the batches can send to the right device
        before passing it to the network.

        Note:
            PyTorch optimizers assume that the parameters have been transfered to the right device
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

        self._transfer_loss_and_metrics_modules_to_right_device()

        return self

    def to(self, device):
        """
        Tranfers the network on the specified device. The device is saved so that the batches can
        send to the right device before passing it to the network.

        Note:
            PyTorch optimizers assume that the parameters have been transfered to the right device
            before their creations. Furthermore, future versions of PyTorch will no longer modify
            the parameters of a PyTorch module in-place when transferring them to another device.
            See this `issue <https://github.com/pytorch/pytorch/issues/7844>`_ and this
            `pull request <https://github.com/pytorch/pytorch/pull/21613>`_ for details.

            Since Poutyne supposes that the optimizer has been initialized before the Poutyne Model,
            necessarily the parameters are not guaranteed to be in sync with those contained in the
            optimizer once the PyTorch module is transferred to another device. Thus, this method
            takes care of this inconsistency by updating the parameters inside the optimizer.

        Args:
            device (torch.torch.device): The device to which the network is sent.

        Returns:
            `self`.
        """
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
