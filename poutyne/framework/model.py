import warnings
import contextlib
import numpy as np

import torch
from torch.utils.data import DataLoader

from poutyne import torch_to_numpy, numpy_to_torch, torch_to
from poutyne.utils import TensorDataset
from .iterators import EpochIterator, StepIterator, _get_step_iterator
from .callbacks import CallbackList, ProgressionCallback, Callback
from .metrics import get_loss_or_metric
from .optimizers import get_optimizer
from .warning_manager import warning_settings
from ..utils import _concat


class Model:
    # pylint: disable=line-too-long
    """
    The Model class encapsulates a PyTorch module/network, a PyTorch optimizer, a loss function and
    metric functions. It allows the user to train a neural network without hand-coding the
    epoch/step logic.

    Args:
        model (torch.nn.Module): A PyTorch module.
        optimizer (torch.optim.Optimizer): Initialized PyTorch optimizer.
        loss_function: Loss function. It can be any PyTorch loss layer or custom loss function. It
            can also be a string with the same name as a PyTorch loss function (either the functional or
            object name). The loss function must have the signature ``loss_function(input, target)``
            where ``input`` is the prediction of the network and ``target`` is the ground truth.
        metrics (list): List of functions with the same signature as the loss function. Each metric
            can be any PyTorch loss function. It can also be a string with the same name as a PyTorch
            loss function (either the functional or object name). 'accuracy' (or just 'acc') is also a
            valid metric. Each metric function is called on each batch of the optimization and on the
            validation batches at the end of the epoch.
            (Default value = None)

    Attributes:
        model (torch.nn.Module): The associated PyTorch module.
        optimizer (torch.optim.Optimizer): The associated PyTorch optimizer.
        loss_function: The associated loss function.
        metrics (list): The associated metric functions.

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

            pytorch_module = torch.nn.Linear(num_features, num_classes) # Our network

            # We create and optimize our model
            model = Model(pytorch_module, 'sgd', 'cross_entropy', metrics=['accuracy'])
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

           pytorch_module = torch.nn.Linear(num_features, num_train_samples)

           model = Model(pytorch_module, 'sgd', 'cross_entropy', metrics=['accuracy'])
           model.fit_generator(train_generator,
                               valid_generator,
                               epochs=5)

        .. code-block:: none

            Epoch 1/5 0.05s Step 25/25: loss: 6.752676, acc: 0.000000, val_loss: 6.575071, val_acc: 0.000000
            Epoch 2/5 0.03s Step 25/25: loss: 6.454859, acc: 0.125000, val_loss: 6.279577, val_acc: 0.000000
            Epoch 3/5 0.03s Step 25/25: loss: 6.158523, acc: 2.125000, val_loss: 5.985811, val_acc: 9.500000
            ...

    """

    def __init__(self, model, optimizer, loss_function, *, metrics=None):
        metrics = [] if metrics is None else metrics

        self.model = model
        self.optimizer = get_optimizer(optimizer, self.model)
        self.loss_function = get_loss_or_metric(loss_function)
        self.metrics = list(map(get_loss_or_metric, metrics))
        self.metrics_names = [metric.__name__ for metric in self.metrics]
        self.device = None

    def fit(self,
            x,
            y,
            validation_data=None,
            batch_size=32,
            epochs=1000,
            steps_per_epoch=None,
            validation_steps=None,
            initial_epoch=1,
            verbose=True,
            callbacks=None):
        # pylint: disable=line-too-long
        # pylint: disable=too-many-arguments
        """
        Trains the model on a dataset. This method creates generators and calls
        the ``fit_generator`` method.

        Args:
            x (Union[Tensor, np.ndarray] or Union[tuple, list] of Union[Tensor, np.ndarray]):
                Training dataset. Union[Tensor, np.ndarray] if the model has a single input.
                Union[tuple, list] of Union[Tensor, np.ndarray] if the model has multiple inputs.
            y (Union[Tensor, np.ndarray] or Union[tuple, list] of Union[Tensor, np.ndarray]):
                Target. Union[Tensor, np.ndarray] if the model has a single output.
                Union[tuple, list] of Union[Tensor, np.ndarray] if the model has multiple outputs.
                validation_data (Optional[tuple of (``x_val``, ``y_val``)]):
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
                (Defaults to ``steps_per_epoch`` if provided or the number of steps needed to
                see the entire validation dataset)
            initial_epoch (int, optional): Epoch at which to start training
                (useful for resuming a previous training run).
                (Default value = 1)
            verbose (bool): Whether to display the progress of the training.
                (Default value = True)
            callbacks (list of poutyne.framework.Callback): List of callbacks that will be called
                during training.
                (Default value = None)

        Returns:
            List of dict containing the history of each epoch.

        Example:
            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function)
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
                      initial_epoch=1,
                      verbose=True,
                      callbacks=None):
        # pylint: disable=too-many-locals, line-too-long
        """
        Trains the model on a dataset using a generator.

        Args:
            train_generator: Generator-like object for the training dataset. The generator must
                yield a batch in the form of a tuple (x, y) where ``x`` is the input and ``y`` is the
                target ``len(x)`` is taken as the batch size (or the first element of ``x`` in case of
                multi inputs). The loss and the metrics are averaged using this batch size. If the batch
                size cannot be, inferred then a warning is raised and the "batch size" defaults to 1.

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
                (Defaults the number of steps needed to see the entire
                training dataset)
            validation_steps (int, optional): Same as for ``steps_per_epoch`` but for the validation dataset.
                (Defaults to ``steps_per_epoch`` if provided or the number of steps needed to see the entire
                validation dataset)
            initial_epoch (int, optional): Epoch at which to start training (useful for resuming a previous
                training run).
                (Default value = 1)
            verbose (bool): Whether to display the progress of the training.
                (Default value = True)
            callbacks (list of poutyne.framework.Callback): List of callbacks that will be called during training.
                (Default value = None)

        Returns:
            List of dict containing the history of each epoch.

        Example:
            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function)
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
                                       metrics_names=self.metrics_names)

        for train_step_iterator, valid_step_iterator in epoch_iterator:
            self.model.train(True)
            with torch.enable_grad():
                for step, (x, y) in train_step_iterator:
                    step.loss, step.metrics, _ = self._fit_batch(x, y, callback=callback_list, step=step.number)
                    step.size = self._get_batch_size(x, y)

            if valid_step_iterator is not None:
                self._validate(valid_step_iterator)

            epoch_iterator.stop_training = self.stop_training

        return epoch_iterator.epoch_logs

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

    def _process_input(self, *args):
        args = numpy_to_torch(args)
        if self.device is not None:
            args = torch_to(args, self.device)
        return args[0] if len(args) == 1 else args

    def train_on_batch(self, x, y, return_pred=False):
        """
        Trains the model for the batch ``(x, y)`` and computes the loss and the metrics, and
        optionaly returns the predictions.

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
        self.model.train(True)
        with torch.enable_grad():
            self._transfer_optimizer_state_to_right_device()
            loss, metrics, pred_y = self._fit_batch(x, y, return_pred=return_pred)
        return self._format_return(loss, metrics, pred_y, return_pred)

    def _format_return(self, loss, metrics, pred_y, return_pred):
        ret = (loss, )

        ret += tuple(metrics.tolist()) if len(metrics) <= 1 else (metrics, )

        if return_pred:
            ret += (pred_y, )

        return ret[0] if len(ret) == 1 else ret

    def predict(self, x, batch_size=32):
        """
        Returns the predictions of the network given a dataset ``x``, where the tensors are
        converted into Numpy arrays.

        Args:
            x (Union[Tensor, np.ndarray] or Union[tuple, list] of Union[Tensor, np.ndarray]):
                Input to the model. Union[Tensor, np.ndarray] if the model has a single input.
                Union[tuple, list] of Union[Tensor, np.ndarray] if the model has multiple inputs.
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)

        Returns:
            Numpy arrays of the predictions.
        """
        x = x if isinstance(x, (tuple, list)) else (x, )
        generator = self._dataloader_from_data(x, batch_size=batch_size)
        pred_y = self.predict_generator(generator)
        return _concat(pred_y)

    def predict_generator(self, generator, *, steps=None):
        """
        Returns the predictions of the network given batches of samples ``x``, where the tensors are
        converted into Numpy arrays.

        generator: Generator-like object for the dataset. The generator must yield a batch of
            samples. See the ``fit_generator()`` method for details on the types of generators
            supported. This should only yield input data ``x`` and not the target ``y``.
        steps (int, optional): Number of iterations done on ``generator``.
            (Defaults the number of steps needed to see the entire dataset)

        Returns:
            List of the predictions of each batch with tensors converted into Numpy arrays.
        """
        if steps is None and hasattr(generator, '__len__'):
            steps = len(generator)
        pred_y = []
        self.model.eval()
        with torch.no_grad():
            for _, x in _get_step_iterator(steps, generator):
                x = self._process_input(x)
                x = x if isinstance(x, (tuple, list)) else (x, )
                pred_y.append(torch_to_numpy(self.model(*x)))
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
        self.model.eval()
        with torch.no_grad():
            x = self._process_input(x)
            x = x if isinstance(x, (tuple, list)) else (x, )
            return torch_to_numpy(self.model(*x))

    def evaluate(self, x, y, batch_size=32, return_pred=False):
        """
        Computes the loss and the metrics of the network on batches of samples and optionally
        returns the predictions.

        Args:
            x (Union[Tensor, np.ndarray] or Union[tuple, list] of Union[Tensor, np.ndarray]):
                Input to the model. Union[Tensor, np.ndarray] if the model has a single input.
                Union[tuple, list] of Union[Tensor, np.ndarray] if the model has multiple inputs.
            y (Union[Tensor, np.ndarray] or Union[tuple, list] of Union[Tensor, np.ndarray]):
                Target, corresponding ground truth.
                Union[Tensor, np.ndarray] if the model has a single output.
                Union[tuple, list] of Union[Tensor, np.ndarray] if the model has multiple outputs.
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)
            return_pred (bool, optional): Whether to return the predictions.
                (Default value = False)

        Returns:
            Float ``loss`` if no metrics were specified and ``return_pred`` is false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is a Numpy array of the predictions.
        """
        generator = self._dataloader_from_data((x, y), batch_size=batch_size)
        ret = self.evaluate_generator(generator, steps=len(generator), return_pred=return_pred)
        if return_pred:
            ret = (*ret[:-1], _concat(ret[-1]))
        return ret

    def evaluate_generator(self, generator, *, steps=None, return_pred=False):
        """
        Computes the loss and the metrics of the network on batches of samples and optionaly returns
        the predictions.

        Args:
            generator: Generator-like object for the dataset. See the ``fit_generator()`` method for
                details on the types of generators supported.
            steps (int, optional): Number of iterations done on ``generator``.
                (Defaults the number of steps needed to see the entire dataset)
            return_pred (bool, optional): Whether to return the predictions.
                (Default value = False)

        Returns:
            Float ``loss`` if no metrics were specified and ``return_pred`` is false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is the list of the predictions of each batch with tensors
            converted into Numpy arrays.

        Example:
            With no metrics:

            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function,
                              metrics=None)
                loss = model.evaluate_generator(test_generator)

            With only one metric:

            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function,
                              metrics=[my_metric_fn])
                loss, my_metric = model.evaluate_generator(test_generator)

            With several metrics:

            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function,
                              metrics=[my_metric1_fn, my_metric2_fn])
                loss, (my_metric1, my_metric2) = model.evaluate_generator(test_generator)

            With metrics and ``return_pred`` flag:

            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function,
                              metrics=[my_metric1_fn, my_metric2_fn])
                loss, (my_metric1, my_metric2), pred_y = model.evaluate_generator(
                    test_generator, return_pred=True
                )
        """
        if steps is None:
            steps = len(generator)
        step_iterator = StepIterator(generator, steps, Callback(), self.metrics_names)
        loss, metrics, pred_y = self._validate(step_iterator, return_pred=return_pred)
        return self._format_return(loss, metrics, pred_y, return_pred)

    def evaluate_on_batch(self, x, y, return_pred=False):
        """
        Computes the loss and the metrics of the network on a single batch of samples and optionally
        returns the predictions.

        Args:
            x: Input data as a batch.
            y: Target data as a batch.
            return_pred (bool, optional): Whether to return the predictions for ``batch``.
                (Default value = False)

        Returns:
            Float ``loss`` if no metrics were specified and ``return_pred`` is false.

            Otherwise, tuple ``(loss, metrics)`` if ``return_pred`` is false.
            ``metrics`` is a Numpy array of size ``n``, where ``n`` is the
            number of metrics if ``n > 1``. If ``n == 1``, then ``metrics`` is a
            float. If ``n == 0``, the ``metrics`` is omitted.

            Tuple ``(loss, metrics, pred_y)`` if ``return_pred`` is true where
            ``pred_y`` is the list of the predictions of each batch with tensors
            converted into Numpy arrays.
        """
        self.model.eval()
        with torch.no_grad():
            loss, metrics, pred_y = self._compute_loss_and_metrics(x, y, return_pred=return_pred)
        return self._format_return(loss, metrics, pred_y, return_pred)

    def _validate(self, step_iterator, return_pred=False):
        pred_list = None
        if return_pred:
            pred_list = []

        self.model.eval()
        with torch.no_grad():
            for step, (x, y) in step_iterator:
                step.loss, step.metrics, pred_y = self._compute_loss_and_metrics(x, y, return_pred=return_pred)
                if return_pred:
                    pred_list.append(pred_y)

                step.size = self._get_batch_size(x, y)

        return step_iterator.loss, step_iterator.metrics, pred_list

    def _compute_loss_and_metrics(self, x, y, return_loss_tensor=False, return_pred=False):
        x, y = self._process_input(x, y)
        x = x if isinstance(x, (list, tuple)) else (x, )
        pred_y = self.model(*x)
        loss = self.loss_function(pred_y, y)
        if not return_loss_tensor:
            loss = float(loss)
        with torch.no_grad():
            metrics = self._compute_metrics(pred_y, y)

        pred_y = torch_to_numpy(pred_y) if return_pred else None
        return loss, metrics, pred_y

    def _compute_metrics(self, pred_y, y):
        return np.array([float(metric(pred_y, y)) for metric in self.metrics])

    def _get_batch_size(self, x, y):
        if torch.is_tensor(x) or isinstance(x, np.ndarray):
            return len(x)
        if isinstance(x, (tuple, list)):
            if torch.is_tensor(x[0]) or isinstance(x[0], np.ndarray):
                return len(x[0])
        if torch.is_tensor(y) or isinstance(y, np.ndarray):
            return len(y)
        if isinstance(y, (tuple, list)):
            if torch.is_tensor(y[0]) or isinstance(y[0], np.ndarray):
                return len(y[0])

        if warning_settings['batch_size'] == 'warn':
            warnings.warn("When 'x' or 'y' are not tensors nor Numpy arrays, "
                          "the batch size is set to 1 and, thus, the computed "
                          "loss and metrics at the end of each epoch is the "
                          "mean of the batches' losses and metrics. To disable "
                          "this warning, set\n"
                          "from poutyne.framework import warning_settings\n"
                          "warning_settings['batch_size'] = 'ignore'")
        return 1

    def load_weights(self, f):
        """
        Loads the weights saved using the ``torch.save()`` method or the ``save_weights()`` method
        of this class. Contrary to ``torch.load()``, the weights are not transfered to the device
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
        torch.save(self.model.state_dict(), f)

    def load_optimizer_state(self, f):
        """
        Loads the optimizer state saved using the ``torch.save()`` method or the
        ``save_optimizer_state()`` method of this class.

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
        param_to_name = {param: name for name, param in self.model.named_parameters()}

        param_name_groups = []
        for group in self.optimizer.param_groups:
            param_name_groups.append([param_to_name[param] for param in group['params']])

        named_state = {param_to_name[param]: state for param, state in self.optimizer.state.items()}

        return param_name_groups, named_state

    def _set_named_optimizer_attrs(self, param_name_groups, named_state):
        name_to_param = dict(self.model.named_parameters())

        for param_name_group, optim_group in zip(param_name_groups, self.optimizer.param_groups):
            optim_group['params'] = [
                name_to_param[param_name] if optim_param is not name_to_param[param_name] else optim_param
                for param_name, optim_param in zip(param_name_group, optim_group['params'])
            ]

        self.optimizer.state = {name_to_param[name]: state for name, state in named_state}

    @contextlib.contextmanager
    def _update_optim_device_context_manager(self):
        param_name_groups, named_state = self._get_named_optimizer_attrs()
        try:
            yield
        finally:
            self._set_named_optimizer_attrs(param_name_groups, named_state)

    def get_weights(self):
        """
        Returns a dictionary containing the parameters of the network. The tensors are just
        references to the parameters. To get copies of the weights, see the ``get_weight_copies()``
        method.
        """
        return self.model.state_dict()

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
            weights (dict): Weights returned by either ``get_weights()`` or ``get_weight_copies()``.
        """
        self.model.load_state_dict(weights)

    def cuda(self, *args, **kwargs):
        """
        Tranfers the network on the GPU. The arguments are passed to the ``torch.nn.Module.cuda()``
        method. Notice that the device is saved so that the batches can send to the right device
        before passing it to the network.

        Note:
            PyTorch optimizers assume that the parameters have been transfered to the right device
            before their creations. Furthermore, future versions of PyTorch won't modify in-place
            anymore the parameters of a PyTorch module when transfering them to another device.
            See this `issue <https://github.com/pytorch/pytorch/issues/7844>`_ and this
            `pull request <https://github.com/pytorch/pytorch/pull/21613>`_ for details.

            Since Poutyne supposes that the optimizer has ben initialized before the Poutyne Model,
            necessarily the parameters are not guaranteed to be in sync with those contained in the
            optimizer once the PyTorch module is transfered to another device. Thus, this method
            takes care of this inconsistency by updating the parameters inside the optimizer.

        Returns:
            `self`.
        """
        with self._update_optim_device_context_manager():
            self.model.cuda(*args, **kwargs)

        # Assuming the PyTorch module has at least one parameter.
        self.device = next(self.model.parameters()).device

        self._transfer_loss_and_metrics_modules_to_right_device()

        return self

    def cpu(self, *args, **kwargs):
        """
        Tranfers the network on the CPU. The arguments are passed to the ``torch.nn.Module.cpu()``
        method. Notice that the device is saved so that the batches can send to the right device
        before passing it to the network.

        Note:
            PyTorch optimizers assume that the parameters have been transfered to the right device
            before their creations. Furthermore, future versions of PyTorch won't modify in-place
            anymore the parameters of a PyTorch module when transfering them to another device.
            See this `issue <https://github.com/pytorch/pytorch/issues/7844>`_ and this
            `pull request <https://github.com/pytorch/pytorch/pull/21613>`_ for details.

            Since Poutyne supposes that the optimizer has been initialized before the Poutyne Model,
            necessarily the parameters are not guaranteed to be in sync with those contained in the
            optimizer once the PyTorch module is transferred to another device. Thus, this method
            takes care of this inconsistency by updating the parameters inside the optimizer.

        Returns:
            `self`.
        """
        with self._update_optim_device_context_manager():
            self.model.cpu(*args, **kwargs)

        # Assuming the PyTorch module has at least one parameter.
        self.device = next(self.model.parameters()).device

        self._transfer_loss_and_metrics_modules_to_right_device()

        return self

    def to(self, device):
        """
        Tranfers the network on the specified device. The device is saved so that the batches can
        send to the right device before passing it to the network.

        Note:
            PyTorch optimizers assume that the parameters have been transfered to the right device
            before their creations. Furthermore, future versions of PyTorch won't modify in-place
            anymore the parameters of a PyTorch module when transfering them to another device.
            See this `issue <https://github.com/pytorch/pytorch/issues/7844>`_ and this
            `pull request <https://github.com/pytorch/pytorch/pull/21613>`_ for details.

            Since Poutyne supposes that the optimizer has been initialized before the Poutyne Model,
            necessarily the parameters are not guaranteed to be in sync with those contained in the
            optimizer once the PyTorch module is transferred to another device. Thus, this method
            takes care of this inconsistency by updating the parameters inside the optimizer.

        Args:
            device (torch.device): The device to which the network is sent.

        Returns:
            `self`.
        """
        self.device = device
        with self._update_optim_device_context_manager():
            self.model.to(self.device)
        self._transfer_loss_and_metrics_modules_to_right_device()
        return self

    def _transfer_loss_and_metrics_modules_to_right_device(self):
        if isinstance(self.loss_function, torch.nn.Module):
            self.loss_function.to(self.device)

        for metric in self.metrics:
            if isinstance(metric, torch.nn.Module):
                metric.to(self.device)
        return self
