import warnings
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from .callbacks import CallbackList, ProgressionCallback
from pytoune import torch_to_numpy, tensors_to_variables, variables_to_tensors

class Model:
    """
    The Model class encapsulates a PyTorch module/network, a PyTorch optimizer,
    a loss function and metric functions. It allows the user to train a neural
    network without hand-coding the epoch/step logic.

    Args:
        model (torch.nn.Module): A PyTorch module.
        optimizer (torch.optim.Optimizer): Initialized PyTorch optimizer.
        loss_function: Loss function. It can be any PyTorch loss layer or
            custom loss function. The loss function must have the signature
            ``loss_function(input, target)`` where ``input`` is the prediction
            of the network and ``target`` is the ground truth.
        metrics (list): List of functions with the same signature as the loss
            function. It is called on each batch of the optimization and on the
            validation batches at the end of the epoch. (Default value = [])

    Attributes:
        model (torch.nn.Module): The associated PyTorch module.
        optimizer (torch.optim.Optimizer): The associated PyTorch optimizer.
        loss_function: The associated loss function.
        metrics (list): The associated metric functions.

    Example:
        Using dataset tensors::

            import torch
            from pytoune.framework import Model

            num_epochs = 10
            batch_size = 20

            num_features = 10

            # Our training dataset with 800 samples.
            num_train_samples = 800
            train_x = torch.rand(num_train_samples, num_features)
            train_y = torch.rand(num_train_samples, 1)

            # Our validation dataset with 200 samples.
            num_valid_samples = 200
            valid_x = torch.rand(num_valid_samples, num_features)
            valid_y = torch.rand(num_valid_samples, 1)

            pytorch_module = torch.nn.Linear(num_features, 1) # Our network
            loss_function = torch.nn.MSELoss() # Our loss function
            optimizer = torch.optim.SGD(pytorch_module.parameters(), lr=1e-3)

            # We create and optimize our model
            model = Model(pytorch_module, optimizer, loss_function)
            model.fit(train_x, train_y,
                      validation_x=valid_x,
                      validation_y=valid_y,
                      epochs=num_epochs,
                      batch_size=batch_size)

        .. code-block:: none

            Epoch 1/10 0.01s Step 40/40: loss: 0.710869, val_loss: 0.489602
            Epoch 2/10 0.01s Step 40/40: loss: 0.448081, val_loss: 0.305897
            Epoch 3/10 0.01s Step 40/40: loss: 0.301377, val_loss: 0.204526
            Epoch 4/10 0.01s Step 40/40: loss: 0.219414, val_loss: 0.148778
            Epoch 5/10 0.01s Step 40/40: loss: 0.173554, val_loss: 0.118251
            Epoch 6/10 0.01s Step 40/40: loss: 0.147825, val_loss: 0.101621
            Epoch 7/10 0.01s Step 40/40: loss: 0.133319, val_loss: 0.092615
            Epoch 8/10 0.01s Step 40/40: loss: 0.125070, val_loss: 0.087767
            Epoch 9/10 0.01s Step 40/40: loss: 0.120307, val_loss: 0.085166
            Epoch 10/10 0.01s Step 40/40: loss: 0.117488, val_loss: 0.083767

        Using PyTorch DataLoader::

           import torch
           from torch.utils.data import DataLoader, TensorDataset
           from pytoune.framework import Model

           num_epochs = 10
           batch_size = 20

           num_features = 10

           # Our training dataset with 800 samples.
           num_train_samples = 800
           train_x = torch.rand(num_train_samples, num_features)
           train_y = torch.rand(num_train_samples, 1)
           train_dataset = TensorDataset(train_x, train_y)
           train_generator = DataLoader(train_dataset, batch_size)

           # Our validation dataset with 200 samples.
           num_valid_samples = 200
           valid_x = torch.rand(num_valid_samples, num_features)
           valid_y = torch.rand(num_valid_samples, 1)
           valid_dataset = TensorDataset(valid_x, valid_y)
           valid_generator = DataLoader(valid_dataset, batch_size)

           pytorch_module = torch.nn.Linear(num_features, 1)
           loss_function = torch.nn.MSELoss()
           optimizer = torch.optim.SGD(pytorch_module.parameters(), lr=1e-3)

           model = Model(pytorch_module, optimizer, loss_function)
           model.fit_generator(train_generator,
                               valid_generator,
                               epochs=num_epochs)

        .. code-block:: none

            Epoch 1/10 0.01s Step 40/40: loss: 0.311442, val_loss: 0.243208
            Epoch 2/10 0.01s Step 40/40: loss: 0.223419, val_loss: 0.183428
            Epoch 3/10 0.01s Step 40/40: loss: 0.173739, val_loss: 0.150269
            Epoch 4/10 0.01s Step 40/40: loss: 0.145618, val_loss: 0.131929
            Epoch 5/10 0.01s Step 40/40: loss: 0.129623, val_loss: 0.121813
            Epoch 6/10 0.02s Step 40/40: loss: 0.120447, val_loss: 0.116241
            Epoch 7/10 0.02s Step 40/40: loss: 0.115111, val_loss: 0.113164
            Epoch 8/10 0.01s Step 40/40: loss: 0.111937, val_loss: 0.111448
            Epoch 9/10 0.01s Step 40/40: loss: 0.109983, val_loss: 0.110464
            Epoch 10/10 0.01s Step 40/40: loss: 0.108717, val_loss: 0.109868

    """

    def __init__(self, model, optimizer, loss_function, metrics=[]):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics
        self.metrics_names = [metric.__name__ for metric in metrics]

    def fit(self, x, y, validation_x=None, validation_y=None, batch_size=32, epochs=1000, steps_per_epoch=None, validation_steps=None, initial_epoch=1, verbose=True, callbacks=[]):
        """
        Trains the model on a dataset. This method creates generators and calls
        the ``fit_generator`` method.

        Args:
            x (Tensor): Training dataset.
            y (Tensor): Ground truth.
            validation_x (Tensor): Validation dataset. The validation datset
                is optional. (Default value = None)
            validation_y (Tensor): Validation ground truth.
                (Default value = None)
            batch_size (int): Number of samples given to the network at one time.
                (Default value = 32)
            epochs (int): Number of times the entire training dataset is seen.
                (Default value = 1000)
            steps_per_epoch (int, optional): Number of batch used during one
                epoch. Obviously, using this argument may cause one epoch not to
                see the entire training dataset or see it multiple times.
                (Defaults the number of steps needed to see the entire
                training dataset)
            validation_steps (int, optional): Same as for ``steps_per_epoch`` but
                for the validation dataset. (Defaults to ``steps_per_epoch`` if
                provided or the number of steps needed to see the entire
                validation dataset)
            initial_epoch (int, optional): Epoch at which to start training
                (useful for resuming a previous training run).
                (Default value = 1)
            verbose (bool): Whether to display the progress of the training.
                (Default value = True)
            callbacks (list of pytoune.framework.Callback): List of callbacks
                that will be called during training. (Default value = [])

        Returns:
            List of dict containing the history of each epoch.

        Example:
            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function)
                history = model.fit(train_x, train_y,
                                    validation_x=valid_x,
                                    validation_y=valid_y,
                                    epochs=num_epochs,
                                    batch_size=batch_size)
                                    verbose=False)
                print(*history, sep="\\n")

            .. code-block:: python

                {'epoch': 1, 'loss': 0.30211143642663957, 'val_loss': 0.25165273696184159}
                {'epoch': 2, 'loss': 0.2192931968718767, 'val_loss': 0.19234802126884459}
                {'epoch': 3, 'loss': 0.17256419658660888, 'val_loss': 0.15897458493709565}
                {'epoch': 4, 'loss': 0.14614091524854303, 'val_loss': 0.14015687778592109}
                {'epoch': 5, 'loss': 0.1311435048468411, 'val_loss': 0.12950410917401314}
                {'epoch': 6, 'loss': 0.12257619202136993, 'val_loss': 0.12342756390571594}
                {'epoch': 7, 'loss': 0.11762827709317207, 'val_loss': 0.11991316601634025}
                {'epoch': 8, 'loss': 0.11471841260790824, 'val_loss': 0.11783143356442452}
                {'epoch': 9, 'loss': 0.11295686885714531, 'val_loss': 0.11654961034655571}
                {'epoch': 10, 'loss': 0.11184301525354386, 'val_loss': 0.1157136932015419}
        """
        train_dataset = TensorDataset(x, y)
        train_generator = DataLoader(train_dataset, batch_size)

        valid_generator = None
        if validation_x is not None or validation_y is not None:
            validation_dataset = TensorDataset(validation_x, validation_y)
            valid_generator = DataLoader(validation_dataset, batch_size)

        return self.fit_generator(train_generator,
                                  valid_generator=valid_generator,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  initial_epoch=initial_epoch,
                                  verbose=verbose,
                                  callbacks=callbacks)

    def fit_generator(self, train_generator, valid_generator=None, epochs=1000, steps_per_epoch=None, validation_steps=None, initial_epoch=1, verbose=True, callbacks=[]):
        """
        Trains the model on a dataset using a generator.

        Args:
            train_generator: Generator-like object for the training dataset.
                The generator must yield a tuple ``(x, y)`` where ``x`` is a
                batch of the training dataset and ``y`` is the corresponding
                ground truths. ``y`` should be a Tensor with the first dimension
                being the batch size since ``len(y)`` is taken as the batch
                size. The loss and the metrics are averaged using this batch
                size. If ``y`` is not a Tensor, then a warning is raised and
                the "batch size" defaults to 1.

                If the generator does not have a method ``__len__()``, the
                ``steps_per_epoch`` argument must be provided. Notice that a
                generator made using the python keyword ``yield`` does not
                have such method. However, a PyTorch DataLoader object has a
                such method.

                Before each epoch, the method ``__iter__()`` on the generator is
                called and the method ``__next__()`` is called for each step on
                resulting object returned by ``__iter__()``. Notice that a call
                to ``__iter__()`` on a generator made using the python keyword
                ``yield`` returns the generator itself.
            valid_generator (optional): Generator-like object for the
                validation dataset. This generator is optional. The generator is
                used the same way as the  generator ``train_generator``. If the
                generator does not have a method ``__len__()``, either the
                ``validation_steps`` or the ``steps_per_epoch`` argument must be
                provided. (Default value = None)
            epochs (int): Number of times the entire training dataset is seen.
                (Default value = 1000)
            steps_per_epoch (int, optional): Number of batch used during one
                epoch. Obviously, using this argument may cause one epoch not to
                see the entire training dataset or see it multiple times.
                (Defaults the number of steps needed to see the entire
                training dataset)
            validation_steps (int, optional): Same as for ``steps_per_epoch``
                but for the validation dataset. (Defaults to ``steps_per_epoch``
                if provided or the number of steps needed to see the entire
                validation dataset)
            initial_epoch (int, optional): Epoch at which to start training
                (useful for resuming a previous training run).
                (Default value = 1)
            verbose (bool): Whether to display the progress of the training.
                (Default value = True)
            callbacks (list of pytoune.framework.Callback): List of callbacks
                that will be called during training. (Default value = [])

        Returns:
            List of dict containing the history of each epoch.

        Example::
            .. code-block:: python

                model = Model(pytorch_module, optimizer, loss_function)
                history = model.fit_generator(train_generator,
                                              valid_generator,
                                              epochs=num_epochs,
                                              verbose=False)
                print(*history, sep="\\n")

            .. code-block:: python

                {'epoch': 1, 'loss': 0.4048105351626873, 'val_loss': 0.35831213593482969}
                {'epoch': 2, 'loss': 0.27947457544505594, 'val_loss': 0.25963697880506514}
                {'epoch': 3, 'loss': 0.20913131050765515, 'val_loss': 0.20263003259897233}
                {'epoch': 4, 'loss': 0.1695468619465828, 'val_loss': 0.16932785511016846}
                {'epoch': 5, 'loss': 0.14716986808925867, 'val_loss': 0.14958457425236701}
                {'epoch': 6, 'loss': 0.13442192394286395, 'val_loss': 0.13765123039484023}
                {'epoch': 7, 'loss': 0.12706457944586874, 'val_loss': 0.13025617450475693}
                {'epoch': 8, 'loss': 0.12272704998031259, 'val_loss': 0.12552770525217055}
                {'epoch': 9, 'loss': 0.12008308945223689, 'val_loss': 0.12238729819655418}
                {'epoch': 10, 'loss': 0.11839052420109511, 'val_loss': 0.12020810544490815}
        """
        if verbose:
            callbacks = [ProgressionCallback()] + callbacks
        callback_list = CallbackList(callbacks)
        callback_list.set_model(self)

        if validation_steps is None:
            if hasattr(valid_generator, '__len__'):
                validation_steps = len(valid_generator)
            elif steps_per_epoch is not None:
                validation_steps = steps_per_epoch
            else:
                raise ValueError("Invalid 'validation_steps' value. Either a value for 'validation_steps' or 'steps_per_epoch' must be provided, or 'valid_generator' must provide a '__len__' method.")
        if steps_per_epoch is None:
            steps_per_epoch = len(train_generator)
        params = {'epochs': epochs, 'steps': steps_per_epoch}
        callback_list.set_params(params)

        epoch_logs = []
        self.stop_training = False
        callback_list.on_train_begin({})
        for epoch in range(initial_epoch, epochs + 1):
            callback_list.on_epoch_begin(epoch, {})
            losses_sum = 0.
            metrics_sum = np.zeros(len(self.metrics))
            sizes_sum = 0.

            self.model.train(True)
            train_iterator = iter(train_generator)
            for step in range(1, steps_per_epoch + 1):
                callback_list.on_batch_begin(step, {})

                self.model.zero_grad()

                x, y = next(train_iterator)
                loss_tensor, metrics_tensors = self._compute_loss_and_metrics(x, y)

                loss_tensor.backward()
                self.optimizer.step()

                loss, metrics = self._loss_and_metrics_tensors_to_numpy(loss_tensor, metrics_tensors)
                size = self._get_batch_size(y)
                losses_sum += loss * size
                metrics_sum += metrics * size
                sizes_sum += size

                metrics_dict = dict(zip(self.metrics_names, metrics))
                batch_logs = {'batch': step, 'size': size, 'loss': loss, **metrics_dict}
                callback_list.on_batch_end(step, batch_logs)

            val_dict = {}
            if valid_generator is not None:
                self.model.eval()
                val_loss, val_metrics = self._validate(valid_generator, validation_steps)
                val_metrics_dict = {'val_' + metric_name:metric for metric_name, metric in zip(self.metrics_names, val_metrics)}
                val_dict = {'val_loss': val_loss, **val_metrics_dict}

            losses_mean = losses_sum / sizes_sum
            metrics_mean = metrics_sum / sizes_sum
            metrics_dict = dict(zip(self.metrics_names, metrics_mean))
            epoch_log = {'epoch': epoch, 'loss': losses_mean, **metrics_dict, **val_dict}
            callback_list.on_epoch_end(epoch, epoch_log)

            epoch_logs.append(epoch_log)

            if self.stop_training:
                break

        callback_list.on_train_end({})

        return epoch_logs

    def predict(self, x, batch_size=32):
        """
        Returns the predictions of the network given a dataset ``x``, where the
        torch variables are converted into numpy arrays.

        Args:
            x (Union[Tensor, np.ndarray])): Dataset for which to predict.
            batch_size (int): Number of samples given to the network at one
                time. (Default value = 32)

        Returns:
            Numpy arrays of the predictions.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        generator = DataLoader(x, batch_size)
        pred_y = self.predict_generator(generator)
        return np.concatenate(pred_y)

    def predict_generator(self, generator, steps=None):
        """
        Returns the predictions of the network given a batch of samples ``x``,
        where the torch variables are converted into numpy arrays.

        generator: Generator-like object for the dataset. The generator must
            yield a tuple a batch of samples.

            If the generator does not have a method ``__len__()``, the
            ``steps`` argument must be provided. Notice that a
            generator made using the python keyword ``yield`` does not
            have such method. However, a PyTorch DataLoader object has a
            such method.

            The method ``__iter__()`` on the generator is called and the
            method ``__next__()`` is called for each step on resulting
            object returned by ``__iter__()``. Notice that a call to
            ``__iter__()`` on a generator made using the python keyword
            ``yield`` returns the generator itself.
        steps (int, optional): Number of iterations done on
            ``generator``. (Defaults the number of steps needed to see the
            entire dataset)

        Returns:
            List of the predictions of each batch with torch variables
            converted into numpy arrays.
        """
        self.model.eval()
        if steps is None:
            steps = len(generator)
        pred_y = []
        iterator = iter(generator)
        for _ in range(steps):
            x = next(iterator)
            x = tensors_to_variables(x, volatile=True)
            pred_y.append(torch_to_numpy(self.model(x)))
        return pred_y


    def evaluate(self, x, y, batch_size=32, return_pred=False):
        """
        Computes the loss and the metrics of the network on a batch of samples
        and optionaly returns the predictions.

        Args:
            x (Union[Tensor, np.ndarray])): Dataset tensor.
            y (Union[Tensor, np.ndarray])): Dataset ground truths.
            batch_size (int): Number of samples given to the network at one
                time. (Default value = 32)
            return_pred (bool, optional): Whether to return the predictions for
                ``x``. (Default value = False)

        Returns:
            Tuple ``(loss, metrics)``. ``loss`` is a float and
            ``metrics`` is a numpy array of size ``n``, where ``n`` is the
            number of metrics. If ``return_pred`` is true, then this method
            returns a tuple ``(loss, metrics, pred_y)`` where ``pred_y`` is a
            numpy arrays of the predictions.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        dataset = TensorDataset(x, y)
        generator = DataLoader(dataset, batch_size)
        loss, metrics, *pred_y = self.evaluate_generator(generator, len(generator), return_pred=return_pred)
        ret = (loss, metrics)
        if return_pred:
            pred_y = np.concatenate(pred_y[0])
            ret += (pred_y,)
        return ret


    def evaluate_generator(self, generator, steps=None, return_pred=False):
        """
        Computes the loss and the metrics of the network on a batch of samples
        and optionaly returns the predictions.

        Args:
            generator: Generator-like object for the dataset. The generator
                must yield a tuple ``(x, y)`` where ``x`` is a batch of the
                dataset and ``y`` is the corresponding ground truths. ``y``
                should be a Tensor with the first dimension being the batch size
                since ``len(y)`` is taken as the batch size. The loss and the
                metrics are averaged using this batch size. If ``y`` is not a
                Tensor, then a warning is raised and the "batch size" defaults
                to 1.

                If the generator does not have a method ``__len__()``, the
                ``steps`` argument must be provided. Notice that a
                generator made using the python keyword ``yield`` does not
                have such method. However, a PyTorch DataLoader object has a
                such method.

                The method ``__iter__()`` on the generator is called and the
                method ``__next__()`` is called for each step on resulting
                object returned by ``__iter__()``. Notice that a call to
                ``__iter__()`` on a generator made using the python keyword
                ``yield`` returns the generator itself.
            steps (int, optional): Number of iterations done on
                ``generator``. (Defaults the number of steps needed to see the
                entire dataset)
            return_pred (bool, optional): Whether to return the predictions for
                ``x``. (Default value = False)

        Returns:
            Tuple ``(loss, metrics)``. ``loss`` is a float and
            ``metrics`` is a numpy array of size ``n``, where ``n`` is the
            number of metrics. If ``return_pred`` is true, then this method
            returns a tuple ``(loss, metrics, pred_y)`` where ``pred_y`` is the
            list of the predictions of each batch with torch variables converted
            into numpy arrays.
        """
        self.model.eval()
        if steps is None:
            steps = len(generator)
        return self._validate(generator, steps, return_pred=return_pred)


    def _validate(self, valid_generator, validation_steps, return_pred=False):
        losses_sum = 0.
        metrics_sum = np.zeros(len(self.metrics))
        sizes_sum = 0
        if return_pred:
            pred_list = []

        valid_iterator = iter(valid_generator)
        for step in range(validation_steps):
            x, y = next(valid_iterator)
            if return_pred:
                loss_tensor, metrics_tensors, pred_y = self._compute_loss_and_metrics(x, y, return_pred=True)
                loss, metrics, pred_y = self._loss_and_metrics_tensors_to_numpy(loss_tensor, metrics_tensors, pred_y)
                pred_list.append(pred_y)
            else:
                loss_tensor, metrics_tensors = self._compute_loss_and_metrics(x, y, return_pred=False)
                loss, metrics = self._loss_and_metrics_tensors_to_numpy(loss_tensor, metrics_tensors)

            size = self._get_batch_size(y)
            losses_sum += loss * size
            metrics_sum += metrics * size
            sizes_sum += size

        loss_mean = losses_sum / sizes_sum
        metrics_mean = metrics_sum / sizes_sum
        ret = (loss_mean, metrics_mean)
        if return_pred:
            ret += (pred_list,)
        return ret

    def _compute_loss_and_metrics(self, x, y, return_pred=False):
        x = tensors_to_variables(x, volatile=not self.model.training)
        y = tensors_to_variables(y, volatile=not self.model.training)
        pred_y = self.model(x)
        loss_tensor = self.loss_function(pred_y, y)
        metrics_tensors = self._compute_metrics(pred_y, y)
        ret = (loss_tensor, metrics_tensors)
        if return_pred:
            ret = ret + (pred_y,)
        return ret

    def _compute_metrics(self, pred_y, y):
        return [metric(pred_y, y) for metric in self.metrics]

    def _loss_and_metrics_tensors_to_numpy(self, loss_tensor, metrics_tensors, pred_y=None):
        loss = float(loss_tensor)
        metrics = np.array([float(metric_tensor) for metric_tensor in metrics_tensors])
        ret = (loss, metrics)
        if pred_y is not None:
            pred_y = torch_to_numpy(pred_y)
            ret = ret + (pred_y,)
        return ret

    def _get_batch_size(self, y):
        if torch.is_tensor(y) or isinstance(y, Variable):
            return len(y)
        else:
            warnings.warn("When 'y' is not a tensor, the batch size is set to 1 and, thus, the computed loss and metrics at the end of each epoch is the mean of the batches' losses and metrics.")
            return 1

    def load_weights(self, f):
        """
        Loads the weights saved using the ``torch.save()`` method or the
        ``save_weights()`` method of this class.

        Args:
            f: File-like object (has to implement fileno that returns a file
            descriptor) or string containing a file name.
        """
        self.set_weights(torch.load(f))

    def save_weights(self, f):
        """
        Saves the weights of the current network.

        Args:
            f: File-like object (has to implement fileno that returns a file
            descriptor) or string containing a file name.
        """
        torch.save(self.model.state_dict(), f)

    def get_weights(self):
        """
        Returns a dictionary containing the parameters of the network. The
        tensors are just references to the parameters. To get copies of the
        weights, see the ``get_weight_copies()`` method.
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
            weights (dict): Weights returned by either ``get_weights()`` or
                ``get_weight_copies()``.
        """
        self.model.load_state_dict(weights)

    def cuda(self, *args, **kwargs):
        """
        Tranfers the network on the GPU. The arguments are passed to the
        ``torch.nn.Module.cuda()`` method. Notice that the method
        ``torch.Tensor.cuda()`` must be called separately on the tensors given
        to the network.

        Returns:
            The return of ``torch.nn.Module.cuda()``.
        """
        return self.model.cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs):
        """
        Tranfers the network on the CPU. The arguments are passed to the
        ``torch.nn.Module.cpu()`` method. Notice that the method
        ``torch.Tensor.cpu()`` must be called separately on the tensors given
        to the network.

        Returns:
            The return of ``torch.nn.Module.cpu()``.
        """
        return self.model.cpu(*args, **kwargs)
