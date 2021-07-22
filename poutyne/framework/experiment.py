# pylint: disable=too-many-lines
import os
import warnings
from typing import Union, Callable, List, Dict, Tuple, Any

try:
    import pandas as pd
except ImportError:
    pd = None
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from . import Model
from ..utils import set_seeds
from .callbacks import ModelCheckpoint, \
    OptimizerCheckpoint, \
    LRSchedulerCheckpoint, \
    PeriodicSaveLambda, \
    AtomicCSVLogger, \
    TensorBoardLogger, \
    BestModelRestore


class Experiment:
    """
    The Experiment class provides a straightforward experimentation tool for efficient and entirely
    customizable finetuning of the whole neural network training procedure with PyTorch. The
    ``Experiment`` object takes care of the training and testing processes while also managing to
    keep traces of all pertinent information via the automatic logging option.

    Args:
        directory (str): Path to the experiment's working directory. Will be used for automatic logging.
        network (torch.nn.Module): A PyTorch network.
        device (Union[torch.torch.device, List[torch.torch.device], str, None]): The device to which the model is sent
            or for multi-GPUs, the list of devices to which the model is to be sent. When using a string for a multiple
            GPUs, the option is "all", for "take them all." By default, the current device is used as the main one.
            If None, the model will be kept on its current device.
            (Default value = None)
        logging (bool): Whether or not to log the experiment's progress. If true, various logging
            callbacks will be inserted to output training and testing stats as well as to save model checkpoints,
            for example, automatically. See :func:`~Experiment.train()` and :func:`~Experiment.test()` for more details.
            (Default value = True)
        optimizer (Union[torch.optim.Optimizer, str]): If Pytorch Optimizer, must already be initialized.
            If str, should be the optimizer's name in Pytorch (i.e. 'Adam' for torch.optim.Adam).
            (Default value = 'sgd')
        loss_function(Union[Callable, str], optional) It can be any PyTorch
            loss layer or custom loss function. It can also be a string with the same name as a PyTorch
            loss function (either the functional or object name). The loss function must have the signature
            ``loss_function(input, target)`` where ``input`` is the prediction of the network and ``target``
            is the ground truth. If ``None``, will default to, in priority order, either the model's own
            loss function or the default loss function associated with the ``task``.
            (Default value = None)
        batch_metrics (List, optional): List of functions with the same signature as the loss function. Each metric
            can be any PyTorch loss function. It can also be a string with the same name as a PyTorch
            loss function (either the functional or object name). 'accuracy' (or just 'acc') is also a
            valid metric. Each metric function is called on each batch of the optimization and on the
            validation batches at the end of the epoch.
            (Default value = None)
        epoch_metrics (List, optional): List of functions with the same signature as
            :class:`~poutyne.EpochMetric`
            (Default value = None)
        monitoring (bool): Whether or not to monitor the training. If True will track the best epoch.
            If False, ``monitor_metric`` and ``monitor_mode`` are not used, and when testing, the last epoch is used to
            test the model instead of the best epoch.
            (Default value = True)
        monitor_metric (str, optional): Which metric to consider for best model performance calculation. Should be in
            the format '{metric_name}' or 'val_{metric_name}' (i.e. 'val_loss'). If None, will follow the value
            suggested by ``task`` or default to 'val_loss'. If ``monitoring`` is set to False, will be ignore.

            .. warning:: If you do not plan on using a validation set, you must set the monitor metric to another
                value.
        monitor_mode (str, optional): Which mode, either 'min' or 'max', should be used when considering the
            ``monitor_metric`` value. If None, will follow the value suggested by ``task`` or default to 'min'.
            If ``monitoring`` is set to False, will be ignore.
        task (str, optional): Any str beginning with either 'classif' or 'reg'. Specifying a ``task``
            can assign default values to the ``loss_function``, ``batch_metrics``, ``monitor_mode`` and
            ``monitor_mode``. For ``task`` that begins with 'reg', the only default value is the loss function
            that is the mean squared error. When beginning with 'classif', the default loss function is the
            cross-entropy loss. The default batch metrics will be the accuracy, the default epoch metrics will be
            the F1 score and the default monitoring will be set on 'val_acc' with a 'max' mode.
            (Default value = None)

    Examples:
        Using a PyTorch DataLoader, on classification task with SGD optimizer::

            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from poutyne import Experiment

            num_features = 20
            num_classes = 5

            # Our training dataset with 800 samples.
            num_train_samples = 800
            train_x = torch.rand(num_train_samples, num_features)
            train_y = torch.randint(num_classes, (num_train_samples, ), dtype=torch.long)
            train_dataset = TensorDataset(train_x, train_y)
            train_generator = DataLoader(train_dataset, batch_size=32)

            # Our validation dataset with 200 samples.
            num_valid_samples = 200
            valid_x = torch.rand(num_valid_samples, num_features)
            valid_y = torch.randint(num_classes, (num_valid_samples, ), dtype=torch.long)
            valid_dataset = TensorDataset(valid_x, valid_y)
            valid_generator = DataLoader(valid_dataset, batch_size=32)

            # Our network
            pytorch_network = torch.nn.Linear(num_features, num_train_samples)

            # Initialization of our experimentation and network training
            exp = Experiment('./simple_example',
                             pytorch_network,
                             optimizer='sgd',
                             task='classif')
            exp.train(train_generator, valid_generator, epochs=5)

    The above code will yield an output similar to the below lines. Note the automatic checkpoint saving
    in the experiment directory when the monitored metric improved.

    .. code-block:: none

            Epoch 1/5 0.09s Step 25/25: loss: 6.351375, acc: 1.375000, val_loss: 6.236106, val_acc: 5.000000
            Epoch 1: val_acc improved from -inf to 5.00000, saving file to ./simple_example/checkpoint_epoch_1.ckpt
            Epoch 2/5 0.10s Step 25/25: loss: 6.054254, acc: 14.000000, val_loss: 5.944495, val_acc: 19.500000
            Epoch 2: val_acc improved from 5.00000 to 19.50000, saving file to ./simple_example/checkpoint_epoch_2.ckpt
            Epoch 3/5 0.09s Step 25/25: loss: 5.759377, acc: 22.875000, val_loss: 5.655412, val_acc: 21.000000
            Epoch 3: val_acc improved from 19.50000 to 21.00000, saving file to ./simple_example/checkpoint_epoch_3.ckpt
            ...

    Training can now easily be resumed from the best checkpoint::

            exp.train(train_generator, valid_generator, epochs=10)

    .. code-block:: none

            Restoring model from ./simple_example/checkpoint_epoch_3.ckpt
            Loading weights from ./simple_example/checkpoint.ckpt and starting at epoch 6.
            Loading optimizer state from ./simple_example/checkpoint.optim and starting at epoch 6.
            Epoch 6/10 0.16s Step 25/25: loss: 4.897135, acc: 22.875000, val_loss: 4.813141, val_acc: 20.500000
            Epoch 7/10 0.10s Step 25/25: loss: 4.621514, acc: 22.625000, val_loss: 4.545359, val_acc: 20.500000
            Epoch 8/10 0.24s Step 25/25: loss: 4.354721, acc: 23.625000, val_loss: 4.287117, val_acc: 20.500000
            ...

    Testing is also very intuitive::

            exp.test(test_generator)

    .. code-block:: none

            Restoring model from ./simple_example/checkpoint_epoch_9.ckpt
            Found best checkpoint at epoch: 9
            lr: 0.01, loss: 4.09892, acc: 23.625, val_loss: 4.04057, val_acc: 21.5
            On best model: test_loss: 4.06664, test_acc: 17.5


    Finally, all the pertinent metrics specified to the Experiment at each epoch are stored in a specific logging
    file, found here at './simple_example/log.tsv'.

    .. code-block:: none

            epoch	time	            lr	    loss	            acc	    val_loss	        val_acc
            1	    0.0721172170015052	0.01	6.351375141143799	1.375	6.23610631942749	5.0
            2	    0.0298177790245972	0.01	6.054253826141357	14.000	5.94449516296386	19.5
            3	    0.0637106419890187	0.01	5.759376544952392	22.875	5.65541223526001	21.0
            ...

    Also, we could use more than one GPU (on a single node) by using the device argument

    .. code-block:: none

            # Initialization of our experimentation and network training
            exp = Experiment('./simple_example',
                             pytorch_network,
                             optimizer='sgd',
                             task='classif',
                             device="all")
            exp.train(train_generator, valid_generator, epochs=5)

    """
    BEST_CHECKPOINT_FILENAME = 'checkpoint_epoch_{epoch}.ckpt'
    MODEL_CHECKPOINT_FILENAME = 'checkpoint.ckpt'
    OPTIMIZER_CHECKPOINT_FILENAME = 'checkpoint.optim'
    LOG_FILENAME = 'log.tsv'
    TENSORBOARD_DIRECTORY = 'tensorboard'
    EPOCH_FILENAME = 'last.epoch'
    LR_SCHEDULER_FILENAME = 'lr_sched_%d.lrsched'
    TEST_LOG_FILENAME = '{name}_log.tsv'

    def __init__(self,
                 directory: str,
                 network: torch.nn.Module,
                 *,
                 device: Union[torch.device, List[torch.device], List[str], None, str] = None,
                 logging: bool = True,
                 optimizer: Union[torch.optim.Optimizer, str] = 'sgd',
                 loss_function: Union[Callable, str] = None,
                 batch_metrics: Union[List, None] = None,
                 epoch_metrics: Union[List, None] = None,
                 monitoring: bool = True,
                 monitor_metric: Union[str, None] = None,
                 monitor_mode: Union[str, None] = None,
                 task: Union[str, None] = None) -> None:
        self.directory = directory
        self.logging = logging

        if task is not None and not task.startswith('classif') and not task.startswith('reg'):
            raise ValueError("Invalid task '%s'" % task)

        batch_metrics = [] if batch_metrics is None else batch_metrics
        epoch_metrics = [] if epoch_metrics is None else epoch_metrics

        loss_function = self._get_loss_function(loss_function, network, task)
        batch_metrics = self._get_batch_metrics(batch_metrics, network, task)
        epoch_metrics = self._get_epoch_metrics(epoch_metrics, network, task)

        self.monitoring = monitoring
        self.monitor_metric = None
        self.monitor_mode = None
        if self.monitoring:
            self._set_monitor(monitor_metric, monitor_mode, task)

        self.model = Model(network,
                           optimizer,
                           loss_function,
                           batch_metrics=batch_metrics,
                           epoch_metrics=epoch_metrics,
                           device=device)

        self.best_checkpoint_filename = self.get_path(Experiment.BEST_CHECKPOINT_FILENAME)
        self.model_checkpoint_filename = self.get_path(Experiment.MODEL_CHECKPOINT_FILENAME)
        self.optimizer_checkpoint_filename = self.get_path(Experiment.OPTIMIZER_CHECKPOINT_FILENAME)
        self.log_filename = self.get_path(Experiment.LOG_FILENAME)
        self.tensorboard_directory = self.get_path(Experiment.TENSORBOARD_DIRECTORY)
        self.epoch_filename = self.get_path(Experiment.EPOCH_FILENAME)
        self.lr_scheduler_filename = self.get_path(Experiment.LR_SCHEDULER_FILENAME)
        self.test_log_filename = self.get_path(Experiment.TEST_LOG_FILENAME)

    def get_path(self, *paths: str) -> str:
        """
        Returns the path inside the experiment directory.
        """
        return os.path.join(self.directory, *paths)

    def _get_loss_function(self, loss_function: Union[Callable, str], network: torch.nn.Module,
                           task: Union[str, None]) -> Union[Callable, str]:
        if loss_function is None:
            if hasattr(network, 'loss_function'):
                return network.loss_function
            if task is not None:
                if task.startswith('classif'):
                    return 'cross_entropy'
                if task.startswith('reg'):
                    return 'mse'
        return loss_function

    def _get_batch_metrics(self, batch_metrics: Union[List, None], network: torch.nn.Module,
                           task: Union[str, None]) -> Union[List, None]:
        if batch_metrics is None or len(batch_metrics) == 0:
            if hasattr(network, 'batch_metrics'):
                return network.batch_metrics
            if task is not None and task.startswith('classif'):
                return ['accuracy']
        return batch_metrics

    def _get_epoch_metrics(self, epoch_metrics: Union[List, None], network, task: Union[str,
                                                                                        None]) -> Union[List, None]:
        if epoch_metrics is None or len(epoch_metrics) == 0:
            if hasattr(network, 'epoch_metrics'):
                return network.epoch_metrics
            if task is not None and task.startswith('classif'):
                return ['f1']
        return epoch_metrics

    def _set_monitor(self, monitor_metric: Union[str, None], monitor_mode: Union[str, None], task: Union[str,
                                                                                                         None]) -> None:
        if monitor_mode is not None and monitor_mode not in ['min', 'max']:
            raise ValueError("Invalid mode '%s'" % monitor_mode)

        self.monitor_metric = 'val_loss'
        self.monitor_mode = 'min'
        if monitor_metric is not None:
            self.monitor_metric = monitor_metric
            if monitor_mode is not None:
                self.monitor_mode = monitor_mode
        elif task is not None and task.startswith('classif'):
            self.monitor_metric = 'val_acc'
            self.monitor_mode = 'max'

    def get_best_epoch_stats(self) -> Dict:
        """
        Returns all computed statistics corresponding to the best epoch according to the
        ``monitor_metric`` and ``monitor_mode`` attributes.

        Returns:
            dict where each key is a column name in the logging output file
            and values are the ones found at the best epoch.
        """
        if not self.monitoring:
            raise ValueError("Monitoring was disabled. Cannot get best epoch.")

        if pd is None:
            raise ImportError("pandas needs to be installed to use this function.")

        history = pd.read_csv(self.log_filename, sep='\t')
        if self.monitor_mode == 'min':
            best_epoch_index = history[self.monitor_metric].idxmin()
        else:
            best_epoch_index = history[self.monitor_metric].idxmax()
        return history.iloc[best_epoch_index:best_epoch_index + 1]

    def get_saved_epochs(self):
        """
        Returns a pandas DataFrame which each row corresponds to an epoch having
        a saved checkpoint.

        Returns:
            pandas DataFrame which each row corresponds to an epoch having a saved
            checkpoint.
        """
        if not self.monitoring:
            raise ValueError("Monitoring was disabled. Except the last epoch, no epoch checkpoint were saved.")

        if pd is None:
            raise ImportError("pandas needs to be installed to use this function.")

        history = pd.read_csv(self.log_filename, sep='\t')
        metrics = history[self.monitor_metric].tolist()
        if self.monitor_mode == 'min':
            monitor_op = lambda x, y: x < y
            current_best = float('Inf')
        elif self.monitor_mode == 'max':
            monitor_op = lambda x, y: x > y
            current_best = -float('Inf')
        saved_epoch_indices = []
        for i, metric in enumerate(metrics):
            if monitor_op(metric, current_best):
                current_best = metric
                saved_epoch_indices.append(i)
        return history.iloc[saved_epoch_indices]

    def _warn_missing_file(self, filename: str) -> None:
        warnings.warn("Missing checkpoint: %s." % filename)

    def _load_epoch_state(self, lr_schedulers: List) -> int:
        # pylint: disable=broad-except
        initial_epoch = 1
        if os.path.isfile(self.epoch_filename):
            try:
                with open(self.epoch_filename, 'r') as f:
                    initial_epoch = int(f.read()) + 1
            except Exception as e:
                print(e)
            if os.path.isfile(self.model_checkpoint_filename):
                try:
                    print("Loading weights from %s and starting at epoch %d." %
                          (self.model_checkpoint_filename, initial_epoch))
                    self.model.load_weights(self.model_checkpoint_filename)
                except Exception as e:
                    print(e)
            else:
                self._warn_missing_file(self.model_checkpoint_filename)
            if os.path.isfile(self.optimizer_checkpoint_filename):
                try:
                    print("Loading optimizer state from %s and starting at epoch %d." %
                          (self.optimizer_checkpoint_filename, initial_epoch))
                    self.model.load_optimizer_state(self.optimizer_checkpoint_filename)
                except Exception as e:
                    print(e)
            else:
                self._warn_missing_file(self.optimizer_checkpoint_filename)
            for i, lr_scheduler in enumerate(lr_schedulers):
                filename = self.lr_scheduler_filename % i
                if os.path.isfile(filename):
                    try:
                        print("Loading LR scheduler state from %s and starting at epoch %d." %
                              (filename, initial_epoch))
                        lr_scheduler.load_state(filename)
                    except Exception as e:
                        print(e)
                else:
                    self._warn_missing_file(filename)
        return initial_epoch

    def _init_model_restoring_callbacks(self, initial_epoch: int, keep_only_last_best: bool,
                                        save_every_epoch: bool) -> List:
        callbacks = []
        best_checkpoint = ModelCheckpoint(self.best_checkpoint_filename,
                                          monitor=self.monitor_metric,
                                          mode=self.monitor_mode,
                                          keep_only_last_best=keep_only_last_best,
                                          save_best_only=not save_every_epoch,
                                          restore_best=not save_every_epoch,
                                          verbose=not save_every_epoch)
        callbacks.append(best_checkpoint)

        if save_every_epoch:
            best_restore = BestModelRestore(monitor=self.monitor_metric, mode=self.monitor_mode, verbose=True)
            callbacks.append(best_restore)

        if initial_epoch > 1:
            # We set the current best metric score in the ModelCheckpoint so that
            # it does not save checkpoint it would not have saved if the
            # optimization was not stopped.
            best_epoch_stats = self.get_best_epoch_stats()
            best_epoch = best_epoch_stats['epoch'].item()
            best_filename = self.best_checkpoint_filename.format(epoch=best_epoch)
            if not save_every_epoch:
                best_checkpoint.best_filename = best_filename
                best_checkpoint.current_best = best_epoch_stats[self.monitor_metric].item()
            else:
                best_restore.best_weights = torch.load(best_filename, map_location='cpu')
                best_restore.current_best = best_epoch_stats[self.monitor_metric].item()

        return callbacks

    def _init_tensorboard_callbacks(self, disable_tensorboard: bool) -> Tuple:
        tensorboard_writer = None
        callbacks = []
        if not disable_tensorboard:
            if SummaryWriter is None:
                warnings.warn(
                    "tensorboard does not seem to be installed. "
                    "To remove this warning, set the 'disable_tensorboard' "
                    "flag to True or install tensorboard.",
                    stacklevel=3)
            else:
                tensorboard_writer = SummaryWriter(self.tensorboard_directory)
                callbacks += [TensorBoardLogger(tensorboard_writer)]
        return tensorboard_writer, callbacks

    def _init_lr_scheduler_callbacks(self, lr_schedulers: List) -> List:
        callbacks = []
        if self.logging:
            for i, lr_scheduler in enumerate(lr_schedulers):
                filename = self.lr_scheduler_filename % i
                callbacks += [LRSchedulerCheckpoint(lr_scheduler, filename, verbose=False)]
        else:
            callbacks += lr_schedulers
        return callbacks

    def train(self, train_generator, valid_generator=None, **kwargs) -> List[Dict]:
        """
        Trains or finetunes the model on a dataset using a generator. If a previous training already occurred
        and lasted a total of `n_previous` epochs, then the model's weights will be set to the last checkpoint and the
        training will be resumed for epochs range (`n_previous`, `epochs`].

        If the Experiment has logging enabled (i.e. self.logging is True), numerous callbacks will be automatically
        included. Notably, two :class:`~poutyne.ModelCheckpoint` objects will take care of saving the last and every
        new best (according to monitor mode) model weights in appropriate checkpoint files.
        :class:`~poutyne.OptimizerCheckpoint` and :class:`~poutyne.LRSchedulerCheckpoint` will also respectively
        handle the saving of the optimizer and LR scheduler's respective states for future retrieval. Moreover, a
        :class:`~poutyne.AtomicCSVLogger` will save all available epoch statistics in an output .tsv file. Lastly, a
        :class:`~poutyne.TensorBoardLogger` handles automatic TensorBoard logging of various neural network
        statistics.

        .. warning:: With **Jupyter Notebooks in Firefox**, if ``colorama`` is installed and colors are enabled (as it
            is by default), a great number of epochs and steps per epoch can cause a spike in memory usage in Firefox.
            The problem does not occur in Google Chrome/Chromium. To avoid this problem, you can disable the colors by
            passing ``progress_options={'coloring': False}``. See
            `this Github issue for details <https://github.com/jupyter/notebook/issues/5897>`__.

        Args:
            train_generator: Generator-like object for the training set. See :func:`~Model.fit_generator()`
                for details on the types of generators supported.
            valid_generator (optional): Generator-like object for the validation set. See
                :func:`~Model.fit_generator()` for details on the types of generators supported.
                (Default value = None)
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called during
                training. These callbacks are added after those used in this method (see above). This allows to assume
                that they are called after those.
                (Default value = None)
            lr_schedulers: List of learning rate schedulers. (Default value = None)
            keep_only_last_best (bool): Whether only the last saved best checkpoint is kept. Applies only when
                 `save_every_epoch` is false.
                 (Default value = False)
            save_every_epoch (bool, optional): Whether or not to save the experiment model's weights after
                every epoch.
                (Default value = False)
            disable_tensorboard (bool, optional): Whether or not to disable the automatic tensorboard logging
                callbacks.
                (Default value = False)
            seed (int, optional): Seed used to make the sampling deterministic.
                (Default value = 42)
            kwargs: Any keyword arguments to pass to :func:`~Model.fit_generator()`.

        Returns:
            List of dict containing the history of each epoch.
        """
        return self._train(self.model.fit_generator, train_generator, valid_generator, **kwargs)

    def train_dataset(self, train_dataset, valid_dataset=None, **kwargs) -> List[Dict]:
        """
        Trains or finetunes the model on a dataset. If a previous training already occurred
        and lasted a total of `n_previous` epochs, then the model's weights will be set to the last checkpoint and the
        training will be resumed for epochs range (`n_previous`, `epochs`].

        If the Experiment has logging enabled (i.e. self.logging is True), numerous callbacks will be automatically
        included. Notably, two :class:`~poutyne.ModelCheckpoint` objects will take care of saving the last and every
        new best (according to monitor mode) model weights in appropriate checkpoint files.
        :class:`~poutyne.OptimizerCheckpoint` and :class:`~poutyne.LRSchedulerCheckpoint` will also respectively
        handle the saving of the optimizer and LR scheduler's respective states for future retrieval. Moreover, a
        :class:`~poutyne.AtomicCSVLogger` will save all available epoch statistics in an output .tsv file. Lastly, a
        :class:`~poutyne.TensorBoardLogger` handles automatic TensorBoard logging of various neural network
        statistics.

        Args:
            train_dataset (~torch.utils.data.Dataset): Training dataset.
            valid_dataset (~torch.utils.data.Dataset): Validation dataset.
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called during
                training. These callbacks are added after those used in this method (see above). This allows to assume
                that they are called after those.
                (Default value = None)
            lr_schedulers: List of learning rate schedulers. (Default value = None)
            keep_only_last_best (bool): Whether only the last saved best checkpoint is kept. Applies only when
                 `save_every_epoch` is false.
                 (Default value = False)
            save_every_epoch (bool, optional): Whether or not to save the experiment model's weights after
                every epoch.
                (Default value = False)
            disable_tensorboard (bool, optional): Whether or not to disable the automatic tensorboard logging
                callbacks.
                (Default value = False)
            seed (int, optional): Seed used to make the sampling deterministic.
                (Default value = 42)
            kwargs: Any keyword arguments to pass to :func:`~Model.fit_dataset()`.

        Returns:
            List of dict containing the history of each epoch.
        """
        return self._train(self.model.fit_dataset, train_dataset, valid_dataset, **kwargs)

    def train_data(self, x, y, validation_data=None, **kwargs) -> List[Dict]:
        """
        Trains or finetunes the model on data under the form of NumPy arrays or torch tensors. If a previous
        training already occurred and lasted a total of `n_previous` epochs, then the model's weights will be set to the
        last checkpoint and the training will be resumed for epochs range (`n_previous`, `epochs`].

        If the Experiment has logging enabled (i.e. self.logging is True), numerous callbacks will be automatically
        included. Notably, two :class:`~poutyne.ModelCheckpoint` objects will take care of saving the last and every
        new best (according to monitor mode) model weights in appropriate checkpoint files.
        :class:`~poutyne.OptimizerCheckpoint` and :class:`~poutyne.LRSchedulerCheckpoint` will also respectively
        handle the saving of the optimizer and LR scheduler's respective states for future retrieval. Moreover, a
        :class:`~poutyne.AtomicCSVLogger` will save all available epoch statistics in an output .tsv file. Lastly, a
        :class:`~poutyne.TensorBoardLogger` handles automatic TensorBoard logging of various neural network
        statistics.

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
            callbacks (List[~poutyne.Callback]): List of callbacks that will be called during
                training. These callbacks are added after those used in this method (see above). This allows to assume
                that they are called after those.
                (Default value = None)
            lr_schedulers: List of learning rate schedulers. (Default value = None)
            keep_only_last_best (bool): Whether only the last saved best checkpoint is kept. Applies only when
                 `save_every_epoch` is false.
                 (Default value = False)
            save_every_epoch (bool, optional): Whether or not to save the experiment model's weights after
                every epoch.
                (Default value = False)
            disable_tensorboard (bool, optional): Whether or not to disable the automatic tensorboard logging
                callbacks.
                (Default value = False)
            seed (int, optional): Seed used to make the sampling deterministic.
                (Default value = 42)
            kwargs: Any keyword arguments to pass to :func:`~Model.fit()`.

        Returns:
            List of dict containing the history of each epoch.
        """
        return self._train(self.model.fit, x, y, validation_data, **kwargs)

    def _train(self,
               training_func,
               *args,
               callbacks: Union[List, None] = None,
               lr_schedulers: Union[List, None] = None,
               keep_only_last_best: bool = False,
               save_every_epoch: bool = False,
               disable_tensorboard: bool = False,
               seed: int = 42,
               **kwargs) -> List[Dict]:
        set_seeds(seed)

        lr_schedulers = [] if lr_schedulers is None else lr_schedulers

        expt_callbacks = []

        tensorboard_writer = None
        initial_epoch = 1
        if self.logging:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            # Restarting optimization if needed.
            initial_epoch = self._load_epoch_state(lr_schedulers)

            expt_callbacks += [AtomicCSVLogger(self.log_filename, separator='\t', append=initial_epoch != 1)]

            if self.monitoring:
                expt_callbacks += self._init_model_restoring_callbacks(initial_epoch, keep_only_last_best,
                                                                       save_every_epoch)
            expt_callbacks += [ModelCheckpoint(self.model_checkpoint_filename, verbose=False)]
            expt_callbacks += [OptimizerCheckpoint(self.optimizer_checkpoint_filename, verbose=False)]

            # We save the last epoch number after the end of the epoch so that the
            # _load_epoch_state() knows which epoch to restart the optimization.
            expt_callbacks += [
                PeriodicSaveLambda(lambda fd, epoch, logs: print(epoch, file=fd), self.epoch_filename, open_mode='w')
            ]

            tensorboard_writer, cb_list = self._init_tensorboard_callbacks(disable_tensorboard)
            expt_callbacks += cb_list
        else:
            if self.monitoring:
                expt_callbacks += [BestModelRestore(monitor=self.monitor_metric, mode=self.monitor_mode, verbose=True)]

        # This method returns callbacks that checkpoints the LR scheduler if logging is enabled.
        # Otherwise, it just returns the list of LR schedulers with a BestModelRestore callback.
        expt_callbacks += self._init_lr_scheduler_callbacks(lr_schedulers)

        if callbacks is not None:
            expt_callbacks += callbacks

        try:
            return training_func(*args, initial_epoch=initial_epoch, callbacks=expt_callbacks, **kwargs)
        finally:
            if tensorboard_writer is not None:
                tensorboard_writer.close()

    def load_checkpoint(self,
                        checkpoint: Union[int, str],
                        *,
                        verbose: bool = False,
                        strict: bool = True) -> Union[Dict, None]:
        """
        Loads the model's weights with the weights at a given checkpoint epoch.

        Args:
            checkpoint (Union[int, str]): Which checkpoint to load the model's weights form.

                - If 'best', will load the best weights according to ``monitor_metric`` and ``monitor_mode``.
                - If 'last', will load the last model checkpoint.
                - If int, will load the checkpoint of the specified epoch.
                - If a path (str), will load the model pickled state_dict weights (for instance, saved as
                  ``torch.save(a_pytorch_network.state_dict(), "./a_path.p")``).
            verbose (bool, optional): Whether or not to print the checkpoint filename, and the best epoch
                number and stats when checkpoint is 'best'.
                (Default value = False)

        Returns:
            If checkpoint is 'best', will return the best epoch stats, as per :func:`~get_best_epoch_stats()`,
            if checkpoint is 'last', will return the last epoch stats, if checkpoint is a int, will return the
            epoch number stats, if a path, will return the stats of that specific checkpoint.
            else None.
        """

        epoch_stats = None
        if isinstance(checkpoint, int):
            epoch_stats, incompatible_keys = self._load_epoch_checkpoint(checkpoint, verbose=verbose, strict=strict)
        elif checkpoint == 'best':
            epoch_stats, incompatible_keys = self._load_best_checkpoint(verbose=verbose, strict=strict)
        elif checkpoint == 'last':
            epoch_stats, incompatible_keys = self._load_last_checkpoint(verbose=verbose, strict=strict)
        else:
            incompatible_keys = self._load_path_checkpoint(path=checkpoint, verbose=verbose, strict=strict)

        if len(incompatible_keys.unexpected_keys) > 0:
            warnings.warn('Unexpected key(s): {}.'.format(', '.join('"{}"'.format(k)
                                                                    for k in incompatible_keys.unexpected_keys)),
                          stacklevel=2)
        if len(incompatible_keys.missing_keys) > 0:
            warnings.warn('Missing key(s): {}.'.format(', '.join('"{}"'.format(k)
                                                                 for k in incompatible_keys.missing_keys)),
                          stacklevel=2)

        return epoch_stats

    def _print_epoch_stats(self, epoch_stats):
        metrics_str = ', '.join('%s: %g' % (metric_name, epoch_stats[metric_name].item())
                                for metric_name in epoch_stats.columns[2:])
        print(metrics_str)

    def _load_epoch_checkpoint(self, epoch: int, *, verbose: bool = False, strict: bool = True) -> None:
        ckpt_filename = self.best_checkpoint_filename.format(epoch=epoch)

        history = pd.read_csv(self.log_filename, sep='\t')
        epoch_stats = history.iloc[epoch - 1:epoch]

        if verbose:
            print(f"Loading checkpoint {ckpt_filename}")
            self._print_epoch_stats(epoch_stats)

        if not os.path.isfile(ckpt_filename):
            raise ValueError(f"No checkpoint found for epoch {epoch}")

        return epoch_stats, self.model.load_weights(ckpt_filename, strict=strict)

    def _load_best_checkpoint(self, *, verbose: bool = False, strict: bool = True) -> Dict:
        best_epoch_stats = self.get_best_epoch_stats()
        best_epoch = best_epoch_stats['epoch'].item()

        ckpt_filename = self.best_checkpoint_filename.format(epoch=best_epoch)

        if verbose:
            print(f"Found best checkpoint at epoch: {best_epoch}")
            self._print_epoch_stats(best_epoch_stats)
            print(f"Loading checkpoint {ckpt_filename}")

        if not os.path.isfile(ckpt_filename):
            raise ValueError(f"No checkpoint found for epoch {best_epoch}")

        return best_epoch_stats, self.model.load_weights(ckpt_filename, strict=strict)

    def _load_last_checkpoint(self, *, verbose: bool = False, strict: bool = True) -> None:
        history = pd.read_csv(self.log_filename, sep='\t')
        epoch_stats = history.iloc[-1:]

        if verbose:
            print(f"Loading checkpoint {self.model_checkpoint_filename}")
            self._print_epoch_stats(epoch_stats)

        return epoch_stats, self.model.load_weights(self.model_checkpoint_filename, strict=strict)

    def _load_path_checkpoint(self, path, verbose: bool = False, strict: bool = True) -> None:
        if verbose:
            print(f"Loading checkpoint {path}")

        return self.model.load_weights(path, strict=strict)

    def test(self, test_generator, **kwargs):
        """
        Computes and returns the loss and the metrics of the model on a given test examples
        generator.

        If the Experiment has logging enabled (i.e. self.logging is True), a checkpoint (the best one by default)
        is loaded and test and validation statistics are saved in a specific test output .tsv file. Otherwise, the
        current weights of the network is used for testing and statistics are only shown in the standard output.

        Args:
            test_generator: Generator-like object for the test set. See :func:`~Model.fit_generator()` for
                details on the types of generators supported.
            checkpoint (Union[str, int]): Which model checkpoint weights to load for the test evaluation.

                - If 'best', will load the best weights according to ``monitor_metric`` and ``monitor_mode``.
                - If 'last', will load the last model checkpoint.
                - If int, will load the checkpoint of the specified epoch.
                - If a path (str), will load the model pickled state_dict weights (for instance, saved as
                  ``torch.save(a_pytorch_network.state_dict(), "./a_path.p")``).

                This argument has no effect when logging is disabled. (Default value = 'best')
            seed (int, optional): Seed used to make the sampling deterministic.
                (Default value = 42)
            name (str): Prefix of the test log file. (Default value = 'test')
            kwargs: Any keyword arguments to pass to :func:`~Model.evaluate_generator()`.

        If the Experiment has logging enabled (i.e. self.logging is True), one callback will be automatically
        included to save the test metrics. Moreover, a :class:`~poutyne.AtomicCSVLogger` will save the test
        metrics in an output .tsv file.

        Returns:
            dict sorting of all the test metrics values by their names.
        """
        return self._test(self.model.evaluate_generator, test_generator, **kwargs)

    def test_dataset(self, test_dataset, **kwargs) -> Dict:
        """
        Computes and returns the loss and the metrics of the model on a given test dataset.

        If the Experiment has logging enabled (i.e. self.logging is True), a checkpoint (the best one by default)
        is loaded and test and validation statistics are saved in a specific test output .tsv file. Otherwise, the
        current weights of the network is used for testing and statistics are only shown in the standard output.

        Args:
            test_dataset (~torch.utils.data.Dataset): Test dataset.
            checkpoint (Union[str, int]): Which model checkpoint weights to load for the test evaluation.

                - If 'best', will load the best weights according to ``monitor_metric`` and ``monitor_mode``.
                - If 'last', will load the last model checkpoint.
                - If int, will load the checkpoint of the specified epoch.
                - If a path (str), will load the model pickled state_dict weights (for instance, saved as
                  ``torch.save(a_pytorch_network.state_dict(), "./a_path.p")``).

                This argument has no effect when logging is disabled. (Default value = 'best')
            seed (int, optional): Seed used to make the sampling deterministic.
                (Default value = 42)
            name (str): Prefix of the test log file. (Default value = 'test')
            kwargs: Any keyword arguments to pass to :func:`~Model.evaluate_dataset()`.

        If the Experiment has logging enabled (i.e. self.logging is True), one callback will be automatically
        included to save the test metrics. Moreover, a :class:`~poutyne.AtomicCSVLogger` will save the test
        metrics in an output .tsv file.

        Returns:
            dict sorting of all the test metrics values by their names.
        """
        return self._test(self.model.evaluate_dataset, test_dataset, **kwargs)

    def test_data(self, x, y, **kwargs) -> Dict:
        """
        Computes and returns the loss and the metrics of the model on a given test dataset.

        If the Experiment has logging enabled (i.e. self.logging is True), a checkpoint (the best one by default)
        is loaded and test and validation statistics are saved in a specific test output .tsv file. Otherwise, the
        current weights of the network is used for testing and statistics are only shown in the standard output.

        Args:
            x (Union[~torch.Tensor, ~numpy.ndarray] or Union[tuple, list] of Union[~torch.Tensor, ~numpy.ndarray]):
                Input to the model. Union[Tensor, ndarray] if the model has a single input.
                Union[tuple, list] of Union[Tensor, ndarray] if the model has multiple inputs.
            y (Union[~torch.Tensor, ~numpy.ndarray] or Union[tuple, list] of Union[~torch.Tensor, ~numpy.ndarray]):
                Target, corresponding ground truth.
                Union[Tensor, ndarray] if the model has a single output.
                Union[tuple, list] of Union[Tensor, ndarray] if the model has multiple outputs.
            checkpoint (Union[str, int]): Which model checkpoint weights to load for the test evaluation.

                - If 'best', will load the best weights according to ``monitor_metric`` and ``monitor_mode``.
                - If 'last', will load the last model checkpoint.
                - If int, will load the checkpoint of the specified epoch.
                - If a path (str), will load the model pickled state_dict weights (for instance, saved as
                  ``torch.save(a_pytorch_network.state_dict(), "./a_path.p")``).

                This argument has no effect when logging is disabled. (Default value = 'best')
            seed (int, optional): Seed used to make the sampling deterministic.
                (Default value = 42)
            name (str): Prefix of the test log file. (Default value = 'test')
            kwargs: Any keyword arguments to pass to :func:`~Model.evaluate()`.
        If the Experiment has logging enabled (i.e. self.logging is True), one callback will be automatically
        included to save the test metrics. Moreover, a :class:`~poutyne.AtomicCSVLogger` will save the test
        metrics in an output .tsv file.

        Returns:
            dict sorting of all the test metrics values by their names.
        """
        return self._test(self.model.evaluate, x, y, **kwargs)

    def _test(self,
              evaluate_func,
              *args,
              checkpoint: Union[str, int] = 'best',
              seed: int = 42,
              name='test',
              verbose=True,
              **kwargs) -> Dict:
        if kwargs.get('return_dict_format') is False:
            raise ValueError("This method only returns a dict.")
        kwargs['return_dict_format'] = True

        set_seeds(seed)

        if self.logging:
            if not self.monitoring and checkpoint == 'best':
                checkpoint = 'last'
            epoch_stats = self.load_checkpoint(checkpoint, verbose=verbose)

        if verbose:
            print(f"Running {name}")
        ret = evaluate_func(*args, **kwargs, verbose=verbose)

        if self.logging:
            test_metrics_dict = ret[0] if isinstance(ret, tuple) else ret
            test_stats = pd.DataFrame([list(test_metrics_dict.values())], columns=list(test_metrics_dict.keys()))
            test_stats.drop(['time'], axis=1, inplace=True)
            if epoch_stats is not None:
                epoch_stats = epoch_stats.reset_index(drop=True)
                test_stats = epoch_stats.join(test_stats)
            test_stats.to_csv(self.test_log_filename.format(name=name), sep='\t', index=False)

        return ret

    def predict(self, x, **kwargs) -> Any:
        """
        Returns the predictions of the network given a dataset ``x``, where the tensors are
        converted into Numpy arrays.

        Args:
            x (Union[~torch.Tensor, ~numpy.ndarray] or Union[tuple, list] of Union[~torch.Tensor, ~numpy.ndarray]):
                Input to the model. Union[Tensor, ndarray] if the model has a single input.
                Union[tuple, list] of Union[Tensor, ndarray] if the model has multiple inputs.
            checkpoint (Union[str, int]): Which model checkpoint weights to load for the prediction.

                - If 'best', will load the best weights according to ``monitor_metric`` and ``monitor_mode``.
                - If 'last', will load the last model checkpoint.
                - If int, will load the checkpoint of the specified epoch.
                - If a path (str), will load the model pickled state_dict weights (for instance, saved as
                  ``torch.save(a_pytorch_network.state_dict(), "./a_path.p")``).

                This argument has no effect when logging is disabled. (Default value = 'best')
            kwargs: Any keyword arguments to pass to :func:`~Model.predict()`.

        Returns:
            Return the predictions in the format outputted by the model.
        """
        return self._predict(self.model.predict, x, **kwargs)

    def predict_dataset(self, dataset, **kwargs) -> Any:
        """
        Returns the predictions of the network given a dataset, where the tensors are
        converted into Numpy arrays.

        Args:
            dataset (~torch.utils.data.Dataset): Dataset. Must not return ``y``, just ``x``.
            checkpoint (Union[str, int]): Which model checkpoint weights to load for the prediction.

                - If 'best', will load the best weights according to ``monitor_metric`` and ``monitor_mode``.
                - If 'last', will load the last model checkpoint.
                - If int, will load the checkpoint of the specified epoch.
                - If a path (str), will load the model pickled state_dict weights (for instance, saved as
                  ``torch.save(a_pytorch_network.state_dict(), "./a_path.p")``).

                This argument has no effect when logging is disabled. (Default value = 'best')
            kwargs: Any keyword arguments to pass to :func:`~Model.predict_dataset()`.

        Returns:
            Return the predictions in the format outputted by the model.
        """
        return self._predict(self.model.predict_dataset, dataset, **kwargs)

    def predict_generator(self, generator, **kwargs) -> Any:
        """
        Returns the predictions of the network given batches of samples ``x``, where the tensors are
        converted into Numpy arrays.

        Args:
            generator: Generator-like object for the dataset. The generator must yield a batch of
                samples. See the :func:`fit_generator()` method for details on the types of generators
                supported. This should only yield input data ``x`` and NOT the target ``y``.
            checkpoint (Union[str, int]): Which model checkpoint weights to load for the prediction.

                - If 'best', will load the best weights according to ``monitor_metric`` and ``monitor_mode``.
                - If 'last', will load the last model checkpoint.
                - If int, will load the checkpoint of the specified epoch.
                - If a path (str), will load the model pickled state_dict weights (for instance, saved as
                  ``torch.save(a_pytorch_network.state_dict(), "./a_path.p")``).

                This argument has no effect when logging is disabled. (Default value = 'best')
            kwargs: Any keyword arguments to pass to :func:`~Model.predict_generator()`.

        Returns:
            Depends on the value of ``concatenate_returns``. By default, (``concatenate_returns`` is true),
            the data structures (tensor, tuple, list, dict) returned as predictions for the batches are
            merged together. In the merge, the tensors are converted into Numpy arrays and are then
            concatenated together. If ``concatenate_returns`` is false, then a list of the predictions
            for the batches is returned with tensors converted into Numpy arrays.
        """
        return self._predict(self.model.predict_generator, generator, **kwargs)

    def predict_on_batch(self, x, **kwargs) -> Any:
        """
        Returns the predictions of the network given a batch ``x``, where the tensors are converted
        into Numpy arrays.

        Args:
            x: Input data as a batch.
            checkpoint (Union[str, int]): Which model checkpoint weights to load for the prediction.

                - If 'best', will load the best weights according to ``monitor_metric`` and ``monitor_mode``.
                - If 'last', will load the last model checkpoint.
                - If int, will load the checkpoint of the specified epoch.
                - If a path (str), will load the model pickled state_dict weights (for instance, saved as
                  ``torch.save(a_pytorch_network.state_dict(), "./a_path.p")``).

                This argument has no effect when logging is disabled. (Default value = 'best')
            verbose (bool): Whether to display the progress of the prediction.
                (Default value = True)

        Returns:
            Return the predictions in the format outputted by the model.
        """
        return self._predict(self.model.predict_on_batch, x, **kwargs)

    def _predict(self,
                 evaluate_func: Callable,
                 *args,
                 verbose=True,
                 checkpoint: Union[str, int] = 'best',
                 **kwargs) -> Any:
        if self.logging:
            if not self.monitoring and checkpoint == 'best':
                checkpoint = 'last'
            self.load_checkpoint(checkpoint, verbose=verbose)

        ret = evaluate_func(*args, verbose=verbose, **kwargs)
        return ret

    def is_better_than(self, another_experiment) -> bool:
        """
        Compare the results of the Experiment with another experiment. To compare, both Experiments need to be
        logged, monitor the same metric and the same monitor mode ("min" or "max").

        Args:
            another_experiment (~poutyne. Experiment): Another Poutyne experiment to compare results with.

        Return:
            Whether the Experiment is better than the Experiment to compare with.
        """
        if not self.logging:
            raise ValueError("The experiment is not logged.")
        if not another_experiment.logging:
            raise ValueError("The experiment to compare to is not logged.")

        if self.monitor_metric != another_experiment.monitor_metric:
            raise ValueError("The monitored metric is not the same between the two experiment.")
        monitored_metric = self.monitor_metric

        if self.monitor_mode != another_experiment.monitor_mode:
            raise ValueError("The monitored mode is not the same between the two experiment.")
        monitor_mode = self.monitor_mode

        checkpoint = 'best' if self.monitoring else 'last'
        self_stats = self.load_checkpoint(checkpoint, verbose=False)
        self_monitored_metric = self_stats[monitored_metric]
        self_monitored_metric_value = self_monitored_metric.item()

        other_checkpoint = 'best' if another_experiment.monitoring else 'last'
        other_stats = self.load_checkpoint(other_checkpoint, verbose=False)
        other_monitored_metric = other_stats[monitored_metric]
        other_monitored_metric_value = other_monitored_metric.item()

        if monitor_mode == 'min':
            is_better_than = self_monitored_metric_value < other_monitored_metric_value
        else:
            is_better_than = self_monitored_metric_value > other_monitored_metric_value
        return is_better_than
