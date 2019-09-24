# pylint: disable=redefined-builtin
import os
import random
import warnings

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from poutyne.framework import Model
from poutyne.utils import set_seeds
from poutyne.framework.callbacks import ModelCheckpoint, \
    OptimizerCheckpoint, \
    LRSchedulerCheckpoint, \
    PeriodicSaveLambda, \
    CSVLogger, \
    TensorBoardLogger, \
    BestModelRestore


class Experiment:
    """
    The Experiment class provides a straightforward experimentation tool for efficient finetuning of
    the whole neural network training procedure with PyTorch. The ``Experiment`` object takes
    care of the training and testing processes while also managing to keep traces of all pertinent
    information via the automatic logging option.

    Args:
        directory (str): Path to the experiment's working directory. Will be used for saving
        model (torch.nn.Module): A PyTorch module.
        device (torch.torch.device): The device to which the model is sent. If None, the model will be
            kept on its current device.
            (Default value = None)
        logging (bool): Whether or not to log the experiment's progress. If true, various logging
            callbacks will be inserted to output training and testing stats as well as to automatically
            save model checkpoints, for example.
            (Default value = True)
        optimizer (Union[torch.optim.Optimizer, str]): If Pytorch Optimizer, must already be initialized.
            If str, should be the optimizer's name in Pytorch (i.e. 'Adam' for torch.optim.Adam).
            (Default value = 'sgd')
        loss_function(Union[Callable, str]) It can be any PyTorch
            loss layer or custom loss function. It can also be a string with the same name as a PyTorch
            loss function (either the functional or object name). The loss function must have the signature
            ``loss_function(input, target)`` where ``input`` is the prediction of the network and ``target``
            is the ground truth. If ``None``, will default to, in priority order, either the model's own
            loss function or the default loss function associated with the ``task``.
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
        monitor_metric (str): Which metric to consider for best model performance calculation. Should be in
            the format '{val, train}_{metric_name}' (i.e. 'val_loss'). If None, will follow the value suggested
            by ``task`` or default to 'val_loss'.
            (Default value = None)
        monitor_mode (str): Which mode, either 'min' or 'max', should be used when considering the ``monitor_metric``
            value. If None, will follow the value suggested by ``task`` or default 'min'.
            (Default value = None)
        task (str): Any str beginning with either 'classif' or 'reg'. Specifying an ``task`` can assign default
            values to the ``loss_function``, ``batch_metrics``, ``monitor_mode`` and ``monitor_mode``. For ``task``
            that begins with 'reg', the only default value is the loss function that is the mean squared error. When
            beginning with 'classif', the default loss function is the cross-entropy loss, the default batch metrics
            will be the accuracy and the default monitoring will be set on 'val_acc' with a 'max' mode.
            (Default value = None)

    """
    BEST_CHECKPOINT_FILENAME = 'checkpoint_epoch_{epoch}.ckpt'
    BEST_CHECKPOINT_TMP_FILENAME = 'checkpoint_epoch.tmp.ckpt'
    MODEL_CHECKPOINT_FILENAME = 'checkpoint.ckpt'
    MODEL_CHECKPOINT_TMP_FILENAME = 'checkpoint.tmp.ckpt'
    OPTIMIZER_CHECKPOINT_FILENAME = 'checkpoint.optim'
    OPTIMIZER_CHECKPOINT_TMP_FILENAME = 'checkpoint.tmp.optim'
    LOG_FILENAME = 'log.tsv'
    TENSORBOARD_DIRECTORY = 'tensorboard'
    EPOCH_FILENAME = 'last.epoch'
    EPOCH_TMP_FILENAME = 'last.tmp.epoch'
    LR_SCHEDULER_FILENAME = 'lr_sched_%d.lrsched'
    LR_SCHEDULER_TMP_FILENAME = 'lr_sched_%d.tmp.lrsched'
    TEST_LOG_FILENAME = 'test_log.tsv'

    def __init__(self,
                 directory,
                 module,
                 *,
                 device=None,
                 logging=True,
                 optimizer='sgd',
                 loss_function=None,
                 batch_metrics=None,
                 epoch_metrics=None,
                 monitor_metric=None,
                 monitor_mode=None,
                 task=None):
        self.directory = directory
        self.logging = logging

        if task is not None and not task.startswith('classif') and not task.startswith('reg'):
            raise ValueError("Invalid task '%s'" % task)

        batch_metrics = [] if batch_metrics is None else batch_metrics
        epoch_metrics = [] if epoch_metrics is None else epoch_metrics

        loss_function = self._get_loss_function(loss_function, module, task)
        batch_metrics = self._get_batch_metrics(batch_metrics, module, task)
        epoch_metrics = self._get_epoch_metrics(epoch_metrics, module)
        self._set_monitor(monitor_metric, monitor_mode, task)

        self.model = Model(module, optimizer, loss_function, batch_metrics=batch_metrics, epoch_metrics=epoch_metrics)
        if device is not None:
            self.model.to(device)

        join_dir = lambda x: os.path.join(directory, x)

        self.best_checkpoint_filename = join_dir(Experiment.BEST_CHECKPOINT_FILENAME)
        self.best_checkpoint_tmp_filename = join_dir(Experiment.BEST_CHECKPOINT_TMP_FILENAME)
        self.model_checkpoint_filename = join_dir(Experiment.MODEL_CHECKPOINT_FILENAME)
        self.model_checkpoint_tmp_filename = join_dir(Experiment.MODEL_CHECKPOINT_TMP_FILENAME)
        self.optimizer_checkpoint_filename = join_dir(Experiment.OPTIMIZER_CHECKPOINT_FILENAME)
        self.optimizer_checkpoint_tmp_filename = join_dir(Experiment.OPTIMIZER_CHECKPOINT_TMP_FILENAME)
        self.log_filename = join_dir(Experiment.LOG_FILENAME)
        self.tensorboard_directory = join_dir(Experiment.TENSORBOARD_DIRECTORY)
        self.epoch_filename = join_dir(Experiment.EPOCH_FILENAME)
        self.epoch_tmp_filename = join_dir(Experiment.EPOCH_TMP_FILENAME)
        self.lr_scheduler_filename = join_dir(Experiment.LR_SCHEDULER_FILENAME)
        self.lr_scheduler_tmp_filename = join_dir(Experiment.LR_SCHEDULER_TMP_FILENAME)
        self.test_log_filename = join_dir(Experiment.TEST_LOG_FILENAME)

    def _get_loss_function(self, loss_function, module, task):
        if loss_function is None:
            if hasattr(module, 'loss_function'):
                return module.loss_function
            if task is not None:
                if task.startswith('classif'):
                    return 'cross_entropy'
                if task.startswith('reg'):
                    return 'mse'
        return loss_function

    def _get_batch_metrics(self, batch_metrics, module, task):
        if batch_metrics is None or len(batch_metrics) == 0:
            if hasattr(module, 'batch_metrics'):
                return module.batch_metrics
            if task is not None and task.startswith('classif'):
                return ['accuracy']
        return batch_metrics

    def _get_epoch_metrics(self, epoch_metrics, module):
        if epoch_metrics is None or len(epoch_metrics) == 0:
            if hasattr(module, 'epoch_metrics'):
                return module.epoch_metrics
        return epoch_metrics

    def _set_monitor(self, monitor_metric, monitor_mode, task):
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

    def get_best_epoch_stats(self):
        """
        Returns all computed statistics corresponding to the best epoch according to the
        ``monitor_metric`` and ``monitor_mode`` attributes.

        Returns:
            dict where each key is a column name in the logging output file
            and values are the ones found at the best epoch.
        """
        if pd is None:
            raise ImportError("pandas needs to be installed to use this function.")

        history = pd.read_csv(self.log_filename, sep='\t')
        if self.monitor_mode == 'min':
            best_epoch_index = history[self.monitor_metric].idxmin()
        else:
            best_epoch_index = history[self.monitor_metric].idxmax()
        return history.iloc[best_epoch_index:best_epoch_index + 1]

    def _warn_missing_file(self, filename):
        warnings.warn("Missing checkpoint: %s." % filename)

    def _load_epoch_state(self, lr_schedulers):
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

    def _init_model_restoring_callbacks(self, initial_epoch, save_every_epoch):
        callbacks = []
        best_checkpoint = ModelCheckpoint(self.best_checkpoint_filename,
                                          monitor=self.monitor_metric,
                                          mode=self.monitor_mode,
                                          save_best_only=not save_every_epoch,
                                          restore_best=not save_every_epoch,
                                          verbose=not save_every_epoch,
                                          temporary_filename=self.best_checkpoint_tmp_filename)
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

    def _init_tensorboard_callbacks(self, disable_tensorboard):
        tensorboard_writer = None
        callbacks = []
        if not disable_tensorboard:
            if SummaryWriter is None:
                warnings.warn("tensorboard does not seem to be installed. "
                              "To remove this warning, set the 'disable_tensorboard' "
                              "flag to True or install tensorboard.")
            else:
                tensorboard_writer = SummaryWriter(self.tensorboard_directory)
                callbacks += [TensorBoardLogger(tensorboard_writer)]
        return tensorboard_writer, callbacks

    def _init_lr_scheduler_callbacks(self, lr_schedulers):
        callbacks = []
        if self.logging:
            for i, lr_scheduler in enumerate(lr_schedulers):
                filename = self.lr_scheduler_filename % i
                tmp_filename = self.lr_scheduler_tmp_filename % i
                callbacks += [
                    LRSchedulerCheckpoint(lr_scheduler, filename, verbose=False, temporary_filename=tmp_filename)
                ]
        else:
            callbacks += lr_schedulers
            callbacks += [BestModelRestore(monitor=self.monitor_metric, mode=self.monitor_mode, verbose=True)]
        return callbacks

    def train(self,
              train_loader,
              valid_loader=None,
              *,
              callbacks=None,
              lr_schedulers=None,
              save_every_epoch=False,
              disable_tensorboard=False,
              epochs=1000,
              steps_per_epoch=None,
              validation_steps=None,
              batches_per_step=1,
              seed=42):
        # pylint: disable=too-many-locals
        """
        Trains the attribute model on a dataset using a loader.

        Args:
            train_loader: Generator-like object for the training set. See :func:`~Model.fit_generator()`
                for details on the types of generators supported.
            valid_loader (optional): Generator-like object for the validation set. See
                :func:`~Model.fit_generator()` for details on the types of generators supported.
                (Default value = None)
            callbacks (List[~poutyne.framework.callbacks.Callback]): List of callbacks that will be called during
                training.
                (Default value = None)
            lr_schedulers (List[~poutyne.framework.callbacks.lr_scheduler.LRScheduler]): List of learning rate schedulers.
                (Default value = None)
            save_every_epoch (bool, optional): Whether or not to save the attribute model's weights after
                every epoch.
                (Default value = False)
            disable_tensorboard (bool, optional): Wheter or not to disable the automatic tensorboard logging
                callbacks.
                (Default value = False)
            epochs (int): Number of times the entire training dataset is seen.
                (Default value = 1000)
            steps_per_epoch (int, optional): Number of batch used during one epoch. Obviously, using this
                argument may cause one epoch not to see the entire training dataset or see it multiple times.
                (Defaults the number of steps needed to see the entire
                training dataset)
            validation_steps (int, optional): Same as for ``steps_per_epoch`` but for the validation dataset.
                (Defaults to ``steps_per_epoch`` if provided or the number of steps needed to see the entire
                validation dataset)
            batches_per_step (int): Number of batches on which to compute the running loss before
                backpropagating it through the network. Note that the total loss used for backpropagation is
                the mean of the `batches_per_step` batch losses.
                (Default value = 1)
            seed (int, optional): Seed used to make the sampling deterministic.
                (Default value = 42)

        Returns:
            List of dict containing the history of each epoch.
        """
        set_seeds(seed)

        callbacks = [] if callbacks is None else callbacks
        lr_schedulers = [] if lr_schedulers is None else lr_schedulers

        # Copy callback list.
        callbacks = list(callbacks)

        tensorboard_writer = None
        initial_epoch = 1
        if self.logging:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            # Restarting optimization if needed.
            initial_epoch = self._load_epoch_state(lr_schedulers)

            callbacks += [CSVLogger(self.log_filename, separator='\t', append=initial_epoch != 1)]

            callbacks += self._init_model_restoring_callbacks(initial_epoch, save_every_epoch)
            callbacks += [
                ModelCheckpoint(self.model_checkpoint_filename,
                                verbose=False,
                                temporary_filename=self.model_checkpoint_tmp_filename)
            ]
            callbacks += [
                OptimizerCheckpoint(self.optimizer_checkpoint_filename,
                                    verbose=False,
                                    temporary_filename=self.optimizer_checkpoint_tmp_filename)
            ]

            # We save the last epoch number after the end of the epoch so that the
            # _load_epoch_state() knows which epoch to restart the optimization.
            callbacks += [
                PeriodicSaveLambda(lambda fd, epoch, logs: print(epoch, file=fd),
                                   self.epoch_filename,
                                   temporary_filename=self.epoch_tmp_filename,
                                   open_mode='w')
            ]

            tensorboard_writer, cb_list = self._init_tensorboard_callbacks(disable_tensorboard)
            callbacks += cb_list

        # This method returns callbacks that checkpoints the LR scheduler if logging is enabled.
        # Otherwise, it just returns the list of LR schedulers with a BestModelRestore callback.
        callbacks += self._init_lr_scheduler_callbacks(lr_schedulers)

        try:
            return self.model.fit_generator(train_loader,
                                            valid_loader,
                                            epochs=epochs,
                                            steps_per_epoch=steps_per_epoch,
                                            validation_steps=validation_steps,
                                            batches_per_step=batches_per_step,
                                            initial_epoch=initial_epoch,
                                            callbacks=callbacks)
        finally:
            if tensorboard_writer is not None:
                tensorboard_writer.close()

    def load_best_checkpoint(self, *, verbose=False):
        """
        Loads the attribute model's weights with the weights of the best checkpoint epoch
        with respect to the ``monitor_metric`` and ``monitor_mode`` attributes.

        Args:
            verbose (bool, optional): Whether or not to print the best epoch number and stats.
                (Default value = False)
        """
        best_epoch_stats = self.get_best_epoch_stats()
        best_epoch = best_epoch_stats['epoch'].item()

        if verbose:
            metrics_str = ', '.join('%s: %g' % (metric_name, best_epoch_stats[metric_name].item())
                                    for metric_name in best_epoch_stats.columns[2:])
            print("Found best checkpoint at epoch: {}".format(best_epoch))
            print(metrics_str)

        self.load_checkpoint(best_epoch)
        return best_epoch_stats

    def load_checkpoint(self, epoch):
        """
        Loads the attribute model's weights with the weights at a given checkpoint epoch.

        Args:
            epoch (int): The checkpoint epoch to load.
        """
        ckpt_filename = self.best_checkpoint_filename.format(epoch=epoch)
        self.model.load_weights(ckpt_filename)

    def load_last_checkpoint(self):
        """
        Loads the attribute model's weights with the weights of the last checkpoint.
        """
        self.model.load_weights(self.model_checkpoint_filename)

    def test(self, test_loader, *, steps=None, checkpoint='best', seed=42):
        """
        Computes and returns the loss and the metrics of the attribute model on a given test examples
        loader.

        Args:
            test_loader: Generator-like object for the test set. See :func:`~Model.fit_generator()` for
                details on the types of generators supported.
            steps (int, optional): Number of iterations done on ``generator``.
                (Defaults the number of steps needed to see the entire dataset)
            checkpoint (Union[str, int]): Which model checkpoint weights to load for the test evaluation.
                If 'best', will load the best weights according to ``monitor_metric`` and ``monitor_mode``.
                If 'last', will load the last model checkpoint. If int, will load the checkpoint of the
                specified epoch.
                (Default value = 'best')
            load_best_checkpoint (bool, optional): Whether or not to load the best checkpoint's weights.
                If set to true, the ``load_last_checkpoint`` argument is ignored.
                (Default value = True)
            load_last_checkpoint (bool, optional): Whether or not to load the last checkpoint's weights.
                (Default value = False)
            seed (int, optional): Seed used to make the sampling deterministic.
                (Default value = 42)

        Returns:
            dict sorting of all the test metrics values by their names.
        """
        set_seeds(seed)

        best_epoch_stats = None
        if checkpoint == 'best':
            best_epoch_stats = self.load_best_checkpoint(verbose=True)
        elif checkpoint == 'last':
            self.load_last_checkpoint()
        elif isinstance(checkpoint, int):
            self.load_checkpoint(checkpoint)
        else:
            raise ValueError("Argument checkpoint must be either 'best', 'last' or int. Found : {}".format(checkpoint))

        test_loss, test_metrics = self.model.evaluate_generator(test_loader, steps=steps)
        if not isinstance(test_metrics, np.ndarray):
            test_metrics = np.array([test_metrics])

        test_metrics_names = ['test_loss'] + \
                             ['test_' + metric_name for metric_name in self.model.metrics_names]
        test_metrics_values = np.concatenate(([test_loss], test_metrics))

        test_metrics_dict = {col: val for col, val in zip(test_metrics_names, test_metrics_values)}
        test_metrics_str = ', '.join('%s: %g' % (col, val) for col, val in test_metrics_dict.items())
        print("On best model: %s" % test_metrics_str)

        if self.logging:
            test_stats = pd.DataFrame([test_metrics_values], columns=test_metrics_names)
            if best_epoch_stats is not None:
                best_epoch_stats = best_epoch_stats.reset_index(drop=True)
                test_stats = best_epoch_stats.join(test_stats)
            test_stats.to_csv(self.test_log_filename, sep='\t', index=False)

        return test_metrics_dict
