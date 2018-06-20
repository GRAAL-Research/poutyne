import os
import warnings
import random

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
import torch
import torch.nn as nn

from pytoune.framework import Model
from pytoune.framework.callbacks import ModelCheckpoint, \
                                        OptimizerCheckpoint, \
                                        LRSchedulerCheckpoint, \
                                        PeriodicSaveLambda, \
                                        CSVLogger, \
                                        TensorBoardLogger, \
                                        BestModelRestore
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

class Experiment:
    BEST_CHECKPOINT_FILENAME            = 'checkpoint_epoch_{epoch}.ckpt'
    BEST_CHECKPOINT_TMP_FILENAME        = 'checkpoint_epoch.tmp.ckpt'
    MODEL_CHECKPOINT_FILENAME           = 'checkpoint.ckpt'
    MODEL_CHECKPOINT_TMP_FILENAME       = 'checkpoint.tmp.ckpt'
    OPTIMIZER_CHECKPOINT_FILENAME       = 'checkpoint.optim'
    OPTIMIZER_CHECKPOINT_TMP_FILENAME   = 'checkpoint.tmp.optim'
    LOG_FILENAME                        = 'log.tsv'
    TENSORBOARD_DIRECTORY               = 'tensorboard'
    EPOCH_FILENAME                      = 'last.epoch'
    EPOCH_TMP_FILENAME                  = 'last.tmp.epoch'
    LR_SCHEDULER_FILENAME               = 'lr_sched_%d.lrsched'
    LR_SCHEDULER_TMP_FILENAME           = 'lr_sched_%d.tmp.lrsched'
    TEST_LOG_FILENAME                   = 'test_log.tsv'

    def __init__(self, directory, module, device=None, logging=True,
                 optimizer=None, loss_function=None, metrics=[],
                 monitor_metric=None, monitor_mode=None, type=None):
        self.directory = directory
        self.logging = logging

        if loss_function is None and hasattr(module, 'loss_function'):
            loss_function = module.loss_function
        if (metrics is None or len(metrics) == 0) and hasattr(module, 'metrics'):
            metrics = module.metrics

        if monitor_mode is not None and monitor_mode not in ['min', 'max']:
            raise ValueError("Invalid mode '%s'" % monitor_mode)

        if monitor_metric is not None:
            self.monitor_metric = monitor_metric
            self.monitor_mode = monitor_mode if monitor_mode is not None else 'min'
        else:
            self.monitor_metric = 'val_loss'
            self.monitor_mode = 'min'
            if type is not None:
                if type.startswith('classif'):
                    metrics = ['accuracy'] if len(metrics) == 0 else metrics
                    self.monitor_metric = 'val_acc'
                    self.monitor_mode = 'max'
                    loss_function = nn.CrossEntropyLoss() if loss_function is None else loss_function
                elif type.startswith('reg'):
                    loss_function = nn.MSELoss() if loss_function is None else loss_function

        self.model = Model(module, optimizer, loss_function, metrics=metrics)
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

    def get_best_epoch_stats(self):
        if pd is None:
            warnings.warn("pandas needs to be installed to use this function.")

        history = pd.read_csv(self.log_filename, sep='\t')
        if self.monitor_mode == 'min':
            best_epoch_index = history[self.monitor_metric].idxmin()
        else:
            best_epoch_index = history[self.monitor_metric].idxmax()
        return history.iloc[best_epoch_index:best_epoch_index + 1]

    def _warn_missing_file(self, filename):
        warnings.warn("Missing checkpoint: %s." % filename)

    def _load_epoch_state(self, lr_schedulers):
        initial_epoch = 1
        if os.path.isfile(self.epoch_filename):
            try:
                with open(self.epoch_filename, 'r') as f:
                    initial_epoch = int(f.read()) + 1
            except Exception as e:
                print(e)
            if os.path.isfile(self.model_checkpoint_filename):
                try:
                    print("Loading weights from %s and starting at epoch %d." % (self.model_checkpoint_filename, initial_epoch))
                    self.model.load_weights(self.model_checkpoint_filename)
                except Exception as e:
                    print(e)
            else:
                self._warn_missing_file(self.model_checkpoint_filename)
            if os.path.isfile(self.optimizer_checkpoint_filename):
                try:
                    print("Loading optimizer state from %s and starting at epoch %d." % (self.optimizer_checkpoint_filename, initial_epoch))
                    self.model.load_optimizer_state(self.optimizer_checkpoint_filename)
                except Exception as e:
                    print(e)
            else:
                self._warn_missing_file(self.optimizer_checkpoint_filename)
            for i, lr_scheduler in enumerate(lr_schedulers):
                filename = self.lr_scheduler_filename % i
                if os.path.isfile(filename):
                    try:
                        print("Loading LR scheduler state from %s and starting at epoch %d." % (filename, initial_epoch))
                        lr_scheduler.load_state(filename)
                    except Exception as e:
                        print(e)
                else:
                    self._warn_missing_file(filename)
        return initial_epoch

    def train(self, train_loader, valid_loader=None,
              callbacks=[], lr_schedulers=[],
              disable_tensorboard=False,
              epochs=1000, steps_per_epoch=None, validation_steps=None,
              seed=42):
        if seed is not None:
            # Make training deterministic.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Copy callback list.
        callbacks = list(callbacks)

        initial_epoch = 1
        if self.logging:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            # Restarting optimization if needed.
            initial_epoch = self._load_epoch_state(lr_schedulers)

            csv_logger = CSVLogger(self.log_filename, separator='\t', append=initial_epoch != 1)


            best_checkpoint = ModelCheckpoint(self.best_checkpoint_filename,
                                              monitor=self.monitor_metric,
                                              mode=self.monitor_mode,
                                              save_best_only=True,
                                              restore_best=True,
                                              verbose=True,
                                              temporary_filename=self.best_checkpoint_tmp_filename)
            if initial_epoch > 1:
                # We set the current best metric score in the ModelCheckpoint so that
                # it does not save checkpoint it would not have saved if the
                # optimization was not stopped.
                best_epoch_stats = self.get_best_epoch_stats()
                best_epoch = best_epoch_stats['epoch'].item()
                best_checkpoint.best_filename = self.best_checkpoint_filename.format(epoch=best_epoch)
                best_checkpoint.current_best = best_epoch_stats[self.monitor_metric].item()

            checkpoint = ModelCheckpoint(self.model_checkpoint_filename, verbose=False, temporary_filename=self.model_checkpoint_tmp_filename)
            optimizer_checkpoint = OptimizerCheckpoint(self.optimizer_checkpoint_filename, verbose=False, temporary_filename=self.optimizer_checkpoint_tmp_filename)

            # We save the last epoch number after the end of the epoch so that the
            # load_epoch_state() knows which epoch to restart the optimization.
            save_epoch_number = PeriodicSaveLambda(lambda fd, epoch, logs: print(epoch, file=fd),
                                                   self.epoch_filename,
                                                   temporary_filename=self.epoch_tmp_filename,
                                                   open_mode='w')

            callbacks += [csv_logger, best_checkpoint, checkpoint, optimizer_checkpoint, save_epoch_number]

            if not disable_tensorboard:
                if SummaryWriter is None:
                    warnings.warn("tensorboardX does not seem to be installed. To remove this warning, set the 'disable_tensorboard' flag to True.")
                else:
                    writer = SummaryWriter(self.tensorboard_directory)
                    tensorboard_logger = TensorBoardLogger(writer)
                    callbacks.append(tensorboard_logger)
            for i, lr_scheduler in enumerate(lr_schedulers):
                filename = self.lr_scheduler_filename % i
                tmp_filename = self.lr_scheduler_tmp_filename % i
                lr_scheduler_checkpoint = LRSchedulerCheckpoint(lr_scheduler, filename, verbose=False, temporary_filename=tmp_filename)
                callbacks.append(lr_scheduler_checkpoint)
        else:
            for lr_scheduler in lr_schedulers:
                callbacks.append(lr_scheduler)
            best_restore = BestModelRestore(monitor=self.monitor_metric, mode=self.monitor_mode)
            callbacks.append(best_restore)

        self.model.fit_generator(train_loader, valid_loader,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            initial_epoch=initial_epoch,
                            callbacks=callbacks)


    def load_best_checkpoint(self, verbose=False):
        best_epoch_stats = self.get_best_epoch_stats()
        best_epoch = best_epoch_stats['epoch'].item()

        if verbose:
            metrics_str = ', '.join('%s: %g' % (metric_name, best_epoch_stats[metric_name].item())
                                        for metric_name in best_epoch_stats.columns[2:])
            print("Found best checkpoint at epoch: {}".format(best_epoch))
            print(metrics_str)

        best_ckpt_filename = self.best_checkpoint_filename.format(epoch=best_epoch)
        self.model.load_weights(best_ckpt_filename)
        return best_epoch_stats

    def load_last_checkpoint(self):
        self.model.load_weights(self.model_checkpoint_filename)

    def test(self, test_loader, steps=None,
             do_load_best_checkpoint=True, do_load_last_checkpoint=False,
             seed=42):
        if seed is not None:
            # Make training deterministic.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        best_epoch_stats = None
        if do_load_best_checkpoint:
            best_epoch_stats = self.load_best_checkpoint(verbose=True)
        elif do_load_last_checkpoint:
            best_epoch_stats = self.load_last_checkpoint()

        test_loss, test_metrics = self.model.evaluate_generator(test_loader, steps=steps)
        if not isinstance(test_metrics, np.ndarray):
            test_metrics = np.array([test_metrics])

        test_metrics_names = ['test_loss'] + ['test_' + metric_name for metric_name in self.model.metrics_names]
        test_metrics_values = np.concatenate(([test_loss], test_metrics))
        test_metrics_str = ', '.join('%s: %g' % (col, val) for col, val in zip(test_metrics_names, test_metrics_values))
        print("On best model: %s" % test_metrics_str)

        if self.logging:
            test_stats = pd.DataFrame([test_metrics_values], columns=test_metrics_names)
            if best_epoch_stats is not None:
                best_epoch_stats = best_epoch_stats.reset_index(drop=True)
                test_stats = best_epoch_stats.join(test_stats)
            test_stats.to_csv(self.test_log_filename, sep='\t', index=False)
