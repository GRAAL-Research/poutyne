import os
import warnings

import numpy as np
import pandas as pd

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

def get_loss_and_metrics(module, loss_function, metrics):
    if loss_function is None and hasattr(module, 'loss_function'):
        loss_function = module.loss_function
    if (metrics is None or len(metrics) == 0) and hasattr(module, 'metrics'):
        metrics = module.metrics
    return loss_function, metrics

def get_best_epoch_stats(directory, monitor, mode):
    if mode not in ['min', 'max']:
        raise ValueError("Invalid mode '%s'" % mode)

    log_filename = os.path.join(directory, LOG_FILENAME)

    history = pd.read_csv(log_filename, sep='\t')
    if mode == 'min':
        best_epoch_index = history[monitor].idxmin()
    else:
        best_epoch_index = history[monitor].idxmax()
    return history.iloc[best_epoch_index:best_epoch_index + 1]

def warn_missing_file(filename):
    warnings.warn("Missing checkpoint: %s." % filename)

def load_epoch_state(model, lr_schedulers, epoch_filename, model_filename, optimizer_filename, lr_scheduler_filename):
    initial_epoch = 1
    if os.path.isfile(epoch_filename):
        try:
            with open(epoch_filename, 'r') as f:
                initial_epoch = int(f.read()) + 1
        except Exception as e:
            print(e)
        if os.path.isfile(model_filename):
            try:
                print("Loading weights from %s and starting at epoch %d." % (model_filename, initial_epoch))
                model.load_weights(model_filename)
            except Exception as e:
                print(e)
        else:
            warn_missing_file(model_filename)
        if os.path.isfile(optimizer_filename):
            try:
                print("Loading optimizer state from %s and starting at epoch %d." % (optimizer_filename, initial_epoch))
                model.load_optimizer_state(optimizer_filename)
            except Exception as e:
                print(e)
        else:
            warn_missing_file(optimizer_filename)
        for i, lr_scheduler in enumerate(lr_schedulers):
            filename = lr_scheduler_filename % i
            if os.path.isfile(filename):
                try:
                    print("Loading LR scheduler state from %s and starting at epoch %d." % (filename, initial_epoch))
                    lr_scheduler.load_state(filename)
                except Exception as e:
                    print(e)
            else:
                warn_missing_file(filename)
    return initial_epoch

def set_best_epoch_model_checkpoint(directory, best_checkpoint_callback, monitor, mode):
    log_filename = os.path.join(directory, LOG_FILENAME)
    best_checkpoint_filename = os.path.join(directory, BEST_CHECKPOINT_FILENAME)
    if os.path.isfile(log_filename):
        best_epoch_stats = get_best_epoch_stats(directory, monitor, mode)
        best_epoch = best_epoch_stats['epoch'].item()
        best_checkpoint_callback.best_filename = best_checkpoint_filename.format(epoch=best_epoch)
        best_checkpoint_callback.current_best = best_epoch_stats[monitor].item()

def train(directory, module, train_loader, valid_loader=None,
          logging=True, optimizer=None, loss_function=None,
          metrics=[], device=None, callbacks=[], lr_schedulers=[],
          monitor_metric='val_loss', monitor_mode='min',
          disable_tensorboard=False,
          epochs=1000, steps_per_epoch=None, validation_steps=None):
    loss_function, metrics = get_loss_and_metrics(module, loss_function, metrics)

    # Copy callback list.
    callbacks = list(callbacks)

    model = Model(module, optimizer, loss_function, metrics=metrics)
    if device is not None:
        model.to(device)

    initial_epoch = 1
    if logging:
        if not os.path.exists(directory):
            os.makedirs(directory)

        best_checkpoint_filename = os.path.join(directory, BEST_CHECKPOINT_FILENAME)
        best_checkpoint_tmp_filename = os.path.join(directory, BEST_CHECKPOINT_TMP_FILENAME)
        model_checkpoint_filename = os.path.join(directory, MODEL_CHECKPOINT_FILENAME)
        model_checkpoint_tmp_filename = os.path.join(directory, MODEL_CHECKPOINT_TMP_FILENAME)
        optimizer_checkpoint_filename = os.path.join(directory, OPTIMIZER_CHECKPOINT_FILENAME)
        optimizer_checkpoint_tmp_filename = os.path.join(directory, OPTIMIZER_CHECKPOINT_TMP_FILENAME)
        log_filename = os.path.join(directory, LOG_FILENAME)
        tensorboard_directory = os.path.join(directory, TENSORBOARD_DIRECTORY)
        epoch_filename = os.path.join(directory, EPOCH_FILENAME)
        epoch_tmp_filename = os.path.join(directory, EPOCH_TMP_FILENAME)
        lr_scheduler_filename = os.path.join(directory, LR_SCHEDULER_FILENAME)
        lr_scheduler_tmp_filename = os.path.join(directory, LR_SCHEDULER_TMP_FILENAME)

        # Restarting optimization if needed.
        initial_epoch = load_epoch_state(model,
                                         lr_schedulers,
                                         epoch_filename,
                                         model_checkpoint_filename,
                                         optimizer_checkpoint_filename,
                                         lr_scheduler_filename)

        csv_logger = CSVLogger(log_filename, separator='\t', append=initial_epoch != 1)


        best_checkpoint = ModelCheckpoint(best_checkpoint_filename, monitor=monitor_metric, mode=monitor_mode, save_best_only=True, restore_best=True, verbose=True, temporary_filename=best_checkpoint_tmp_filename)
        # We set the current best metric score in the ModelCheckpoint so that
        # it does not save checkpoint it would not have saved if the
        # optimization was not stopped.
        set_best_epoch_model_checkpoint(directory, best_checkpoint, monitor_metric, monitor_mode)

        checkpoint = ModelCheckpoint(model_checkpoint_filename, verbose=False, temporary_filename=model_checkpoint_tmp_filename)
        optimizer_checkpoint = OptimizerCheckpoint(optimizer_checkpoint_filename, verbose=False, temporary_filename=optimizer_checkpoint_tmp_filename)

        # We save the last epoch number after the end of the epoch so that the
        # load_epoch_state() knows which epoch to restart the optimization.
        save_epoch_number = PeriodicSaveLambda(lambda fd, epoch, logs: print(epoch, file=fd),
                                               epoch_filename,
                                               temporary_filename=epoch_tmp_filename,
                                               open_mode='w')

        callbacks += [csv_logger, best_checkpoint, checkpoint, optimizer_checkpoint, save_epoch_number]

        if not disable_tensorboard:
            if SummaryWriter is None:
                warnings.warn("tensorboardX does not seem to be installed. To remove this warning, set the 'disable_tensorboard' flag to True.")
            else:
                writer = SummaryWriter(tensorboard_directory)
                tensorboard_logger = TensorBoardLogger(writer)
                callbacks.append(tensorboard_logger)
        for i, lr_scheduler in enumerate(lr_schedulers):
            filename = lr_scheduler_filename % i
            tmp_filename = lr_scheduler_tmp_filename % i
            lr_scheduler_checkpoint = LRSchedulerCheckpoint(lr_scheduler, filename, verbose=False, temporary_filename=tmp_filename)
            callbacks.append(lr_scheduler_checkpoint)
    else:
        for lr_scheduler in lr_schedulers:
            callbacks.append(lr_scheduler)
        best_restore = BestModelRestore(monitor=monitor_metric, mode=monitor_mode)
        callbacks.append(best_restore)

    model.fit_generator(train_loader, valid_loader,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch,
                        callbacks=callbacks)

    return model

def train_classifier(directory, module, train_loader, valid_loader,
                    loss_function=nn.CrossEntropyLoss(), metrics=['accuracy'],
                    monitor_metric='val_acc', monitor_mode='max',
                    **kwargs):
    return train(directory,
                 module,
                 train_loader,
                 valid_loader,
                 loss_function=loss_function,
                 metrics=metrics,
                 monitor_metric=monitor_metric,
                 monitor_mode=monitor_mode,
                 **kwargs)

def train_regressor(*args,
                    loss_function=nn.MSELoss(),
                    **kwargs):
    return train(*args,
                 loss_function=loss_function,
                 **kwargs)

def test(directory, module, test_loader,
         logging=True, load_best_checkpoint=True,
         loss_function=None,
         metrics=[], device=None,
         monitor_metric='val_loss', monitor_mode='min',
         steps=None):
    loss_function, metrics = get_loss_and_metrics(module, loss_function, metrics)

    model = Model(module, None, loss_function, metrics=metrics)
    if device is not None:
        model.to(device)

    if load_best_checkpoint:
        best_checkpoint_filename = os.path.join(directory, BEST_CHECKPOINT_FILENAME)
        best_epoch_stats = get_best_epoch_stats(directory, monitor_metric, monitor_mode)
        best_epoch = best_epoch_stats['epoch'].item()

        metrics_names = model.metrics_names + ['val_' + metric_name for metric_name in model.metrics_names]
        # Some metrics may not have been used during training
        metrics_str = ', '.join('%s: %g' % (metric_name, best_epoch_stats[metric_name].item())
                                    for metric_name in metrics_names
                                        if best_epoch_stats.get(metric_name) is not None)
        print("Found best checkpoint at epoch: {}".format(best_epoch))
        print(metrics_str)

        best_ckpt_filename = best_checkpoint_filename.format(epoch=best_epoch)
        model.load_weights(best_ckpt_filename)

    test_loss, test_metrics = model.evaluate_generator(test_loader, steps=steps)
    if not isinstance(test_metrics, np.ndarray):
        test_metrics = np.array([test_metrics])

    test_metrics_names = ['test_loss'] + ['test_' + metric_name for metric_name in model.metrics_names]
    test_metrics_values = np.concatenate(([test_loss], test_metrics))
    test_metrics_str = ', '.join('%s: %g' % (col, val) for col, val in zip(test_metrics_names, test_metrics_values))
    print("On best model: %s" % test_metrics_str)

    if logging:
        test_stats = pd.DataFrame([test_metrics_values], columns=test_metrics_names)
        best_epoch_stats = best_epoch_stats.reset_index(drop=True)
        best_epoch_stats = best_epoch_stats.join(test_stats)

        test_log_filename = os.path.join(directory, TEST_LOG_FILENAME)
        best_epoch_stats.to_csv(test_log_filename, sep='\t', index=False)

    return model

def test_classifier(*args,
                    loss_function=nn.CrossEntropyLoss(),
                    metrics=['accuracy'],
                    monitor_metric='val_acc', monitor_mode='max',
                    **kwargs):
    return test(*args,
                loss_function=loss_function,
                metrics=metrics,
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                **kwargs)

def test_regressor(*args,
                   loss_function=nn.MSELoss(),
                   **kwargs):
    return test(*args,
                loss_function=loss_function,
                **kwargs)
