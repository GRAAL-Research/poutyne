import os
import warnings

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

def train(directory, module, train_loader, valid_loader=None,
          logging=True, optimizer=None, loss_function=None,
          metrics=[], device=None, callbacks=[], lr_schedulers=[],
          monitor_metric='val_loss', monitor_mode='min',
          disable_tensorboard=False,
          epochs=1000, steps_per_epoch=None, validation_steps=None):

    if loss_function is None and hasattr(module, 'loss_function'):
        loss_function = module.loss_function
    if (metrics is None or len(metrics) == 0) and hasattr(module, 'metrics'):
        metrics = module.metrics

    # Copy callback list.
    callbacks = list(callbacks)

    model = Model(module, optimizer, loss_function, metrics=metrics)
    if device is not None:
        model.to(device)

    initial_epoch = 1
    if logging:
        if not os.path.exists(directory):
            os.makedirs(directory)

        best_checkpoint_filename = os.path.join(directory, 'checkpoint_epoch_{epoch}.ckpt')
        best_checkpoint_tmp_filename = os.path.join(directory, 'checkpoint_epoch.tmp.ckpt')
        model_checkpoint_filename = os.path.join(directory, 'checkpoint.ckpt')
        model_checkpoint_tmp_filename = os.path.join(directory, 'checkpoint.tmp.ckpt')
        optimizer_checkpoint_filename = os.path.join(directory, 'checkpoint.optim')
        optimizer_checkpoint_tmp_filename = os.path.join(directory, 'checkpoint.tmp.optim')
        log_filename = os.path.join(directory, 'log.tsv')
        tensorboard_directory = os.path.join(directory, 'tensorboard')
        epoch_filename = os.path.join(directory, 'last.epoch')
        epoch_tmp_filename = os.path.join(directory, 'last.tmp.epoch')
        lr_scheduler_filename = os.path.join(directory, 'lr_sched_%d.lrsched')
        lr_scheduler_tmp_filename = os.path.join(directory, 'lr_sched_%d.tmp.lrsched')

        # Restarting optimization if needed.
        initial_epoch = load_epoch_state(model,
                                         lr_schedulers,
                                         epoch_filename,
                                         model_checkpoint_filename,
                                         optimizer_checkpoint_filename,
                                         lr_scheduler_filename)

        csv_logger = CSVLogger(log_filename, separator='\t', append=initial_epoch != 1)
        best_checkpoint = ModelCheckpoint(best_checkpoint_filename, monitor=monitor_metric, mode=monitor_mode, save_best_only=True, restore_best=True, verbose=True, temporary_filename=best_checkpoint_tmp_filename)
        checkpoint = ModelCheckpoint(model_checkpoint_filename, verbose=False, temporary_filename=model_checkpoint_tmp_filename)
        optimizer_checkpoint = OptimizerCheckpoint(optimizer_checkpoint_filename, verbose=False, temporary_filename=optimizer_checkpoint_tmp_filename)
        save_epoch_number = PeriodicSaveLambda(lambda fd, epoch, logs: print(epoch, file=fd),
                                               epoch_filename,
                                               temporary_filename=epoch_tmp_filename,
                                               open_mode='w')
        callbacks += [csv_logger, best_checkpoint, checkpoint, optimizer_checkpoint, save_epoch_number]
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
