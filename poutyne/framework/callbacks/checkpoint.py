import warnings

from .periodic import PeriodicSaveCallback
from .lr_scheduler import PyTorchLRSchedulerWrapper, ReduceLROnPlateau


class ModelCheckpoint(PeriodicSaveCallback):
    """
    Save the model after every epoch. See
    `poutyne.framework.PeriodicSaveCallback` for the arguments' descriptions.

    Args:
        restore_best (bool): If `restore_best` is true, the weights of the
            network will be reset to the last best checkpoint done. This option
            only works when `save_best_only` is also true.
            (Default value = False)

    See:
        poutyne.framework.PeriodicSaveCallback
    """

    def __init__(self, *args, restore_best=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.restore_best = restore_best
        if self.restore_best and not self.save_best_only:
            raise ValueError("The 'restore_best' argument only works when "
                             "'save_best_only' is also true.")

    def save_file(self, fd, epoch, logs):
        self.model.save_weights(fd)

    def on_train_end(self, logs):
        if self.restore_best:
            if self.best_filename is not None:
                if self.verbose:
                    print('Restoring model from %s' % self.best_filename)
                self.model.load_weights(self.best_filename)
            else:
                warnings.warn('No  weights to restore!')

class OptimizerCheckpoint(PeriodicSaveCallback):
    """
    Save the state of the optimizer after every epoch. The optimizer can be
    reloaded as follows.

    .. code-block:: python

        model = Model(model, optimizer, loss_function)
        model.load_optimizer_state(filename)

    See `poutyne.framework.PeriodicSaveCallback` for the arguments'
    descriptions.

    See:
        poutyne.framework.PeriodicSaveCallback
    """
    def save_file(self, fd, epoch, logs):
        self.model.save_optimizer_state(fd)

class LRSchedulerCheckpoint(PeriodicSaveCallback):
    """
    Save the state of an LR scheduler callback after every epoch. The LR
    scheduler callback should not be passed to the fit*() methods since it is
    called by this callback instead. The LR scheduler can be reloaded as
    follows.

    .. code-block:: python

        lr_scheduler = AnLRSchedulerCallback(...)
        lr_scheduler.load_state(filename)

    See `poutyne.framework.PeriodicSaveCallback` for the arguments'
    descriptions.

    Args:
        lr_scheduler: An LR scheduler callback.

    See:
        poutyne.framework.PeriodicSaveCallback
    """
    def __init__(self, lr_scheduler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = lr_scheduler

        if not isinstance(self.lr_scheduler, (PyTorchLRSchedulerWrapper, ReduceLROnPlateau)):
            raise ValueError("Unknown scheduler callback '%s'." % lr_scheduler)

    def save_file(self, fd, epoch, logs):
        self.lr_scheduler.save_state(fd)

    def set_params(self, params):
        self.lr_scheduler.set_params(params)
        super().set_params(params)

    def set_model(self, model):
        self.lr_scheduler.set_model(model)
        super().set_model(model)

    def on_epoch_begin(self, epoch, logs):
        self.lr_scheduler.on_epoch_begin(epoch, logs)
        super().on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs):
        self.lr_scheduler.on_epoch_end(epoch, logs)
        super().on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs):
        self.lr_scheduler.on_batch_begin(batch, logs)
        super().on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs):
        self.lr_scheduler.on_batch_end(batch, logs)
        super().on_batch_end(batch, logs)

    def on_backward_end(self, batch):
        self.lr_scheduler.on_backward_end(batch)
        super().on_backward_end(batch)

    def on_train_begin(self, logs):
        self.lr_scheduler.on_train_begin(logs)
        super().on_train_begin(logs)

    def on_train_end(self, logs):
        self.lr_scheduler.on_train_end(logs)
        super().on_train_end(logs)
