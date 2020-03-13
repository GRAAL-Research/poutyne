import warnings
from typing import Dict, BinaryIO

from .lr_scheduler import _PyTorchLRSchedulerWrapper, ReduceLROnPlateau
from .periodic import PeriodicSaveCallback


class ModelCheckpoint(PeriodicSaveCallback):
    """
    Save the model after every epoch. See
    :class:`~poutyne.framework.callbacks.PeriodicSaveCallback` for the arguments' descriptions.

    Args:
        restore_best (bool): If `restore_best` is true, the weights of the network will be reset to
            the last best checkpoint done. This option only works when `save_best_only` is also true.
            (Default value = False)

    See:
        :class:`~poutyne.framework.callbacks.PeriodicSaveCallback`
    """

    def __init__(self, *args, restore_best: bool = False, **kwargs):
        super().__init__(*args, **kwargs)

        self.restore_best = restore_best
        if self.restore_best and not self.save_best_only:
            raise ValueError("The 'restore_best' argument only works when 'save_best_only' is also true.")

    def save_file(self, fd: BinaryIO, epoch_number: int, logs: Dict):
        self.model.save_weights(fd)

    def on_train_end(self, logs: Dict):
        if self.restore_best:
            if self.best_filename is not None:
                if self.verbose:
                    print('Restoring model from %s' % self.best_filename)
                self.model.load_weights(self.best_filename)
            else:
                warnings.warn('No  weights to restore!')


class OptimizerCheckpoint(PeriodicSaveCallback):
    """
    Save the state of the optimizer after every epoch. The optimizer can be reloaded as follows.

    .. code-block:: python

        model = Model(model, optimizer, loss_function)
        model.load_optimizer_state(filename)

    See :class:`~poutyne.framework.callbacks.PeriodicSaveCallback` for the arguments' descriptions.

    See:
        :class:`~poutyne.framework.callbacks.PeriodicSaveCallback`
    """

    def save_file(self, fd: BinaryIO, epoch_number: int, logs: Dict):
        self.model.save_optimizer_state(fd)


class LRSchedulerCheckpoint(PeriodicSaveCallback):
    """
    Save the state of an LR scheduler callback after every epoch. The LR scheduler callback should
    not be passed to the fit*() methods since it is called by this callback instead. The LR
    scheduler can be reloaded as follows.

    .. code-block:: python

        lr_scheduler = AnLRSchedulerCallback(...)
        lr_scheduler.load_state(filename)

    See :class:`~poutyne.framework.callbacks.PeriodicSaveCallback` for the arguments' descriptions.

    Args:
        lr_scheduler: An LR scheduler callback.

    See:
        :class:`~poutyne.framework.callbacks.PeriodicSaveCallback`
    """

    def __init__(self, lr_scheduler: _PyTorchLRSchedulerWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = lr_scheduler

        if not isinstance(self.lr_scheduler, (_PyTorchLRSchedulerWrapper, ReduceLROnPlateau)):
            raise ValueError("Unknown scheduler callback '%s'." % lr_scheduler)

    def save_file(self, fd: BinaryIO, epoch_number: int, logs: Dict):
        self.lr_scheduler.save_state(fd)

    def set_params(self, params: Dict):
        self.lr_scheduler.set_params(params)
        super().set_params(params)

    def set_model(self, model):
        self.lr_scheduler.set_model(model)
        super().set_model(model)

    def on_epoch_begin(self, epoch_number: int, logs: Dict):
        self.lr_scheduler.on_epoch_begin(epoch_number, logs)
        super().on_epoch_begin(epoch_number, logs)

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        self.lr_scheduler.on_epoch_end(epoch_number, logs)
        super().on_epoch_end(epoch_number, logs)

    def on_train_batch_begin(self, batch_number: int, logs: Dict):
        self.lr_scheduler.on_train_batch_begin(batch_number, logs)
        super().on_train_batch_begin(batch_number, logs)

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        self.lr_scheduler.on_train_batch_end(batch_number, logs)
        super().on_train_batch_end(batch_number, logs)

    def on_backward_end(self, batch_number: int):
        self.lr_scheduler.on_backward_end(batch_number)
        super().on_backward_end(batch_number)

    def on_train_begin(self, logs: Dict):
        self.lr_scheduler.on_train_begin(logs)
        super().on_train_begin(logs)

    def on_train_end(self, logs: Dict):
        self.lr_scheduler.on_train_end(logs)
        super().on_train_end(logs)
