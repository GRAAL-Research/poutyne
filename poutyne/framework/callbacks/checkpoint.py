import warnings
from typing import Dict, IO
import torch

from .lr_scheduler import _PyTorchLRSchedulerWrapper, ReduceLROnPlateau
from .periodic import PeriodicSaveCallback


class ModelCheckpoint(PeriodicSaveCallback):
    """
    Save the model after every epoch. See
    :class:`~poutyne.PeriodicSaveCallback` for the arguments' descriptions.

    See:
        :class:`~poutyne.PeriodicSaveCallback`
    """

    def save_file(self, fd: IO, epoch_number: int, logs: Dict):
        self.model.save_weights(fd)

    def restore(self, fd: IO):
        self.model.load_weights(fd)


class OptimizerCheckpoint(PeriodicSaveCallback):
    """
    Save the state of the optimizer after every epoch. The optimizer can be reloaded as follows.

    .. code-block:: python

        model = Model(model, optimizer, loss_function)
        model.load_optimizer_state(filename)

    See :class:`~poutyne.PeriodicSaveCallback` for the arguments' descriptions.

    See:
        :class:`~poutyne.PeriodicSaveCallback`
    """

    def save_file(self, fd: IO, epoch_number: int, logs: Dict):
        self.model.save_optimizer_state(fd)

    def restore(self, fd: IO):
        self.model.load_optimizer_state(fd)


class LRSchedulerCheckpoint(PeriodicSaveCallback):
    """
    Save the state of an LR scheduler callback after every epoch. The LR scheduler callback should
    not be passed to the fit*() methods since it is called by this callback instead. The LR
    scheduler can be reloaded as follows.

    .. code-block:: python

        lr_scheduler = AnLRSchedulerCallback(...)
        lr_scheduler.load_state(filename)

    See :class:`~poutyne.PeriodicSaveCallback` for the arguments' descriptions.

    Args:
        lr_scheduler: An LR scheduler callback.

    See:
        :class:`~poutyne.PeriodicSaveCallback`
    """

    def __init__(self, lr_scheduler: _PyTorchLRSchedulerWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = lr_scheduler

        if not isinstance(self.lr_scheduler, (_PyTorchLRSchedulerWrapper, ReduceLROnPlateau)):
            raise ValueError("Unknown scheduler callback '%s'." % lr_scheduler)

    def save_file(self, fd: IO, epoch_number: int, logs: Dict):
        self.lr_scheduler.save_state(fd)

    def restore(self, fd: IO):
        self.lr_scheduler.load_state(fd)

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


class StateCheckpoint(PeriodicSaveCallback):
    def __init__(self, name_to_stateful, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_to_stateful = name_to_stateful

    def save_file(self, fd: IO, epoch_number: int, logs: Dict):
        states = {k: v.state_dict() for k, v in self.name_to_stateful.items()}
        torch.save(states, fd)

    def restore(self, fd: IO):
        states = torch.load(fd, map_location='cpu')

        unexpected_keys = set(states.keys()) - set(self.name_to_stateful)
        missing_keys = set(self.name_to_stateful) - set(states.keys())
        if len(unexpected_keys) > 0:
            warnings.warn('Unexpected key(s): {}.'.format(', '.join('"{}"'.format(*unexpected_keys))))
        if len(missing_keys) > 0:
            warnings.warn('Missing key(s): {}.'.format(', '.join('"{}"'.format(*missing_keys))))

        for name, state in states.items():
            if name in self.name_to_stateful:
                self.name_to_stateful[name].load_state_dict(state)
