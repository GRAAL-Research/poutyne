"""
Poutyne's callbacks for learning rate schedulers are just wrappers around `PyTorch's learning rate
schedulers <http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate>`_ and thus have
the same arguments except for the optimizer that has to be omitted.
"""
import sys
import inspect
import torch.optim.lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .callbacks import Callback


class LRScheduler(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) > 0 and isinstance(args[0], Optimizer):
            raise ValueError("In the LR scheduler callbacks, the optimizer is "
                             "automatically passed to the PyTorch's LR scheduler. "
                             "You must remove it from the arguments.")
        self.args = args
        self.kwargs = kwargs
        self.scheduler = None
        self.loaded_state = None

    def on_train_begin(self, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        self.scheduler.step(epoch)

    def load_state(self, f):
        if self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(f, map_location='cpu'))
        else:
            self.loaded_state = torch.load(f, map_location='cpu')

    def save_state(self, f):
        torch.save(self.scheduler.state_dict(), f)


class _PyTorchLRSchedulerWrapper(LRScheduler):
    def __init__(self, torch_lr_scheduler, *args, **kwargs):
        super(_PyTorchLRSchedulerWrapper, self).__init__(*args, **kwargs)
        self.torch_lr_scheduler = torch_lr_scheduler

    def on_train_begin(self, logs):
        optimizer = self.model.optimizer
        self.scheduler = self.torch_lr_scheduler(optimizer, *self.args, **self.kwargs)

        # Load state if the scheduler was not initialized when the user asked
        # to load its state
        if self.loaded_state is not None:
            self.scheduler.load_state_dict(self.loaded_state)
            self.loaded_state = None


def new_init(torch_lr_scheduler):
    def f(self, *args, **kwargs):
        super(type(self), self).__init__(torch_lr_scheduler, *args, **kwargs)

    return f


for name, module_cls in torch.optim.lr_scheduler.__dict__.items():
    if inspect.isclass(module_cls) and \
            issubclass(module_cls, _LRScheduler) and \
            module_cls != _LRScheduler:
        _new_cls = type(
            name, (_PyTorchLRSchedulerWrapper, ), {
                '__init__':
                new_init(module_cls),
                '__doc__':
                """
                See:
                    :class:`~torch.optim.lr_scheduler.{name}`
                """.format(name=name)
            })
        setattr(sys.modules[__name__], name, _new_cls)


class ReduceLROnPlateau(LRScheduler):
    """
    Args:
        monitor (str): The quantity to monitor. (Default value = 'val_loss')
    See:
        :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`
    """

    def __init__(self, *args, monitor='val_loss', **kwargs):
        super(ReduceLROnPlateau, self).__init__(*args, **kwargs)
        self.monitor = monitor

    def on_train_begin(self, logs):
        optimizer = self.model.optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, *self.args, **self.kwargs)

        # Load state if the scheduler was not initialized when the user asked
        # to load its state
        if self.loaded_state is not None:
            self.scheduler.load_state_dict(self.loaded_state)
            self.loaded_state = None

    def on_epoch_end(self, epoch, logs):
        self.scheduler.step(logs[self.monitor], epoch)