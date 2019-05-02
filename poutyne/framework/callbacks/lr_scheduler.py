"""
Poutyne's callbacks for learning rate schedulers are just wrappers around `PyTorch's learning
rate schedulers <http://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate>`_
and thus have the same arguments except for the optimizer that has to be
omitted.
"""
import inspect
import torch.optim.lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

from .callbacks import Callback


class _PyTorchLRSchedulerWrapper(Callback):
    def __init__(self, torch_lr_scheduler, *args, **kwargs):
        super().__init__()
        self.torch_lr_scheduler = torch_lr_scheduler
        self.args = args
        self.kwargs = kwargs
        self.scheduler = None
        self.loaded_state = None

    def on_train_begin(self, logs):
        optimizer = self.model.optimizer
        self.scheduler = self.torch_lr_scheduler(optimizer, *self.args, **self.kwargs)

        # Load state if the scheduler was not initialized when the user asked
        # to load its state
        if self.loaded_state is not None:
            self.scheduler.load_state_dict(self.loaded_state)
            self.loaded_state = None

    def on_epoch_end(self, epoch, logs):
        self.scheduler.step(epoch)

    def load_state(self, f):
        if self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(f, map_location='cpu'))
        else:
            self.loaded_state = torch.load(f, map_location='cpu')

    def save_state(self, f):
        torch.save(self.scheduler.state_dict(), f)


for name, cls in torch.optim.lr_scheduler.__dict__.items():
    if inspect.isclass(cls) and issubclass(cls, _LRScheduler) and cls != _LRScheduler:
        # pylint: disable=exec-used
        exec('''class {name}(_PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch {name}
        <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.{name}>`_
    """
    def __init__(self, *args, **kwargs):
        super().__init__(torch.optim.lr_scheduler.{name}, *args, **kwargs)'''.format(name=name),
             globals(),
             locals())


class ReduceLROnPlateau(Callback):
    """
    Args:
        monitor (string): The quantity to monitor. (Default value = 'val_loss')
    See:
        `PyTorch ReduceLROnPlateau
        <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_
    """
    def __init__(self, *args, monitor='val_loss', **kwargs):
        super().__init__()
        self.monitor = monitor
        self.args = args
        self.kwargs = kwargs
        self.scheduler = None
        self.loaded_state = None

    def on_train_begin(self, logs):
        optimizer = self.model.optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, *self.args, **self.kwargs
        )

        # Load state if the scheduler was not initialized when the user asked
        # to load its state
        if self.loaded_state is not None:
            self.scheduler.load_state_dict(self.loaded_state)
            self.loaded_state = None

    def on_epoch_end(self, epoch, logs):
        self.scheduler.step(logs[self.monitor], epoch)

    def load_state(self, f):
        if self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(f, map_location='cpu'))
        else:
            self.loaded_state = torch.load(f, map_location='cpu')

    def save_state(self, f):
        torch.save(self.scheduler.state_dict(), f)
