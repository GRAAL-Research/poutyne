import torch.optim.lr_scheduler

from .callbacks import Callback


class PyTorchLRSchedulerWrapper(Callback):
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

class LambdaLR(PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch LambdaLR
        <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.LambdaLR>`_
    """
    def __init__(self, *args, **kwargs):
        super().__init__(torch.optim.lr_scheduler.LambdaLR, *args, **kwargs)


class StepLR(PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch StepLR
        <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.StepLR>`_
    """
    def __init__(self, *args, **kwargs):
        super().__init__(torch.optim.lr_scheduler.StepLR, *args, **kwargs)


class MultiStepLR(PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch MultiStepLR
        <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.MultiStepLR>`_
    """
    def __init__(self, *args, **kwargs):
        super().__init__(torch.optim.lr_scheduler.MultiStepLR, *args, **kwargs)


class ExponentialLR(PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch ExponentialLR
        <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ExponentialLR>`_
    """
    def __init__(self, *args, **kwargs):
        super().__init__(torch.optim.lr_scheduler.ExponentialLR, *args, **kwargs)


class CosineAnnealingLR(PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch CosineAnnealingLR
        <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_
    """
    def __init__(self, *args, **kwargs):
        super().__init__(torch.optim.lr_scheduler.CosineAnnealingLR, *args, **kwargs)

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
