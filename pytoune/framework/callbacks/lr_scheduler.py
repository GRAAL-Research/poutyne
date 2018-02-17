import torch.optim.lr_scheduler

from .callbacks import Callback


class PyTorchLRSchedulerWrapper(Callback):
    def __init__(self, torch_lr_scheduler, *args, **kwargs):
        self.torch_lr_scheduler = torch_lr_scheduler
        self.args = args
        self.kwargs = kwargs

    def on_train_begin(self, logs):
        optimizer = self.model.optimizer
        self.scheduler = self.torch_lr_scheduler(optimizer, *self.args, **self.kwargs)

    def on_epoch_end(self, epoch, logs):
        self.scheduler.step(epoch)


class LambdaLR(PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch LambdaLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.LambdaLR>`_
    """
    def __init__(self, *args, **kwargs):
        super(LambdaLR, self).__init__(torch.optim.lr_scheduler.LambdaLR, *args, **kwargs)


class StepLR(PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch StepLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.StepLR>`_
    """
    def __init__(self, *args, **kwargs):
        super(StepLR, self).__init__(torch.optim.lr_scheduler.StepLR, *args, **kwargs)


class MultiStepLR(PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch MultiStepLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.MultiStepLR>`_
    """
    def __init__(self, *args, **kwargs):
        super(MultiStepLR, self).__init__(torch.optim.lr_scheduler.MultiStepLR, *args, **kwargs)


class ExponentialLR(PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch ExponentialLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ExponentialLR>`_
    """
    def __init__(self, *args, **kwargs):
        super(ExponentialLR, self).__init__(torch.optim.lr_scheduler.ExponentialLR, *args, **kwargs)


class CosineAnnealingLR(PyTorchLRSchedulerWrapper):
    """
    See:
        `PyTorch CosineAnnealingLR <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR>`_
    """
    def __init__(self, *args, **kwargs):
        super(CosineAnnealingLR, self).__init__(torch.optim.lr_scheduler.CosineAnnealingLR, *args, **kwargs)


class ReduceLROnPlateau(Callback):
    """
    Args:
        monitor (string): The quantity to monitor. (Default value = 'val_loss')
    See:
        `PyTorch ReduceLROnPlateau <http://pytorch.org/docs/master/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau>`_
    """
    def __init__(self, *args, monitor='val_loss', **kwargs):
        self.monitor = monitor
        self.args = args
        self.kwargs = kwargs

    def on_train_begin(self, logs):
        optimizer = self.model.optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, *self.args, **self.kwargs)

    def on_epoch_end(self, epoch, logs):
        self.scheduler.step(logs[self.monitor], epoch)
