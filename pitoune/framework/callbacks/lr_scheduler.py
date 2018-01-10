import torch.optim.lr_scheduler

from .callbacks import Callback

class PyTorchLRSchedulerWrapper(Callback):
    def __init__(self, torch_lr_scheduler, monitor='val_loss', *args, **kwargs):
        self.torch_lr_scheduler
        self.monitor = monitor
        self.args = args
        self.kwargs = kwargs

    def on_train_begin(self, logs=None):
        optimizer = self.model.optimizer
        self.scheduler = self.torch_lr_scheduler(optimizer, *self.args, **self.kwargs)

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step(logs[-1][self.monitor])

class LambdaLR(PyTorchLRSchedulerWrapper):
    def __init__(self, *args, **kwargs):
        super(PyTorchLRSchedulerWrapper, self).__init__(torch.optim.lr_scheduler.LambdaLR, *args, **kwargs)

class StepLR(PyTorchLRSchedulerWrapper):
    def __init__(self, *args, **kwargs):
        super(PyTorchLRSchedulerWrapper, self).__init__(torch.optim.lr_scheduler.StepLR, *args, **kwargs)

class MultiStepLR(PyTorchLRSchedulerWrapper):
    def __init__(self, *args, **kwargs):
        super(PyTorchLRSchedulerWrapper, self).__init__(torch.optim.lr_scheduler.MultiStepLR, *args, **kwargs)

class ExponentialLR(PyTorchLRSchedulerWrapper):
    def __init__(self, *args, **kwargs):
        super(PyTorchLRSchedulerWrapper, self).__init__(torch.optim.lr_scheduler.ExponentialLR, *args, **kwargs)

class CosineAnnealingLR(PyTorchLRSchedulerWrapper):
    def __init__(self, *args, **kwargs):
        super(PyTorchLRSchedulerWrapper, self).__init__(torch.optim.lr_scheduler.CosineAnnealingLR, *args, **kwargs)

class ReduceLROnPlateau(PyTorchLRSchedulerWrapper):
    def __init__(self, *args, **kwargs):
        super(PyTorchLRSchedulerWrapper, self).__init__(torch.optim.lr_scheduler.ReduceLROnPlateau, *args, **kwargs)
