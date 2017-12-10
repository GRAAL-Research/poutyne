import torch.optim.lr_scheduler

from .callbacks import Callback

class ReduceLROnPlateau(Callback):
    def __init__(self, monitor='val_loss', *args, **kwargs):
        self.monitor = monitor
        self.args = args
        self.kwargs = kwargs

    def on_train_begin(self, logs=None):
        optimizer = self.model.optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, *self.args, **self.kwargs)

    def on_epoch_end(self, epoch, logs=None):
        self.scheduler.step(logs[-1][self.monitor])
