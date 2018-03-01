import warnings

from .callbacks import Callback


class BestModelRestore(Callback):
    """
    Restore the weights of the best model at the end of the training depending
    on a monitored quantity.

    Args:
        monitor (string): Quantity to monitor. (Default value = 'val_loss')
        mode (string): One of {min, max}.
            Whether the monitored has to be maximized or minimized. For
            instance, for `val_accuracy`, this should be `max`, and for
            `val_loss`, this should be `min`, etc. (Default value = 'min')
    """
    def __init__(self, monitor='val_loss', mode='min'):
        self.monitor = monitor

        if mode not in ['min', 'max']:
            raise ValueError("Invalid mode '%s'" % mode)
        if mode == 'min':
            self.monitor_op = lambda x,y: x < y
            self.current_best = float('Inf')
        elif mode == 'max':
            self.monitor_op = lambda x,y: x > y
            self.current_best = -float('Inf')
        self.best_weights = None

    def on_epoch_end(self, epoch, logs):
        if self.monitor_op(logs[self.monitor], self.current_best):
            self.current_best = logs[self.monitor]
            self.best_weights = self.model.get_weight_copies()

    def on_train_end(self, logs):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
        else:
            warnings.warn('No  weights to restore!')
