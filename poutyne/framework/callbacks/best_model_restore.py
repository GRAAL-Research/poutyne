import warnings
from typing import Dict

from .callbacks import Callback


class BestModelRestore(Callback):
    """
    Restore the weights of the best model at the end of the training depending on a monitored quantity.

    Args:
        monitor (str): Quantity to monitor. (Default value = 'val_loss')
        mode (str): One of {'min', 'max'}.
            Whether the monitored has to be maximized or minimized. For instance, for `val_accuracy`,
            this should be `max`, and for `val_loss`, this should be `min`, etc.
            (Default value = 'min')
        verbose (bool): Whether to display a message when the model has improved or when restoring
            the best model.
            (Default value = False)
    """

    def __init__(self, *, monitor: str = 'val_loss', mode: str = 'min', verbose: bool = False):
        super().__init__()
        self.monitor = monitor

        if mode not in ['min', 'max']:
            raise ValueError("Invalid mode '%s'" % mode)
        if mode == 'min':
            self.monitor_op = lambda x, y: x < y
            self.current_best = float('Inf')
        elif mode == 'max':
            self.monitor_op = lambda x, y: x > y
            self.current_best = -float('Inf')
        self.best_weights = None
        self.verbose = verbose

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        if self.monitor_op(logs[self.monitor], self.current_best):
            old_best = self.current_best
            self.current_best = logs[self.monitor]

            if self.verbose:
                print('Epoch %d: %s improved from %0.5f to %0.5f' %
                      (epoch_number, self.monitor, old_best, self.current_best))
            self.best_weights = self.model.get_weight_copies()

    def on_train_end(self, logs: Dict):
        if self.best_weights is not None:
            if self.verbose:
                print('Restoring best model')
            self.model.set_weights(self.best_weights)
        else:
            warnings.warn('No  weights to restore!')
