import itertools
from typing import Dict

from .callbacks import Callback
from poutyne.framework.callbacks.color_formatting import ColorProgress


class ProgressionCallback(Callback):
    def __init__(self, coloring=False):
        super().__init__()

        self.color_progress = ColorProgress(coloring)

    def on_train_begin(self, logs: Dict):
        self.metrics = ['loss'] + self.model.metrics_names
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']

    def on_epoch_begin(self, epoch_number: int, logs: Dict):
        self.step_times_sum = 0.
        self.epoch_number = epoch_number

    def on_epoch_end(self, epoch_number: int, logs: Dict):
        epoch_total_time = logs['time']

        metrics_str = self._get_metrics_string(logs)
        if self.steps is not None:
            self.color_progress.on_epoch_end(self.epoch_number, self.epochs, epoch_total_time, self.steps, metrics_str)
        else:
            self.color_progress.on_epoch_end(self.epoch_number, self.epochs, epoch_total_time, self.last_step,
                                             metrics_str)

    def on_train_batch_end(self, batch_number: int, logs: Dict):
        self.step_times_sum += logs['time']

        metrics_str = self._get_metrics_string(logs)

        times_mean = self.step_times_sum / batch_number
        if self.steps is not None:
            remaining_time = times_mean * (self.steps - batch_number)

            self.color_progress.on_train_batch_end_steps(self.epoch_number, self.epochs, remaining_time, batch_number,
                                                         self.steps, metrics_str)
        else:
            self.color_progress.on_train_batch_end(self.epoch_number, self.epochs, times_mean, batch_number,
                                                   metrics_str)
            self.last_step = batch_number

    def _get_metrics_string(self, logs: Dict):
        train_metrics_str_gen = ('{}: {:f}'.format(k, logs[k]) for k in self.metrics if logs.get(k) is not None)
        val_metrics_str_gen = ('{}: {:f}'.format('val_' + k, logs['val_' + k]) for k in self.metrics
                               if logs.get('val_' + k) is not None)
        return ', '.join(itertools.chain(train_metrics_str_gen, val_metrics_str_gen))
