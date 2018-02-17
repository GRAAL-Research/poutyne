import timeit
import sys

import itertools

from .callbacks import Callback


class ProgressionCallback(Callback):
    def on_train_begin(self, logs):
        self.metrics = ['loss'] + self.model.metrics_names
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']

    def on_train_end(self, logs):
        pass

    def on_epoch_begin(self, epoch, logs):
        self.step_times_sum = 0.
        self.epoch = epoch
        self.epoch_begin_time = timeit.default_timer()
        sys.stdout.write("\rEpoch %d/%d" % (self.epoch, self.epochs))
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs):
        self.epoch_end_time = timeit.default_timer()
        self.epoch_total_time = self.epoch_end_time - self.epoch_begin_time

        metrics_str = self._get_metrics_string(logs)
        print("\rEpoch %d/%d %.2fs Step %d/%d: %s" % (
        self.epoch, self.epochs, self.epoch_total_time, self.steps, self.steps, metrics_str))

    def on_batch_begin(self, step, logs):
        self.batch_begin_time = timeit.default_timer()

    def on_batch_end(self, step, logs):
        self.batch_end_time = timeit.default_timer()
        self.step_times_sum += self.batch_end_time - self.batch_begin_time

        times_mean = self.step_times_sum / step
        remaining_time = times_mean * (self.steps - step)

        metrics_str = self._get_metrics_string(logs)
        sys.stdout.write("\rEpoch %d/%d ETA %.0fs Step %d/%d: %s" % (
        self.epoch, self.epochs, remaining_time, step, self.steps, metrics_str))
        sys.stdout.flush()

    def _get_metrics_string(self, logs):
        train_metrics_str_gen = ('{}: {:f}'.format(k, logs[k]) for k in self.metrics if logs.get(k))
        val_metrics_str_gen = ('{}: {:f}'.format('val_' + k, logs['val_' + k]) for k in self.metrics if
                               logs.get('val_' + k))
        return ', '.join(itertools.chain(train_metrics_str_gen, val_metrics_str_gen))
