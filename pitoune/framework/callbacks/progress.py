import timeit
import sys

import itertools

from .callbacks import Callback

class ProgressionCallback(Callback):
    def on_train_begin(self, logs=None):
        self.metrics = ['loss'] + self.params['metrics']
        self.n_epochs = self.params['n_epochs']
        self.steps_per_epoch = self.params['steps_per_epoch']

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self.step_times_sum = 0.
        self.epoch = epoch
        self.epoch_begin_time = timeit.default_timer()
        sys.stdout.write("\rEpoch %d/%d" % (self.epoch, self.n_epochs))
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_end_time = timeit.default_timer()
        self.epoch_total_time = self.epoch_end_time - self.epoch_begin_time

        metrics_str = self._get_metrics_string(logs)
        print("\rEpoch %d/%d %ds Step %d/%d: %s" % (self.epoch, self.n_epochs, self.epoch_total_time, self.steps_per_epoch, self.steps_per_epoch, metrics_str))

    def on_batch_begin(self, step, logs=None):
        self.batch_begin_time = timeit.default_timer()

    def on_batch_end(self, step, logs=None):
        self.batch_end_time = timeit.default_timer()
        self.step_times_sum += self.batch_end_time - self.batch_begin_time

        times_mean = self.step_times_sum / step
        remaining_time = times_mean * (self.steps_per_epoch - step)

        metrics_str = self._get_metrics_string(logs)
        sys.stdout.write("\rEpoch %d/%d ETA %ds Step %d/%d: %s" % (self.epoch, self.n_epochs, remaining_time, step, self.steps_per_epoch, metrics_str))
        sys.stdout.flush()

    def _get_metrics_string(self, logs):
        train_metrics_str_gen = ('{}: {:f}'.format(k, logs[-1][k]) for k in self.metrics if logs[-1].get(k))
        val_metrics_str_gen = ('{}: {:f}'.format('val_' + k, logs[-1]['val_' + k]) for k in self.metrics if logs[-1].get('val_' + k))
        return ', '.join(itertools.chain(train_metrics_str_gen, val_metrics_str_gen))
