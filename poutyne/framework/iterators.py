import itertools
import timeit

import numpy as np


class Step:
    def __init__(self, number):
        self.number = number

        self.loss = None
        self.metrics = None
        self.size = None


def cycle(iterable):  # Equivalent to itertools cycle, without any extra memory requirement
    while True:
        for x in iterable:
            yield x


def _get_step_iterator(steps, generator):
    count_iterator = range(1, steps + 1) if steps is not None else itertools.count(1)
    generator = cycle(generator) if steps is not None else generator
    return zip(count_iterator, generator)


class StepIterator:
    def __init__(self, generator, steps_per_epoch, batch_metrics_names, callback=None, mode=None):
        # pylint: disable=too-many-arguments
        self.generator = generator
        self.steps_per_epoch = steps_per_epoch
        self.batch_metrics_names = batch_metrics_names

        self.on_batch_begin = lambda *_: None
        self.on_batch_end = lambda *_: None

        if mode == 'train':
            self.on_batch_begin = callback.on_train_batch_begin
            self.on_batch_end = callback.on_train_batch_end
        elif mode == 'test':
            self.on_batch_begin = callback.on_test_batch_begin
            self.on_batch_end = callback.on_test_batch_end

        self.losses_sum = 0.
        self.metrics_sum = np.zeros(len(self.batch_metrics_names))
        self.sizes_sum = 0.
        self.epoch_metrics = None

    @property
    def loss(self):
        return self.losses_sum / self.sizes_sum

    @property
    def metrics(self):
        return self.metrics_sum / self.sizes_sum

    def __iter__(self):
        time_since_last_batch = timeit.default_timer()
        for step, data in _get_step_iterator(self.steps_per_epoch, self.generator):
            self.on_batch_begin(step, {})

            step_data = Step(step)
            yield step_data, data

            self.losses_sum += step_data.loss * step_data.size
            self.metrics_sum += step_data.metrics * step_data.size
            self.sizes_sum += step_data.size

            batch_end_time = timeit.default_timer()
            batch_total_time = batch_end_time - time_since_last_batch
            time_since_last_batch = batch_end_time

            metrics_dict = dict(zip(self.batch_metrics_names, step_data.metrics))

            batch_logs = {
                'batch': step,
                'size': step_data.size,
                'time': batch_total_time,
                'loss': step_data.loss,
                **metrics_dict
            }

            self.on_batch_end(step, batch_logs)


class EpochIterator:
    """
    Epoch iterator used in the training phase of the model.
    """

    def __init__(self,
                 train_generator,
                 valid_generator,
                 *,
                 epochs,
                 steps_per_epoch,
                 validation_steps,
                 initial_epoch=1,
                 callback,
                 batch_metrics_names,
                 epoch_metrics_names):
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.epochs = epochs
        self._init_steps(train_generator, valid_generator, steps_per_epoch, validation_steps)

        self.initial_epoch = initial_epoch
        self.callback = callback
        self.batch_metrics_names = batch_metrics_names
        self.epoch_metrics_names = epoch_metrics_names
        self.epoch_logs = []
        self.stop_training = False

        params = {'epochs': self.epochs, 'steps': self.steps_per_epoch}
        self.callback.set_params(params)

    def _init_steps(self, train_generator, valid_generator, steps_per_epoch, validation_steps):
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps

        if valid_generator is not None:
            if validation_steps is None:
                if hasattr(valid_generator, '__len__'):
                    self.validation_steps = len(valid_generator)
                elif steps_per_epoch is not None:
                    self.validation_steps = steps_per_epoch
        if steps_per_epoch is None and hasattr(train_generator, '__len__'):
            self.steps_per_epoch = len(train_generator)

    def __iter__(self):
        self.callback.on_train_begin({})
        for epoch in range(self.initial_epoch, self.epochs + 1):
            self.callback.on_epoch_begin(epoch, {})
            epoch_begin_time = timeit.default_timer()

            train_step_iterator = StepIterator(self.train_generator,
                                               self.steps_per_epoch,
                                               self.batch_metrics_names,
                                               self.callback,
                                               mode="train")

            valid_step_iterator = None
            if self.valid_generator is not None:
                valid_step_iterator = StepIterator(self.valid_generator, self.validation_steps,
                                                   self.batch_metrics_names)

            yield train_step_iterator, valid_step_iterator

            val_dict = {}
            if valid_step_iterator is not None:
                val_metrics_dict = {
                    'val_' + metric_name: metric
                    for metric_name, metric in zip(self.batch_metrics_names, valid_step_iterator.metrics)
                }
                val_metrics_dict.update({
                    'val_' + metric_name: metric
                    for metric_name, metric in zip(self.epoch_metrics_names, valid_step_iterator.epoch_metrics)
                })

                val_dict = {'val_loss': valid_step_iterator.loss, **val_metrics_dict}

            epoch_total_time = timeit.default_timer() - epoch_begin_time
            metrics_dict = dict(zip(self.batch_metrics_names, train_step_iterator.metrics))
            metrics_dict.update(dict(zip(self.epoch_metrics_names, train_step_iterator.epoch_metrics)))

            epoch_log = {
                'epoch': epoch,
                'loss': train_step_iterator.loss,
                'time': epoch_total_time,
                **metrics_dict,
                **val_dict
            }
            self.callback.on_epoch_end(epoch, epoch_log)

            self.epoch_logs.append(epoch_log)

            if self.stop_training:
                break

        self.callback.on_train_end({})
