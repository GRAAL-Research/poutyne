"""
Copyright (c) 2022 Poutyne and all respective contributors.

Each contributor holds copyright over their respective contributions. The project versioning (Git)
records all such contribution source information.

This file is part of Poutyne.

Poutyne is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version.

Poutyne is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with Poutyne. If not, see
<https://www.gnu.org/licenses/>.
"""

import itertools
import timeit


class Step:
    def __init__(self, number):
        self.number = number

        self.loss = None
        self.batch_metrics = None
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
    def __init__(
        self,
        generator,
        steps_per_epoch,
        batch_metrics_names,
        epoch_metrics_names,
        callback=None,
        mode=None,
    ):
        # pylint: disable=too-many-arguments
        self.generator = generator
        self.steps_per_epoch = steps_per_epoch
        self.prefix = "" if mode == "train" else f"{mode}_"
        self.batch_metrics_names = [self.prefix + metric_name for metric_name in batch_metrics_names]
        self.epoch_metrics_names = [self.prefix + metric_name for metric_name in epoch_metrics_names]

        self.on_batch_begin = lambda *_: None
        self.on_batch_end = lambda *_: None

        if mode == 'train':
            self.on_batch_begin = callback.on_train_batch_begin
            self.on_batch_end = callback.on_train_batch_end
        elif mode == 'test':
            self.on_batch_begin = callback.on_test_batch_begin
            self.on_batch_end = callback.on_test_batch_end
        elif mode == 'val':
            self.on_batch_begin = callback.on_valid_batch_begin
            self.on_batch_end = callback.on_valid_batch_end

        self.loss = None
        self.batch_metrics = None
        self.epoch_metrics = None

    @property
    def metrics_logs(self):
        logs = dict(zip(self.batch_metrics_names, self.batch_metrics))
        logs.update(dict(zip(self.epoch_metrics_names, self.epoch_metrics)))

        logs = {f'{self.prefix}loss': self.loss, **logs}
        return logs

    def __iter__(self):
        time_since_last_batch = timeit.default_timer()
        for step, data in _get_step_iterator(self.steps_per_epoch, self.generator):
            self.on_batch_begin(step, {})

            step_data = Step(step)
            yield step_data, data

            batch_end_time = timeit.default_timer()
            batch_total_time = batch_end_time - time_since_last_batch
            time_since_last_batch = batch_end_time

            metrics_log = dict(zip(self.batch_metrics_names, step_data.batch_metrics))
            batch_logs = {
                'batch': step,
                'size': step_data.size,
                'time': batch_total_time,
                f'{self.prefix}loss': step_data.loss,
                **metrics_log,
            }

            self.on_batch_end(step, batch_logs)


class EpochIterator:
    """
    Epoch iterator used in the training phase of the model.
    """

    def __init__(
        self,
        model,
        train_generator,
        valid_generator,
        *,
        epochs,
        steps_per_epoch,
        validation_steps,
        initial_epoch=1,
        callback,
        batch_metrics_names,
        epoch_metrics_names,
    ):
        self.model = model
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.epochs = epochs
        self._init_steps(train_generator, valid_generator, steps_per_epoch, validation_steps)

        self.initial_epoch = initial_epoch
        self.callback = callback
        self.batch_metrics_names = batch_metrics_names
        self.epoch_metrics_names = epoch_metrics_names
        self.epoch_logs = []

        params = {'epochs': self.epochs, 'steps': self.steps_per_epoch}
        if self.validation_steps is not None:
            params.update({'valid_steps': self.validation_steps})

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

            train_step_iterator = StepIterator(
                self.train_generator,
                self.steps_per_epoch,
                self.batch_metrics_names,
                self.epoch_metrics_names,
                self.callback,
                mode="train",
            )

            valid_step_iterator = None
            if self.valid_generator is not None:
                valid_step_iterator = StepIterator(
                    self.valid_generator,
                    self.validation_steps,
                    self.batch_metrics_names,
                    self.epoch_metrics_names,
                    self.callback,
                    mode="val",
                )

            yield train_step_iterator, valid_step_iterator

            epoch_total_time = timeit.default_timer() - epoch_begin_time

            val_metrics_log = {} if valid_step_iterator is None else valid_step_iterator.metrics_logs
            epoch_log = {
                'epoch': epoch,
                'time': epoch_total_time,
                **train_step_iterator.metrics_logs,
                **val_metrics_log,
            }
            self.callback.on_epoch_end(epoch, epoch_log)

            self.epoch_logs.append(epoch_log)

            if self.model.stop_training:
                break

        self.callback.on_train_end({})
