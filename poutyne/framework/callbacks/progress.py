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
from typing import Dict, Callable

from .callbacks import Callback
from .color_formatting import ColorProgress


class ProgressionCallback(Callback):
    """
    Default progression callback used in :class:`~poutyne.Model`. You can use the ``progress_options``
    in :class:`~poutyne.Model` instead of instantiating this callback. If you choose to use this callback
    anyway, make sure to pass ``verbose=False`` to :func:`~poutyne.Model.fit()` or
    :func:`~poutyne.Model.fit_generator()`.

    Args:
        coloring (Union[bool, Dict], optional): If bool, whether to display the progress of the training with
            default colors highlighting.
            If Dict, the field and the color to use as `colorama <https://pypi.org/project/colorama/>`_ . The fields
            are ``text_color``, ``ratio_color``, ``metric_value_color``, ``time_color`` and ``progress_bar_color``.
            In both case, will be ignore if verbose is set to False.
            (Default value = True)
        progress_bar (bool): Whether or not to display a progress bar showing the epoch progress.
            Note that if the size of the output text with the progress bar is larger than the shell output size,
            the formatting could be impacted (a line for every step).
            (Default value = True)
        equal_weights (bool): Whether or not the duration of each step is weighted equally when computing the
            average time of the steps and, thus, the ETA. By default, newer step times receive more weights than
            older step times. Set this to true to have equal weighting instead.
        show_on_valid (bool): Whether or not to display the progression during the validation phase.
            (Default value = True)
        show_every_n_train_steps (Union[str, int]): Show a subset of the training steps. If ``'all'``, show all steps.
            If ``'none'``, do not show the steps (i.e. only show the stats at the end of the epoch). If an integer
            ``n``, only show every n-th steps. (Default value = 'all').
        show_every_n_valid_steps (Union[str, int]): Show a subset of the validation steps. If ``'all'``, show all steps.
            If ``'none'``, do not show the steps (i.e. only show the stats at the end of the epoch). If an integer
            ``n``, only show every n-th steps. (Default value = 'all').
        show_every_n_test_steps (Union[str, int]): Show a subset of the testing steps. If ``'all'``, show all steps.
            If ``'none'``, do not show the steps (i.e. only show the stats at the end of the testing). If an integer
            ``n``, show only every n-th steps. (Default value = 'all').
    """

    EVERY_N_STEPS_CHOICES = ['all', 'none']

    def __init__(
        self,
        *,
        coloring=True,
        progress_bar=True,
        equal_weights=False,
        show_on_valid=True,
        show_every_n_train_steps='all',
        show_every_n_valid_steps='all',
        show_every_n_test_steps='all',
    ) -> None:
        super().__init__()
        self.color_progress = ColorProgress(coloring=coloring)
        self.progress_bar = progress_bar
        self.equal_weights = equal_weights
        self.show_on_valid = show_on_valid
        self.step_times_weighted_sum = 0.0

        assert (
            isinstance(show_every_n_train_steps, int)
            or show_every_n_train_steps in ProgressionCallback.EVERY_N_STEPS_CHOICES
        )
        assert (
            isinstance(show_every_n_valid_steps, int)
            or show_every_n_valid_steps in ProgressionCallback.EVERY_N_STEPS_CHOICES
        )
        assert (
            isinstance(show_every_n_test_steps, int)
            or show_every_n_test_steps in ProgressionCallback.EVERY_N_STEPS_CHOICES
        )
        self.show_every_n_train_steps = show_every_n_train_steps
        self.show_every_n_valid_steps = show_every_n_valid_steps
        self.show_every_n_test_steps = show_every_n_test_steps

        self.train_last_step = None
        self.valid_last_step = None

    def set_params(self, params: Dict):
        super().set_params(params)
        self._train_steps = self.params['steps']
        self._valid_steps = self.params.get('valid_steps')
        self._test_steps = self.params['steps']
        self._predict_steps = self.params['steps']

    def _set_progress_bar(self):
        if self.progress_bar and self.steps is not None:
            self.color_progress.set_progress_bar(self.steps)
        elif self.progress_bar and self.steps is None:
            # Specific case where we don't have steps for the training
            # but we do during valid so we create a progress bar and
            # when we return to the train, it's all messed up
            self.color_progress.close_progress_bar()

    def on_train_begin(self, logs: Dict) -> None:
        self.metrics = ['loss'] + self.model.metrics_names
        self.epochs = self.params['epochs']
        self.steps = self._train_steps

    def on_valid_begin(self, logs: Dict) -> None:
        if self.show_on_valid:
            self.step_times_weighted_sum = 0.0

            self.metrics = ['loss'] + self.model.metrics_names
            self.steps = self._valid_steps

            self._set_progress_bar()

            self.color_progress.on_valid_begin()

    def on_test_begin(self, logs: Dict) -> None:
        self.step_times_weighted_sum = 0.0

        self.metrics = ['loss'] + self.model.metrics_names
        self.steps = self._test_steps

        self._set_progress_bar()

        self.color_progress.on_test_begin()

    def on_predict_begin(self, logs: Dict) -> None:
        self.step_times_weighted_sum = 0.0

        self.metrics = []
        self.steps = self._predict_steps

        self._set_progress_bar()

        self.color_progress.on_predict_begin()

    def on_epoch_begin(self, epoch_number: int, logs: Dict) -> None:
        self.step_times_weighted_sum = 0.0
        self.epoch_number = epoch_number
        self.steps = self._train_steps

        self._set_progress_bar()

        self.color_progress.on_epoch_begin(epoch_number=self.epoch_number, epochs=self.epochs)

    def on_epoch_end(self, epoch_number: int, logs: Dict) -> None:
        self.steps = self._train_steps
        epoch_total_time = logs['time']

        metrics_str = self._get_metrics_string(logs)
        self.color_progress.on_epoch_end(
            total_time=epoch_total_time,
            train_last_steps=self.train_last_step,
            valid_last_steps=self.valid_last_step,
            metrics_str=metrics_str,
        )

    def on_test_end(self, logs: Dict) -> None:
        test_total_time = logs['time']
        progress_fun = self.color_progress.on_test_end

        self._end_progress(logs, test_total_time, progress_fun)

    def on_predict_end(self, logs: Dict) -> None:
        predict_total_time = logs['time']
        progress_fun = self.color_progress.on_predict_end

        self._end_progress(logs, predict_total_time, progress_fun)

    def _test_show_batch_end(self, show_every_n_steps_flag, batch_number):
        return show_every_n_steps_flag == 'all' or (
            isinstance(show_every_n_steps_flag, int) and batch_number % show_every_n_steps_flag == 0
        )

    def on_train_batch_end(self, batch_number: int, logs: Dict) -> None:
        train_step_times_rate = self._compute_step_times_rate(batch_number, logs)
        progress_batch_end_fun = self.color_progress.on_train_batch_end

        do_print = self._test_show_batch_end(self.show_every_n_train_steps, batch_number)
        self._batch_end_progress(
            logs=logs,
            step_times_rate=train_step_times_rate,
            batch_number=batch_number,
            func=progress_batch_end_fun,
            do_print=do_print,
        )
        self.train_last_step = batch_number

    def on_valid_batch_end(self, batch_number: int, logs: Dict) -> None:
        if self.show_on_valid:
            valid_step_times_rate = self._compute_step_times_rate(batch_number, logs)
            progress_batch_end_fun = self.color_progress.on_valid_batch_end

            do_print = self._test_show_batch_end(self.show_every_n_valid_steps, batch_number)
            self._batch_end_progress(
                logs=logs,
                step_times_rate=valid_step_times_rate,
                batch_number=batch_number,
                func=progress_batch_end_fun,
                do_print=do_print,
            )

        self.valid_last_step = batch_number

    def on_test_batch_end(self, batch_number: int, logs: Dict) -> None:
        test_step_times_rate = self._compute_step_times_rate(batch_number, logs)
        progress_batch_end_fun = self.color_progress.on_test_batch_end

        do_print = self._test_show_batch_end(self.show_every_n_test_steps, batch_number)
        self._batch_end_progress(
            logs=logs,
            step_times_rate=test_step_times_rate,
            batch_number=batch_number,
            func=progress_batch_end_fun,
            do_print=do_print,
        )

    def on_predict_batch_end(self, batch_number: int, logs: Dict) -> None:
        predict_step_times_rate = self._compute_step_times_rate(batch_number, logs)
        progress_batch_end_fun = self.color_progress.on_predict_batch_end

        do_print = self._test_show_batch_end(self.show_every_n_test_steps, batch_number)
        self._batch_end_progress(
            logs=logs,
            step_times_rate=predict_step_times_rate,
            batch_number=batch_number,
            func=progress_batch_end_fun,
            do_print=do_print,
        )

    def _get_metrics_string(self, logs: Dict) -> str:
        train_metrics_str_gen = (f'{k}: {logs[k]:f}' for k in self.metrics if logs.get(k) is not None)
        val_metrics_str_gen = (
            f"{'val_' + k}: {logs['val_' + k]:f}" for k in self.metrics if logs.get('val_' + k) is not None
        )
        test_metrics_str_gen = (
            f"{'test_' + k}: {logs['test_' + k]:f}" for k in self.metrics if logs.get('test_' + k) is not None
        )
        return ', '.join(itertools.chain(train_metrics_str_gen, val_metrics_str_gen, test_metrics_str_gen))

    def _compute_step_times_rate(self, batch_number: int, logs: Dict) -> float:
        if self.equal_weights:
            self.step_times_weighted_sum += logs['time']
            step_times_rate = self.step_times_weighted_sum / batch_number
        else:
            self.step_times_weighted_sum += batch_number * logs['time']
            normalizing_factor = (batch_number * (batch_number + 1)) / 2
            step_times_rate = self.step_times_weighted_sum / normalizing_factor
        return step_times_rate

    def _end_progress(self, logs: Dict, total_time: float, func: Callable) -> None:
        """
        Update the progress at the end of a test or valid phase.
        """
        metrics_str = self._get_metrics_string(logs)
        if self.steps is not None:
            func(total_time=total_time, steps=self.steps, metrics_str=metrics_str)
        else:
            func(total_time=total_time, steps=self.last_step, metrics_str=metrics_str)

    def _batch_end_progress(
        self, *, logs: Dict, step_times_rate: float, batch_number: int, func: Callable, do_print: bool
    ) -> None:
        """
        Update the progress at the end of train, valid or test batch.
        """
        metrics_str = self._get_metrics_string(logs)
        if self.steps is not None:
            remaining_time = step_times_rate * (self.steps - batch_number)
            func(
                remaining_time=remaining_time,
                batch_number=batch_number,
                metrics_str=metrics_str,
                steps=self.steps,
                do_print=do_print,
            )
        else:
            func(remaining_time=step_times_rate, batch_number=batch_number, metrics_str=metrics_str, do_print=do_print)
            self.last_step = batch_number


class EpochProgressionCallback(Callback):
    EVERY_N_EPOCHS_CHOICES = ['all', 'none']

    def __init__(self, *, coloring=True, show_every_n_epochs='all') -> None:
        super().__init__()
        self.color_progress = ColorProgress(coloring=coloring)

        assert (
            isinstance(show_every_n_epochs, int)
            or show_every_n_epochs in EpochProgressionCallback.EVERY_N_EPOCHS_CHOICES
        )
        self.show_every_n_epochs = show_every_n_epochs

    def set_params(self, params: Dict):
        super().set_params(params)
        self._train_steps = self.params['steps']
        self._valid_steps = self.params.get('valid_steps')

    def on_train_begin(self, logs: Dict) -> None:
        self.metrics = ['loss'] + self.model.metrics_names
        self.epochs = self.params['epochs']

    def _test_show_epoch(self, epoch_number):
        return self.show_every_n_epochs != 'none' and (
            self.show_every_n_epochs == 'all' or epoch_number % self.show_every_n_epochs == 0
        )

    def on_epoch_begin(self, epoch_number: int, logs: Dict) -> None:
        if self._test_show_epoch(epoch_number):
            self.color_progress.on_epoch_begin(epoch_number=epoch_number, epochs=self.epochs)

    def on_epoch_end(self, epoch_number: int, logs: Dict) -> None:
        if self._test_show_epoch(epoch_number):
            self.color_progress.on_epoch_end(
                total_time=logs['time'],
                train_last_steps=self._train_steps,
                valid_last_steps=self._valid_steps,
                metrics_str=self._get_metrics_string(logs),
            )

    def _get_metrics_string(self, logs: Dict) -> str:
        train_metrics_str_gen = (f'{k}: {logs[k]:f}' for k in self.metrics if logs.get(k) is not None)
        val_metrics_str_gen = (
            f"{'val_' + k}: {logs['val_' + k]:f}" for k in self.metrics if logs.get('val_' + k) is not None
        )
        return ', '.join(itertools.chain(train_metrics_str_gen, val_metrics_str_gen))
