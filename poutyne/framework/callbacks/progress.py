import itertools
from typing import Dict

from .callbacks import Callback
from .color_formatting import ColorProgress


class ProgressionCallback(Callback):
    """
    Default progression callback used in :class:`~poutyne.Model`. You can use the ``progress_options``
    in :class:`~poutyne.Model` instead of instanciating this callback. If you choose to use this callback
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
    """

    def __init__(self, *, coloring=True, progress_bar=True, equal_weights=False) -> None:
        super().__init__()
        self.color_progress = ColorProgress(coloring)
        self.progress_bar = progress_bar
        self.equal_weights = equal_weights

    def on_train_begin(self, logs: Dict) -> None:
        self.metrics = ['loss'] + self.model.metrics_names
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']

        if self.progress_bar and self.steps is not None:
            self.color_progress.set_progress_bar(self.steps)

    def on_epoch_begin(self, epoch_number: int, logs: Dict) -> None:
        self.step_times_weighted_sum = 0.
        self.epoch_number = epoch_number

        self.color_progress.on_epoch_begin(self.epoch_number, self.epochs)

    def on_epoch_end(self, epoch_number: int, logs: Dict) -> None:
        epoch_total_time = logs['time']

        metrics_str = self._get_metrics_string(logs)
        if self.steps is not None:
            self.color_progress.on_epoch_end(epoch_total_time, self.steps, metrics_str)
        else:
            self.color_progress.on_epoch_end(epoch_total_time, self.last_step, metrics_str)

    def on_train_batch_end(self, batch_number: int, logs: Dict) -> None:
        if self.equal_weights:
            self.step_times_weighted_sum += logs['time']
            step_times_rate = self.step_times_weighted_sum / batch_number
        else:
            self.step_times_weighted_sum += batch_number * logs['time']
            normalizing_factor = (batch_number * (batch_number + 1)) / 2
            step_times_rate = self.step_times_weighted_sum / normalizing_factor

        metrics_str = self._get_metrics_string(logs)

        if self.steps is not None:
            remaining_time = step_times_rate * (self.steps - batch_number)
            self.color_progress.on_train_batch_end(remaining_time, batch_number, metrics_str, self.steps)
        else:
            self.color_progress.on_train_batch_end(step_times_rate, batch_number, metrics_str)
            self.last_step = batch_number

    def on_test_begin(self, logs: Dict) -> None:
        self.step_times_weighted_sum = 0.

        self.metrics = ['loss'] + self.model.metrics_names
        self.steps = self.params['steps']

        if self.progress_bar and self.steps is not None:
            self.color_progress.set_progress_bar(self.steps)

    def on_test_batch_end(self, batch_number: int, logs: Dict) -> None:
        if self.equal_weights:
            self.step_times_weighted_sum += logs['time']
            step_times_rate = self.step_times_weighted_sum / batch_number
        else:
            self.step_times_weighted_sum += batch_number * logs['time']
            normalizing_factor = (batch_number * (batch_number + 1)) / 2
            step_times_rate = self.step_times_weighted_sum / normalizing_factor

        metrics_str = self._get_metrics_string(logs)

        if self.steps is not None:
            remaining_time = step_times_rate * (self.steps - batch_number)
            self.color_progress.on_test_batch_end(remaining_time, batch_number, metrics_str, self.steps)
        else:
            self.color_progress.on_test_batch_end(step_times_rate, batch_number, metrics_str)
            self.last_step = batch_number

    def on_test_end(self, logs: Dict) -> None:
        print("")  # To clean the sys.stdout end

    def _get_metrics_string(self, logs: Dict):
        train_metrics_str_gen = ('{}: {:f}'.format(k, logs[k]) for k in self.metrics if logs.get(k) is not None)
        val_metrics_str_gen = ('{}: {:f}'.format('val_' + k, logs['val_' + k]) for k in self.metrics
                               if logs.get('val_' + k) is not None)
        return ', '.join(itertools.chain(train_metrics_str_gen, val_metrics_str_gen))
