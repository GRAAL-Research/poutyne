import sys
import warnings
from typing import Dict


class EmptyStringAttrClass:
    """
    Class to emulate the Fore and Style class of colorama with a class that as an empty string for every attributes.
    """

    def __getattr__(self, attr):
        return ''


try:
    from colorama import Fore, Style, init

    colorama = True

    init()
except ModuleNotFoundError:
    colorama = None

    Fore = EmptyStringAttrClass()
    Style = EmptyStringAttrClass()

default_color_settings = {"display_name": None,
                          "text_color": 'LIGHTYELLOW_EX',
                          "ratio_color": "LIGHTBLUE_EX",
                          "metric_value_color": "LIGHTCYAN_EX",
                          "time_color": "GREEN"}


class ColorProgress:
    def __init__(self, coloring):
        color_settings = None
        if isinstance(coloring, Dict):
            if colorama is None:
                warnings.warn("The colorama package was not imported. Consider installing it for colorlog.",
                              ImportWarning)
            color_settings = coloring
        elif coloring:
            if colorama is None:
                warnings.warn("The colorama package was not imported. Consider installing it for colorlog.",
                              ImportWarning)
            color_settings = default_color_settings

        if color_settings is not None:
            self.text_color = getattr(Fore, color_settings.get("text_color"))
            self.ratio_color = getattr(Fore, color_settings.get("ratio_color"))
            self.metric_value_color = getattr(Fore, color_settings.get("metric_value_color"))
            self.time_color = getattr(Fore, color_settings.get("time_color"))
        else:
            self.text_color = ""
            self.ratio_color = ""
            self.metric_value_color = ""
            self.time_color = ""

    def on_epoch_begin(self, epoch_number, epochs):
        sys.stdout.write(self._epoch_format(epoch_number, epochs) + Style.RESET_ALL )
        sys.stdout.flush()

    def on_epoch_end(self, epoch_number, epochs, epoch_total_time, steps, metrics_str):
        print(
            self._epoch_format(epoch_number, epochs) + self._epoch_total_time(epoch_total_time) + self._step_format(
                steps, steps) + self._metric_format(metrics_str) + Style.RESET_ALL)
        sys.stdout.flush()

    def on_train_batch_end_steps(self, epoch_number, epochs, remaining_time, batch_number, steps, metrics_str):
        sys.stdout.write(
            self._epoch_format(epoch_number, epochs) + self._ETA_format(remaining_time) + self._step_format(
                batch_number, steps) + self._metric_format(metrics_str) + Style.RESET_ALL)
        sys.stdout.flush()

    def on_train_batch_end(self, epoch_number, epochs, times_mean, batch_number, metrics_str):
        sys.stdout.write(
            self._epoch_format(epoch_number, epochs) + self._ETA_format(times_mean) + self._step_format(
                batch_number) + self._metric_format(metrics_str) + Style.RESET_ALL)
        sys.stdout.flush()

    def _epoch_format(self, epoch_number, epochs) -> str:
        return self.text_color + "\rEpoch " + self.ratio_color + "%d/%d " % (
            epoch_number, epochs)

    def _epoch_total_time(self, epoch_total_time):
        return self.time_color + "%.2fs " % epoch_total_time

    def _ETA_format(self, time) -> str:
        return self.text_color + "ETA " + self.time_color + "%.0fs " % time

    def _step_format(self, batch_number, steps=None):
        formatted_step = ""
        if steps is None:
            pass
        else:
            formatted_step = self.text_color + "Step " + self.ratio_color + "%d/%d: " % (batch_number, steps)
        return formatted_step

    def _metric_format(self, metrics_str):
        formatted_metrics = ""
        for metric in metrics_str.split(","):
            name_value = metric.split(":")
            name = name_value[0]
            value = name_value[1]
            formatted_metrics += self.text_color + name + ": " + self.metric_value_color + value + " "
        return formatted_metrics
