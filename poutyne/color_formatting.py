import sys
import warnings
from typing import Dict, Union


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
except ImportError:
    colorama = None

    Fore = EmptyStringAttrClass()
    Style = EmptyStringAttrClass()

default_color_settings = {
    "text_color": 'LIGHTYELLOW_EX',
    "ratio_color": "LIGHTBLUE_EX",
    "metric_value_color": "LIGHTCYAN_EX",
    "time_color": "GREEN"
}


class ColorProgress:
    def __init__(self, coloring: Union[bool, Dict]):
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

    def on_epoch_begin(self, epoch_number: int, epochs: int) -> None:
        sys.stdout.write(self._epoch_formatting(epoch_number, epochs) + Style.RESET_ALL)
        sys.stdout.flush()

    def on_epoch_end(self, epoch_number: int, epochs: int, epoch_total_time: float, steps: int,
                     metrics_str: str) -> None:
        # pylint: disable=too-many-arguments
        print(
            self._epoch_formatting(epoch_number, epochs) + self._epoch_total_time_formatting(epoch_total_time) +
            self._step_formatting(steps, steps) + self._metric_formatting(metrics_str) + Style.RESET_ALL)

    def on_train_batch_end_steps(self, epoch_number: int, epochs: int, remaining_time: float, batch_number: int,
                                 steps: int, metrics_str: str) -> None:
        # pylint: disable=too-many-arguments
        sys.stdout.write(
            self._epoch_formatting(epoch_number, epochs) + self._ETA_formatting(remaining_time) +
            self._step_formatting(batch_number, steps) + self._metric_formatting(metrics_str) + Style.RESET_ALL)
        sys.stdout.flush()

    def on_train_batch_end(self, epoch_number: int, epochs: int, times_mean: float, batch_number: int,
                           metrics_str: str) -> None:
        # pylint: disable=too-many-arguments
        sys.stdout.write(
            self._epoch_formatting(epoch_number, epochs) + self._ETA_formatting(times_mean) +
            self._step_formatting(batch_number) + self._metric_formatting(metrics_str) + Style.RESET_ALL)
        sys.stdout.flush()

    def _epoch_formatting(self, epoch_number: int, epochs: int) -> str:
        return self.text_color + "\rEpoch " + self.ratio_color + "%d/%d " % (epoch_number, epochs)

    def _epoch_total_time_formatting(self, epoch_total_time: float) -> str:
        return self.time_color + "%.2fs " % epoch_total_time

    def _ETA_formatting(self, time: float) -> str:
        return self.text_color + "ETA " + self.time_color + "%.0fs " % time

    def _step_formatting(self, batch_number: int, steps: Union[int, None] = None) -> str:
        if steps is None:
            formatted_step = self.text_color + "Step " + self.ratio_color + "%d: " % batch_number
        else:
            formatted_step = self.text_color + "Step " + self.ratio_color + "%d/%d: " % (batch_number, steps)
        return formatted_step

    def _metric_formatting(self, metrics_str: str) -> str:
        formatted_metrics = ""
        for metric in metrics_str.split(","):
            name_value = metric.split(":")
            name = name_value[0]
            value = name_value[1]
            formatted_metrics += self.text_color + name + ": " + self.metric_value_color + value + " "
        return formatted_metrics
